# hf_backend.py
import time
import threading
import torch
from typing import Dict, Iterator, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# Simple cache so we only load once
_model_cache = {"model": None, "tok": None, "base": None, "adapter": None, "fourbit": None}

def load_hf(base_id: str, adapter_path: str, load_in_4bit: bool):
    """Load base + LoRA once and cache."""
    if (_model_cache["model"] is not None and
        _model_cache["base"] == base_id and
        _model_cache["adapter"] == adapter_path and
        _model_cache["fourbit"] == load_in_4bit):
        return _model_cache["model"], _model_cache["tok"]

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = dict(device_map="auto", torch_dtype=dtype)

    if load_in_4bit:
        # requires bitsandbytes installed
        kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ))

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)

    # decoder-only best practices
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    _model_cache.update({"model": model, "tok": tok, "base": base_id, "adapter": adapter_path, "fourbit": load_in_4bit})
    return model, tok

def _build_chat_text(tok, messages):
    """Use the model chat template."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def stream_generate(messages, *, base_id: str, adapter_path: str,
                    load_in_4bit: bool, max_new_tokens: int) -> Iterator[str]:
    """
    Stream tokens using HF TextIteratorStreamer. Yields text chunks.
    Stores simple metrics in a dict returned by get_last_metrics().
    """
    model, tok = load_hf(base_id, adapter_path, load_in_4bit)

    prompt_text = _build_chat_text(tok, messages)
    inputs = tok([prompt_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    # metrics we’ll fill
    metrics = {"prompt_tokens": int(inputs["input_ids"].shape[1]),
               "gen_tokens": 0, "ttft_ms": None, "gen_ms": None, "wall_s": None}
    t0 = time.perf_counter()
    first_token_time = [None]

    def _gen():
        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0, top_p=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

    thread = threading.Thread(target=_gen)
    thread.start()

    produced = []
    for chunk in streamer:
        if first_token_time[0] is None:
            first_token_time[0] = time.perf_counter()
        produced.append(chunk)
        yield chunk

    thread.join()
    t1 = time.perf_counter()
    if first_token_time[0] is not None:
        metrics["ttft_ms"] = (first_token_time[0] - t0) * 1000.0
        metrics["gen_ms"]  = (t1 - first_token_time[0]) * 1000.0
    metrics["wall_s"] = t1 - t0

    # rough output token count from final text
    final_text = "".join(produced)
    metrics["gen_tokens"] = len(tok(final_text, add_special_tokens=False).input_ids)

    _last_metrics.update(metrics)

_last_metrics: Dict[str, float] = {}
def get_last_metrics() -> Dict[str, float]:
    return dict(_last_metrics)
