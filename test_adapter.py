import argparse, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

EXPLAIN_PROMPT = """You are a medical expert. Answer the MCQ and briefly justify in 3–6 sentences.

Question:
{question}

Options:
A. {opa}
B. {opb}
C. {opc}
D. {opd}

Respond in the format:
Answer: <A/B/C/D>
Explanation: <3–6 sentences>
"""

ANSWER_ONLY_PROMPT = """You are a medical expert. Answer this MCQ with a single letter.

Question:
{question}

Options:
A. {opa}
B. {opb}
C. {opc}
D. {opd}

Respond in the format:
Answer: <A/B/C/D>
"""

def build_chat_text(tokenizer, prompt: str):
    """Use the model's chat template so formatting matches the base."""
    messages = [
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": prompt.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def load_base_and_adapter(base_id: str, adapter_path: str, load_in_4bit: bool):
    kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if load_in_4bit:
        kwargs.update(dict(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32))

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)  # apply LoRA

    # decoder-only best practices
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    model.eval()
    return model, tok

def generate(model, tok, prompt_text: str, max_new_tokens=256):
    text = build_chat_text(tok, prompt_text)
    inputs = tok([text], return_tensors="pt").to(model.device)

    # timings (rough, wall-clock)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    t1 = time.perf_counter()

    full = tok.decode(out.sequences[0], skip_special_tokens=True)
    # Extract only the assistant part after the last prompt
    # (simple split; chat templates differ but this is sufficient for quick testing)
    response = full[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

    # crude token counts
    in_tokens  = inputs["input_ids"].shape[1]
    out_tokens = out.sequences.shape[1] - in_tokens
    print(f"\n--- Output ---\n{response}\n")
    print(f"Tokens in/out: {in_tokens}/{out_tokens} • Wall time: {t1 - t0:.2f}s")
    return response

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. meta-llama/Meta-Llama-3-8B-Instruct or Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True, help="Path to your saved LoRA folder (contains adapter_model.safetensors)")
    ap.add_argument("--mode", choices=["answer", "explain"], default="explain",
                    help="answer = letter only; explain = letter + brief explanation")
    ap.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit (bitsandbytes) to reduce VRAM")
    ap.add_argument("--max-new", type=int, default=256)
    # quick demo question (you can pass your own later by editing below)
    ap.add_argument("--question", default="Which one of these is absorbed in ileum?")
    ap.add_argument("--opa", default="Vitamin D")
    ap.add_argument("--opb", default="B12")
    ap.add_argument("--opc", default="Iron")
    ap.add_argument("--opd", default="Fat")
    args = ap.parse_args()

    model, tok = load_base_and_adapter(args.base, args.adapter, args.load_in_4bit)

    if args.mode == "explain":
        prompt = EXPLAIN_PROMPT.format(
            question=args.question, opa=args.opa, opb=args.opb, opc=args.opc, opd=args.opd
        )
    else:
        prompt = ANSWER_ONLY_PROMPT.format(
            question=args.question, opa=args.opa, opb=args.opb, opc=args.opc, opd=args.opd
        )

    generate(model, tok, prompt, max_new_tokens=args.max_new)

if __name__ == "__main__":
    main()
