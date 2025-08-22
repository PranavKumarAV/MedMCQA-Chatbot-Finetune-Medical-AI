# chat_core.py
import streamlit as st
import ollama
from config import MODEL, MAX_TURNS, BUDGET_CHARS, BACKEND, HF_BASE_ID, HF_ADAPTER_PATH, HF_LOAD_IN_4BIT, HF_MAX_NEW_TOKENS
from hf_backend import stream_generate, get_last_metrics

def _cap_turns(messages):
    """Keep only the last MAX_TURNS user+assistant turns (preserve optional system at front)."""
    if not messages:
        return messages
    head = messages[:1] if messages[0].get("role") == "system" else []
    tail = messages[1:] if head else messages[:]
    if len(tail) <= 2 * MAX_TURNS:
        return head + tail
    return head + tail[-2 * MAX_TURNS :]

def _trim_for_model(messages, budget_chars: int = BUDGET_CHARS):
    """
    Returns (trimmed_messages, before_chars, after_chars).
    Char-based safety trim to prevent huge contexts.
    """
    def total_chars(ms):
        return sum(len(m.get("content", "")) for m in ms)

    before = total_chars(messages)
    if before <= budget_chars:
        return messages, before, before

    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    tail = messages[1:] if head else messages[:]

    while tail and (total_chars(head + tail) > budget_chars):
        tail = tail[2:] if len(tail) >= 2 else tail[1:]

    trimmed = head + tail
    after = total_chars(trimmed)
    return trimmed, before, after

def _apply_system(messages, use_sys: bool, sys_text: str):
    """Prepend system message (not shown in chat) if enabled."""
    if not use_sys or not sys_text.strip():
        return list(messages)
    sys_msg = {"role": "system", "content": sys_text.strip()}
    non_sys = [m for m in messages if m.get("role") != "system"]
    return [sys_msg] + non_sys

def _prepare_for_model(use_system: bool, system_prompt_text: str):
    """
    Apply (optional) system, cap turns, then trim; store trim stats for display.
    Returns the prepared message list to send to the model.
    """
    msgs = _apply_system(st.session_state.messages, use_system, system_prompt_text)
    msgs = _cap_turns(msgs)
    trimmed, before, after = _trim_for_model(msgs)
    st.session_state["__last_trim_info"] = (before, after)
    return trimmed

def _save_metrics_from_chunk(chunk):
    """Extract token/time metrics from a streaming chunk (dict or object)."""
    try:
        if isinstance(chunk, dict):
            if chunk.get("done"):
                st.session_state["__last_metrics"] = {
                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                    "eval_count": chunk.get("eval_count"),
                    "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                    "eval_duration": chunk.get("eval_duration"),
                    "total_duration": chunk.get("total_duration"),
                }
        else:
            if getattr(chunk, "done", False):
                st.session_state["__last_metrics"] = {
                    "prompt_eval_count": getattr(chunk, "prompt_eval_count", None),
                    "eval_count": getattr(chunk, "eval_count", None),
                    "prompt_eval_duration": getattr(chunk, "prompt_eval_duration", None),
                    "eval_duration": getattr(chunk, "eval_duration", None),
                    "total_duration": getattr(chunk, "total_duration", None),
                }
    except Exception:
        pass

def _ollama_stream(to_send):
    response = ollama.chat(model=MODEL, stream=True, messages=to_send)
    for chunk in response:
        _save_metrics_from_chunk(chunk)  # your existing function
        token = ""
        if isinstance(chunk, dict):
            msg = chunk.get("message")
            if isinstance(msg, dict):
                token = msg.get("content", "")
            if not token:
                token = chunk.get("response", "")
        else:
            msg = getattr(chunk, "message", None)
            if msg is not None:
                if isinstance(msg, dict):
                    token = msg.get("content", "")
                else:
                    token = getattr(msg, "content", "") or ""
            if not token:
                token = getattr(chunk, "response", "") or ""
        if token:
            st.session_state["full_message"] += token
            yield token

def _ollama_once(to_send):
    resp = ollama.chat(model=MODEL, stream=False, messages=to_send)
    text, metrics = "", {}
    if isinstance(resp, dict):
        msg = resp.get("message")
        if isinstance(msg, dict) and "content" in msg:
            text = msg["content"]
        elif "response" in resp:
            text = resp["response"]
        metrics = {
            "prompt_eval_count": resp.get("prompt_eval_count"),
            "eval_count": resp.get("eval_count"),
            "prompt_eval_duration": resp.get("prompt_eval_duration"),
            "eval_duration": resp.get("eval_duration"),
            "total_duration": resp.get("total_duration"),
        }
    else:
        msg = getattr(resp, "message", None)
        if msg is not None:
            if isinstance(msg, dict):
                text = msg.get("content", "")
            else:
                text = getattr(msg, "content", "") or ""
        if not text:
            text = getattr(resp, "response", "") or ""
        metrics = {
            "prompt_eval_count": getattr(resp, "prompt_eval_count", None),
            "eval_count": getattr(resp, "eval_count", None),
            "prompt_eval_duration": getattr(resp, "prompt_eval_duration", None),
            "eval_duration": getattr(resp, "eval_duration", None),
            "total_duration": getattr(resp, "total_duration", None),
        }
    st.session_state["__last_metrics"] = metrics
    return text, metrics

def _hf_stream(to_send):
    # to_send is a chat list; stream_generate expects same
    for chunk in stream_generate(
        to_send,
        base_id=HF_BASE_ID,
        adapter_path=HF_ADAPTER_PATH,
        load_in_4bit=HF_LOAD_IN_4BIT,
        max_new_tokens=HF_MAX_NEW_TOKENS,
    ):
        st.session_state["full_message"] += chunk
        yield chunk
    # store metrics in the same key shape expected by your UI
    m = get_last_metrics()
    st.session_state["__last_metrics"] = {
        "prompt_eval_count": m.get("prompt_tokens"),
        "eval_count": m.get("gen_tokens"),
        "prompt_eval_duration": int((m.get("ttft_ms") or 0) * 1e6),
        "eval_duration": int((m.get("gen_ms") or 0) * 1e6),
        "total_duration": int((m.get("wall_s") or 0) * 1e9),
    }

def _hf_once(to_send):
    # Non-stream fallback for HF: just run a single generate call and return text+metrics.
    # We can reuse the streaming path and collect the text.
    buf = []
    for chunk in _hf_stream(to_send):
        buf.append(chunk)
    return "".join(buf), get_last_metrics()

# === Public functions used by main.py ===
def generate_response(use_system: bool, system_prompt_text: str):
    to_send = _prepare_for_model(use_system, system_prompt_text)
    if BACKEND == "ollama":
        return _ollama_stream(to_send)
    else:
        return _hf_stream(to_send)

def chat_once_fallback(use_system: bool, system_prompt_text: str):
    to_send = _prepare_for_model(use_system, system_prompt_text)
    if BACKEND == "ollama":
        return _ollama_once(to_send)
    else:
        return _hf_once(to_send)
