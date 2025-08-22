# main.py
import time
import streamlit as st
from config import PAGE_TITLE, PAGE_ICON
from ui import render_header, render_sidebar, render_chat_history
from chat_core import generate_response, chat_once_fallback

# ---- Page config ----
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ---- Header ----
render_header()

# ---- Sidebar controls ----
with st.sidebar:
    use_system, system_prompt_text = render_sidebar()

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render history ----
render_chat_history()

# ---- Input & response (unchanged behavior) ----
prompt = st.chat_input("Type your message…")
if prompt:
    # User turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant turn
    t0 = time.perf_counter()
    st.session_state["full_message"] = ""
    with st.chat_message("assistant"):
        # Trim indicator (from any previous prep; will refresh during generate)
        before_after = st.session_state.get("__last_trim_info")
        if before_after and before_after[0] > before_after[1]:
            st.caption(f"Context trimmed: {before_after[0]} → {before_after[1]} chars")

        # Stream
        stream = generate_response(use_system=use_system, system_prompt_text=system_prompt_text)
        streamed = st.write_stream(stream)  # may return None/[]

        assistant_text = streamed if isinstance(streamed, str) and streamed.strip() \
                         else st.session_state.get("full_message", "").strip()

        # Fallback if stream produced nothing
        if not assistant_text:
            txt, _ = chat_once_fallback(use_system=use_system, system_prompt_text=system_prompt_text)
            assistant_text = txt or "_(no response)_"
            st.markdown(assistant_text)

        # Elapsed
        elapsed = time.perf_counter() - t0
        st.caption(f"{len(assistant_text)} chars • {elapsed:.2f}s")

        # Token/throughput metrics if we have them
        m = st.session_state.get("__last_metrics") or {}
        toks_in, toks_out = m.get("prompt_eval_count"), m.get("eval_count")
        pe_ms = (m.get("prompt_eval_duration") or 0) / 1e6 if m.get("prompt_eval_duration") else None
        gen_ms = (m.get("eval_duration") or 0) / 1e6 if m.get("eval_duration") else None
        if toks_in is not None and toks_out is not None:
            extra = []
            if pe_ms is not None:  extra.append(f"TTFT {pe_ms:.0f} ms")
            if gen_ms is not None: extra.append(f"gen {gen_ms:.0f} ms")
            tail = f" • {' • '.join(extra)}" if extra else ""
            st.caption(f"{toks_in}/{toks_out} tokens{tail}")

    # Save assistant turn
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
