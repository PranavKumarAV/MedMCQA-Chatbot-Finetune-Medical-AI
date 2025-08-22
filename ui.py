# ui.py
import json
import streamlit as st
from config import MODEL

def render_header():
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
          <div style="display:flex;align-items:center;gap:.6rem;">
            <span style="font-size:1.8rem;">💬 Chatbot</span>
            <span style="font-size:.85rem;padding:.15rem .5rem;border:1px solid #ddd;border-radius:999px;color:#555;">
              model: <b>""" + MODEL + """</b>
            </span>
          </div>
        </div>
        <hr style="margin-top:.2rem;margin-bottom:1rem;border:none;border-top:1px solid #eee;" />
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    st.subheader("Session")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("__last_trim_info", None)
        st.session_state.pop("__last_metrics", None)
        st.session_state.pop("full_message", None)
        st.rerun()

    st.markdown("---")
    st.subheader("Options")
    use_system = st.checkbox("Use system prompt", value=False)
    prompt_value = """You are a medical expert. Answer the MCQ and briefly justify in 3–6 sentences.

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

    Answer: 
    """

    system_prompt_text = st.text_area(
        "System prompt",
        value=prompt_value,
        height=120,
        disabled=not use_system,
    )

    st.markdown("---")
    st.subheader("Transcript")
    export = [{"role": m["role"], "content": m["content"]} for m in st.session_state.get("messages", [])]
    st.download_button(
        "⬇️ Download chat.json",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="chat.json",
        mime="application/json",
        use_container_width=True,
    )

    return use_system, system_prompt_text

def render_chat_history():
    chat_area = st.container()
    with chat_area:
        for m in st.session_state.get("messages", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
