import streamlit as st
from src.chatbot import ContextMultilingualChatbot

st.set_page_config(page_title="NLP Multilingual Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Context-Aware Multilingual Chatbot")
st.caption("Mini Project ID: P_097 | NLP & Transformers Intent Engine")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = ContextMultilingualChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    languages = {
        "English": "en",
        "Spanish (Español)": "es",
        "French (Français)": "fr",
        "German (Deutsch)": "de",
        "Hindi (हिंदी)": "hi",
        "Mandarin (中文)": "zh-CN",
        "Japanese (日本語)": "ja"
    }
    selected_lang = st.selectbox("Target Output Language", list(languages.keys()))
    lang_code = languages[selected_lang]

    if st.button("🗑️ Clear Context Memory"):
        st.session_state.messages = []
        st.session_state.chatbot.clear_memory()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "intent" in msg:
            intent_val = msg["intent"]
            conf_val = msg["confidence"]
            st.caption(f"🎯 **Intent Detected**: `{intent_val}` | Confidence: `{conf_val:.2%}`")

if user_input := st.chat_input("Type a message in any language..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Processing intent and generating response..."):
        res = st.session_state.chatbot.process_message(user_input, target_language=lang_code)

    with st.chat_message("assistant"):
        st.write(res["response"])
        intent_val = res["intent"]
        conf_val = res["confidence"]
        st.caption(f"🎯 **Intent Detected**: `{intent_val}` | Confidence: `{conf_val:.2%}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": res["response"],
        "intent": res["intent"],
        "confidence": res["confidence"]
    })
