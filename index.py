import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# 페이지 설정
st.set_page_config(page_title="Multi-LLM Chatbot", layout="centered")

st.title("🤖 Multi-LLM Chatbot")
st.write("Gemini 또는 OpenAI 모델을 선택해 대화해보세요.")

# 사이드바에서 API 키 입력
with st.sidebar:
    st.header("🔑 API Key 입력")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", type="password")

    st.header("⚙️ 모델 선택")
    provider = st.selectbox("모델 제공자 선택", ["OpenAI", "Gemini"])

    if provider == "OpenAI":
        model_name = st.selectbox("모델", ["gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-4.1-2025-04-14"])
    else:
        model_name = st.selectbox("모델", ["gemini-2.5-flash", "gemini-2.5-pro"])

# 대화 상태 초기화 (멀티턴 캐싱)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 UI
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 사용자 메시지 저장
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # OpenAI 응답
    if provider == "OpenAI" and openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=st.session_state["messages"]
        )
        answer = response.choices[0].message.content

    # Gemini 응답
    elif provider == "Gemini" and gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        # Gemini는 messages 형태가 아니라, history를 직접 전달해야 함
        chat = model.start_chat(history=[
            {"role": m["role"], "parts": [m["content"]]} for m in st.session_state["messages"]
        ])
        response = chat.send_message(user_input)
        answer = response.text

    else:
        answer = "⚠️ API 키를 입력해주세요."

    # AI 응답 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# 대화 기록 출력 (멀티턴)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
