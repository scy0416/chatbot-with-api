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
        model_name = st.selectbox(
            "모델",
            ["gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-4.1-2025-04-14"]
        )
    else:
        model_name = st.selectbox(
            "모델",
            ["gemini-2.5-flash", "gemini-2.5-pro"]
        )

# 대화 상태 초기화 (멀티턴 캐싱)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Gemini는 별도 chat 객체 필요
if "gemini_chat" not in st.session_state:
    st.session_state["gemini_chat"] = None

# 채팅 UI
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 사용자 메시지 저장
    st.session_state["messages"].append({"role": "user", "content": user_input})

    answer = None

    # OpenAI 응답
    if provider == "OpenAI" and openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=st.session_state["messages"]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ OpenAI 호출 중 오류 발생: {e}"

    # Gemini 응답
    elif provider == "Gemini" and gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(model_name)

            # Gemini는 role 변환 필요
            gemini_history = []
            for m in st.session_state["messages"]:
                if m["role"] == "user":
                    gemini_history.append({"role": "user", "parts": [m["content"]]})
                else:  # assistant → Gemini에서는 "model"
                    gemini_history.append({"role": "model", "parts": [m["content"]]})

            # chat 객체를 세션에 저장해 멀티턴 유지
            st.session_state["gemini_chat"] = model.start_chat(history=gemini_history)
            response = st.session_state["gemini_chat"].send_message(user_input)
            answer = response.text
        except Exception as e:
            answer = f"⚠️ Gemini 호출 중 오류 발생: {e}"

    else:
        answer = "⚠️ API 키를 입력해주세요."

    # AI 응답 저장
    if answer:
        st.session_state["messages"].append({"role": "assistant", "content": answer})

# 대화 기록 출력 (멀티턴)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
