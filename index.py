import os
from uuid import uuid4

import streamlit as st

# LangChain (0.2+ 권장)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# -----------------------------
# 기본 환경 세팅 / 키 로드
# -----------------------------
st.set_page_config(page_title="LangChain Multi-turn Chatbot", page_icon="🤖", layout="centered")

if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
    st.stop()  # 안전하게 중단
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# -----------------------------
# UI 사이드바
# -----------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    default_model = st.secrets["openai"].get("model", "gpt-4o-mini")
    model_name = st.selectbox("OpenAI 모델", [default_model, "gpt-4o-mini", "gpt-4.1", "gpt-5-mini-2025-08-07"], index=0)
    temperature = st.slider("창의성 (temperature)", 0.0, 1.2, 0.7, 0.1)
    sys_prompt = st.text_area(
        "System Prompt",
        value=(
            "당신은 정확하고 친절한 한국어 AI 비서입니다. "
            "사실 확인을 중시하며, 코드/예시는 실행 가능하게 간결히 제시하세요."
        ),
        height=120,
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        clear_btn = st.button("🧹 대화 초기화", use_container_width=True)
    with col_b:
        new_sess_btn = st.button("🔄 새 세션 ID", use_container_width=True)

    with st.container(border=True):
        st.write("문의/건의/피드백 사항이 있다면 개발자에게 알려주세요!\n바로바로 고치거나 추가하도록 하겠습니다!")
        st.badge("scy0416@gmail.com", icon=":material/mail:", color="green")

# -----------------------------
# 세션/히스토리 저장용 상태
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "stores" not in st.session_state:
    # 여러 세션을 동시에 관리하고 싶을 수 있어 dict로 보관
    st.session_state.stores = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """LangChain이 호출하는 히스토리 팩토리 함수"""
    store = st.session_state.stores
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 버튼 동작
if clear_btn:
    st.session_state.stores[st.session_state.session_id] = ChatMessageHistory()
if new_sess_btn:
    st.session_state.session_id = str(uuid4())

# -----------------------------
# LangChain 구성
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    streaming=True,  # 스트리밍 활성화
)

# 프롬프트 → LLM 파이프라인
chain = prompt | llm

# 멀티턴 히스토리 래퍼
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,           # 히스토리 공급 함수
    input_messages_key="input",    # 입력 필드명
    history_messages_key="history" # 히스토리 placeholder 키
    # output_messages_key 기본값 "output" (ChatModel을 쓰면 자동으로 AI 메시지로 기록됩니다)
)

# -----------------------------
# 상단 UI
# -----------------------------
st.title("🤖 LangChain 멀티턴 챗봇 (Streamlit + OpenAI)")
st.caption(f"세션 ID: `{st.session_state.session_id}`")

# 지금까지의 히스토리 보여주기 (새로고침 시 재렌더)
history = get_session_history(st.session_state.session_id)
for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# -----------------------------
# 입력 & 스트리밍 응답
# -----------------------------
user_input = st.chat_input("메시지를 입력하세요…")

def as_text_stream(chunks):
    """LangChain의 AIMessageChunk 스트림 → 텍스트 스트림으로 변환"""
    for chunk in chunks:
        content = getattr(chunk, "content", None)
        if content:
            yield content

if user_input:
    # 먼저 사용자의 말 풍선 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    # 어시스턴트 말풍선에 스트리밍 출력
    with st.chat_message("assistant"):
        cfg = {"configurable": {"session_id": st.session_state.session_id}}
        stream = conversational_chain.stream(
            {"system_prompt": sys_prompt, "input": user_input},
            config=cfg,
        )
        st.write_stream(as_text_stream(stream))

    # 스트리밍이 끝나면 LangChain이 자동으로 히스토리에
    # (human → ai) 쌍을 기록합니다. Streamlit은 다음 rerun 때 위에서 다시 그려요.
