import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

from db import (
    init_db,
    create_chat,
    save_message,
    get_messages,
    get_last_chat
)

# 🔐 환경변수 로드
load_dotenv("../../.env")

# 🤖 OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🗄️ DB 초기화
init_db()

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("💬 Chatbot (SQLite 저장)")

# ------------------------
# 🧠 chat_id 초기화 (핵심🔥)
# ------------------------
if "chat_id" not in st.session_state:
    last_chat = get_last_chat()

    if last_chat:
        st.session_state.chat_id = last_chat
    else:
        st.session_state.chat_id = create_chat()


# ------------------------
# 🆕 새 채팅 버튼
# ------------------------
if st.button("➕ 새 채팅"):
    st.session_state.chat_id = create_chat()
    st.rerun()


# ------------------------
# 💬 DB 기준 메시지 로드 (항상 최신)
# ------------------------
messages = get_messages(st.session_state.chat_id)


# ------------------------
# 💬 기존 메시지 출력
# ------------------------
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------------
# ✏️ 사용자 입력
# ------------------------
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 1. 화면에 유저 메시지 출력
    with st.chat_message("user"):
        st.write(user_input)

    # 2. DB 저장
    save_message(st.session_state.chat_id, "user", user_input)

    # 3. 최신 대화 다시 가져오기
    messages = get_messages(st.session_state.chat_id)

    # 4. GPT 호출
    response = client.responses.create(
        model="gpt-5.3",
        input=messages
    )

    reply = response.output[0].content[0].text

    # 5. 화면 출력
    with st.chat_message("assistant"):
        st.write(reply)

    # 6. DB 저장
    save_message(st.session_state.chat_id, "assistant", reply)

    # 7. 새로고침 (상태 동기화)
    st.rerun()