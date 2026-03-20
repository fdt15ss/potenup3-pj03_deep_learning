import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

from db import (
    init_db,
    create_chat,
    save_message,
    get_messages,
    get_last_chat,
    get_all_chats,
    delete_chat,
    update_chat_title,
    generate_title
)

# 🔐 환경변수
load_dotenv("../../.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🗄️ DB
init_db()

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("💬 Chatbot")

# ------------------------
# 🧠 chat_id 초기화
# ------------------------
if "chat_id" not in st.session_state:
    last_chat = get_last_chat()
    st.session_state.chat_id = last_chat if last_chat else create_chat()

if "editing_chat_id" not in st.session_state:
    st.session_state.editing_chat_id = None

# ------------------------
# 📂 사이드바
# ------------------------
st.sidebar.title("📂 채팅 목록")

# ➕ 새 채팅
if st.sidebar.button("➕ 새 채팅"):
    st.session_state.chat_id = create_chat()
    st.session_state.editing_chat_id = None
    st.rerun()

chats = get_all_chats()

for chat_id, created_at, title in chats:
    display_title = title if title else generate_title(chat_id)
    is_selected = chat_id == st.session_state.chat_id

    # ------------------------
    # ✏️ 수정 모드
    # ------------------------
    if st.session_state.editing_chat_id == chat_id:

        new_title = st.sidebar.text_input(
            "제목 수정",
            value=display_title,
            key=f"edit_input_{chat_id}"
        )

        col1, col2 = st.sidebar.columns(2)

        # 💾 저장 버튼
        if col1.button("💾 저장", key=f"save_{chat_id}"):
            update_chat_title(chat_id, new_title)
            st.session_state.editing_chat_id = None
            st.rerun()

        # ❌ 취소 버튼
        if col2.button("취소", key=f"cancel_{chat_id}"):
            st.session_state.editing_chat_id = None
            st.rerun()

    else:
        col1, col2 = st.sidebar.columns([5, 3])

        # 👉 채팅 선택 버튼
        label = f"👉 {display_title}" if is_selected else display_title

        if col1.button(label, key=f"chat_{chat_id}"):
            st.session_state.chat_id = chat_id
            st.rerun()

        # 👉 텍스트 버튼 (아이콘 제거)
        btn_col1, btn_col2 = col2.columns(2)

        # 수정
        if btn_col1.button("수정", key=f"edit_btn_{chat_id}"):
            st.session_state.editing_chat_id = chat_id

        # 삭제
        if btn_col2.button("삭제", key=f"del_{chat_id}"):
            delete_chat(chat_id)

            if chat_id == st.session_state.chat_id:
                new_chat = get_last_chat()
                st.session_state.chat_id = new_chat if new_chat else create_chat()

            st.session_state.editing_chat_id = None
            st.rerun()


# ------------------------
# 💬 메시지 로드
# ------------------------
messages = get_messages(st.session_state.chat_id)

# ------------------------
# 💬 채팅 출력
# ------------------------
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------------
# ✏️ 입력
# ------------------------
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 🔥 1. 먼저 세션에 추가 (안 사라지게)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 화면에 바로 출력
    with st.chat_message("user"):
        st.write(user_input)

    # 3. DB 저장
    save_message(st.session_state.chat_id, "user", user_input)

    # 4. 스피너 + GPT 호출
    with st.spinner("🤖 답변 생성 중..."):
        messages = get_messages(st.session_state.chat_id)

        response = client.responses.create(
            model="gpt-5.4",
            input=messages
        )

        reply = response.output[0].content[0].text

    # 5. assistant 출력
    with st.chat_message("assistant"):
        st.write(reply)

    # 6. DB 저장
    save_message(st.session_state.chat_id, "assistant", reply)

    # 🔥 7. 세션에도 추가 (즉시 반영)
    st.session_state.messages.append({"role": "assistant", "content": reply})