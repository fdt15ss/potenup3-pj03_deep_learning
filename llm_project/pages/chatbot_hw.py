import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv("../.env")
client = OpenAI()

st.set_page_config(page_title="츤데레 챗봇", page_icon="😒")

# 👉 파일 경로
CHAT_FILE = "chat_history.json"

# 👉 챗봇 정체성 표시
st.markdown("""
# 😒 츤데레 챗봇

이 챗봇은 기본적으로 **불친절하고 귀찮아하지만**,  
결국에는 **도움은 주는 츤데레 성격**을 가지고 있습니다.

- 말투: 퉁명스럽고 건방짐
- 태도: 귀찮아하지만 결국 답변함
- 특징: 가끔은 친절한 말이 튀어나옴
""")



# ---------------------------
# 🔥 1. 파일에서 히스토리 불러오기
# ---------------------------
def load_chat():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------------------------
# 🔥 2. 파일에 저장
# ---------------------------
def save_chat(messages):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# ---------------------------
# 세션 상태 초기화
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

# ---------------------------
# 기존 메시지 출력
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------------------
# 사용자 입력
# ---------------------------
if prompt := st.chat_input("뭐 물어볼 건데?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    system_prompt = """
    너는 츤데레 AI다.

    특징:
    - 기본적으로 불친절하고 귀찮아함
    - 하지만 질문에는 반드시 답변함
    - 가끔은 은근히 친절함이 드러남
    - 말투는 짧고 퉁명스럽게
    - 과하게 욕설은 쓰지 말 것
    """

    # 🔥 GPT 호출 (히스토리 포함)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            *st.session_state.messages
        ],
    )

    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 🔥 3. 파일에 저장
    save_chat(st.session_state.messages)

    with st.chat_message("assistant"):
        st.write(reply)

if st.button("🗑️ 대화 초기화"):
    st.session_state.messages = []
    save_chat([])  # 파일도 같이 초기화
    st.rerun()
