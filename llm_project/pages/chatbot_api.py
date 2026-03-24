import streamlit as st
import requests

fastapi_url = "http://localhost:8080/chat"

st.title("챗봇 만들기")

# 저장소 만들기
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for chat in st.session_state["chat_history"]:
    st.chat_message(chat["role"]).markdown(chat["content"])

user_input = st.chat_input(
    placeholder="메세지를 입력하세요"
)

if user_input:
    st.chat_message("user").markdown(user_input)

    # answer = "안녕하세요" # Fastapi /chat 요청으로 대체해보세요
    data = {
        "message": user_input
    }
    try:
        answer = requests.post(
            url=fastapi_url,
            json=data
        ).json()['text']
    except:
        answer = "LLM에 접근할 수 없는 상태입니다."

    st.chat_message("ai").markdown(answer)

    # 데이터 저장
    st.session_state["chat_history"].extend(
        [
            {"role": "user", "content": user_input},
            {"role" : "ai", "content": answer}
        ]
    )