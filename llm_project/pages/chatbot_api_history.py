import streamlit as st 
import requests

fastapi_url = "http://localhost:8888/chat_with_history"

st.title("대화가 반영된 챗봇 만들기")

# 저장소 만들기
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# print(st.session_state["chat_history"])
# 히스토리 출력
for chat in st.session_state["chat_history"]:
    st.chat_message(chat["role"]).markdown(chat["content"])

user_input = st.chat_input(
    placeholder="메세지를 입력하세요"
)

if user_input:
    # 유저 입력 텍스트 표시 
    # st.chat_message("user",avatar="이미지url").markdown(user_input)
    st.chat_message("user").markdown(user_input)
    # 유저 입력 텍스트 chat_history에 저장
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # answer = "안녕하세요" # Fastapi /chat 요청으로 대체해보세요
    data = {
        "history": st.session_state["chat_history"]
    }
    try:
        answer = requests.post(
            url=fastapi_url,
            json=data
        ).json()["text"]
    except:
        answer = "LLM에 접근할 수 없는 상태입니다."
    # AI 대답 텍스트 표시
    st.chat_message("ai").markdown(answer)

    # 히스토리 저장
    st.session_state["chat_history"].append(
        {"role": "ai", "content": answer}
    )