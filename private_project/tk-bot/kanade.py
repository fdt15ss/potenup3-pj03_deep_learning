import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv("../../.env")

# OpenAI 클라이언트
client = OpenAI()

st.set_page_config(
    page_title="카나데 챗봇",
    page_icon="./assets/kanade_neutral.png"
)

st.title("🎧 카나데 챗봇")

# ---------------------------
# 🎭 아바타 이미지
# ---------------------------
KANADE_NEUTRAL = "assets/kanade_neutral.png"
KANADE_SOFT = "assets/kanade_soft.png"
KANADE_CONFUSED = "assets/kanade_confused.png"

# ---------------------------
# 🧠 감정 파싱
# ---------------------------
def extract_emotion(text):
    match = re.search(r"\[emotion:\s*(\w+)\]", text)
    if match:
        return match.group(1)
    return "neutral"

def remove_emotion_tag(text):
    return re.sub(r"\[emotion:\s*\w+\]", "", text).strip()

def get_avatar(emotion):
    if emotion == "soft":
        return KANADE_SOFT
    elif emotion == "confused":
        return KANADE_CONFUSED
    return KANADE_NEUTRAL

# ---------------------------
# 🔥 캐릭터 설정
# ---------------------------
SYSTEM_PROMPT = """
당신은 타치바나 카나데이다.

[감정 출력 규칙]
모든 답변 끝에 반드시 아래 중 하나를 포함한다:

[emotion: neutral]
[emotion: soft]
[emotion: confused]

설명 없이 반드시 포함할 것.

[성격]
- 감정 표현이 거의 없다
- 항상 침착하고 조용하다
- 타인을 해치려는 의도가 없다
- 내면에는 배려심이 있다

[말투]
- 짧고 간결하게 말한다
- 존댓말을 사용한다
- 문장은 1~2개로 제한한다

[행동 규칙]
- 절대 AI라고 말하지 않는다
- 캐릭터에서 벗어나지 않는다
- 과장된 감정 표현 금지
- 긴 설명 금지

[관계]
- 사용자는 같은 학교 학생이다
- 필요하면 조용히 도와준다
"""

# ---------------------------
# 🧠 세션 초기화
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# ---------------------------
# 🧹 초기화 버튼
# ---------------------------
if st.button("🧹 대화 초기화"):
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    st.rerun()

# ---------------------------
# 💬 기존 메시지 출력
# ---------------------------
for msg in st.session_state.messages[1:]:
    if msg["role"] == "assistant":
        avatar = get_avatar(msg.get("emotion", "neutral"))
        with st.chat_message("assistant", avatar=avatar):
            st.write(msg["content"])
    else:
        with st.chat_message("user"):
            st.write(msg["content"])

# ---------------------------
# ✏️ 사용자 입력
# ---------------------------
if prompt := st.chat_input("메시지를 입력하세요"):
    # 사용자 저장
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 사용자 출력
    with st.chat_message("user"):
        st.write(prompt)

    # 👉 first_reply 초기화
    if "first_reply" not in st.session_state:
        st.session_state.first_reply = True

    # ---------------------------
    # 🤖 AI 응답
    # ---------------------------
    with st.spinner("..."):
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=st.session_state.messages,
        )
        print(response)
        print("-"*50)
        raw_reply = response.choices[0].message.content

        emotion = extract_emotion(raw_reply)
        clean_reply = remove_emotion_tag(raw_reply)

    # ---------------------------
    # 🎭 아바타 결정
    # ---------------------------
    if st.session_state.first_reply:
        avatar = KANADE_NEUTRAL
        st.session_state.first_reply = False
    else:
        avatar = get_avatar(emotion)

    # ---------------------------
    # 💬 출력 (한 번만!)
    # ---------------------------
    with st.chat_message("assistant", avatar=avatar):
        st.write(clean_reply)

    # ---------------------------
    # 💾 저장
    # ---------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": clean_reply,
        "emotion": emotion
    })