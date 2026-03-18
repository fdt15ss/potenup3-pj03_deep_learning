import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

from ultralytics.models.sam import SAM3SemanticPredictor

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(page_title="SAM 3 Segmentation 체험", layout="wide")

st.title("📌 SAM 3 Object Segmentation 체험")

st.markdown("""
## SAM 3 (Segment Anything Model)

텍스트로 원하는 객체를 입력하면  
해당 객체를 자동으로 분할합니다.

👉 여러 객체를 추가해서 동시에 탐지해보세요.
""")

# -------------------------
# 이미지 저장 폴더 생성
# -------------------------
IMAGE_DIR = "./images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# -------------------------
# 모델 로드
# -------------------------
@st.cache_resource
def load_model():
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="../models/sam3.pt",
        half=True,
        save=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)

predictor = load_model()

# -------------------------
# 세션 상태 초기화
# -------------------------
if "prompts" not in st.session_state:
    st.session_state.prompts = [""]

# -------------------------
# 입력 UI
# -------------------------
st.subheader("🔤 탐지할 객체 입력")

for i in range(len(st.session_state.prompts)):
    st.session_state.prompts[i] = st.text_input(
        f"객체 {i+1}",
        value=st.session_state.prompts[i],
        key=f"prompt_{i}"
    )

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("➕ 객체 추가"):
        st.session_state.prompts.append("")
        st.rerun()

with col_btn2:
    if st.button("➖ 객체 제거"):
        if len(st.session_state.prompts) > 1:
            st.session_state.prompts.pop()
            st.rerun()

# -------------------------
# 파일 업로드
# -------------------------
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

# -------------------------
# 실행
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 🔥 파일 저장 (덮어쓰기 방지 위해 이름 유지)
    save_path = os.path.join(IMAGE_DIR, uploaded_file.name)

    # 같은 이름 있으면 덮어쓰기 방지 (옵션)
    base, ext = os.path.splitext(uploaded_file.name)
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(IMAGE_DIR, f"{base}_{counter}{ext}")
        counter += 1

    image.save(save_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    if st.button("🔍 Segmentation 실행"):
        text_prompts = [p for p in st.session_state.prompts if p.strip() != ""]

        if len(text_prompts) == 0:
            st.warning("최소 하나의 객체를 입력하세요!")
        else:
            with st.spinner("SAM 3 추론 중..."):

                # 🔥 저장된 경로 사용
                predictor.set_image(save_path)

                results = predictor(text=text_prompts)

                result_img = results[0].plot()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Segmentation 결과")
                st.image(result_img, use_container_width=True)