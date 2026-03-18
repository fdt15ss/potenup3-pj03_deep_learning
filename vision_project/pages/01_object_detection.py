# ChatGPT를 활용해서 업데이트한 결과
# 페이지 목표
# title: Object Detection 체험하기
# markdown ## YOLO
# markdwon YOLO 설명 
# 파일 업로더
# 추출하기 버튼
# 버튼을 누르면 yolo로 이미지 예측해서 결과 반환하기

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests

FASTAPI_URL = "http://localhost:8080"


# 모델 로드
model = YOLO("../models/yolo26n.pt")

st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* 제목 스타일 */
h1 {
    text-align: center;
    font-weight: 800;
    font-size: 42px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 카드 스타일 */
[data-testid="stExpander"] {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}

/* 업로드 박스 */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    border: 2px dashed #38bdf8;
    padding: 20px;
    background-color: rgba(56, 189, 248, 0.05);
}

/* 버튼 */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 15px rgba(56,189,248,0.6);
}

/* 결과 카드 */
[data-testid="stVerticalBlock"] > div:has(div[data-testid="stImage"]) {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
}

/* 텍스트 결과 박스 */
[data-testid="stContainer"] {
    border-radius: 16px;
    padding: 15px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

/* 스피너 */
[data-testid="stSpinner"] {
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)


# 페이지 제목
st.title("Object Detection 체험하기")

# 설명
with st.expander(label="모델 카드"):
    st.markdown("#### Object Detection YOLO 모델")
    st.markdown("""
    YOLO(You Only Look Once)는 이미지에서 객체를 실시간으로 탐지하는 딥러닝 모델입니다.  
    이미지를 한 번만 보고 객체의 위치와 종류를 동시에 예측합니다.
    """)

# 모델 설명
with st.expander("📘 모델 설명", expanded=True):
    st.markdown("""
    #### YOLO Object Detection 모델

    - object detection은 이미지를 입력 받아 객체의 위치와 종류를 탐지하는 모델입니다.  
    - 이미지를 입력하면 객체의 bounding box와 클래스 정보를 내뱉습니다.  
    - object detection으로 사람, 동물, 사물 등의 위치와 종류를 탐지할 수 있습니다.  
    - object detection을 학습할 때에는 데이터의 라벨링 품질과 클래스 불균형을 주의해야 합니다.  
    """)


st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>이미지를 업로드하면 AI가 객체를 탐지합니다</p>",
    unsafe_allow_html=True
)

# 파일 업로드)
uploaded_file = st.file_uploader(
    label="이미지를 업로드 하세요",
    type=["jpg", "jpeg", "png"]
)

# 이미지 보여주기
if uploaded_file is not None:
    _, col, _ = st.columns(3)
    with col:
        image = Image.open(uploaded_file)
        st.image(
            image, 
            caption="업로드한 이미지", 
            width="stretch"
        )

# 버튼
predict_button = st.button(
    "예측하기", 
    type="primary",
    width="stretch"
)

# 버튼2
predict_button2 = st.button(
    "Fastapi에서 예측하기", 
    type="secondary",
    width="stretch"
)

# 예측 실행
if predict_button:
    st.markdown("### 🔍 예측 결과")
    if uploaded_file is None:
        st.warning("먼저 이미지를 업로드해주세요!")
    else:
        with st.spinner("🔍 예측 중입니다..."):
            # PIL → numpy 변환
            img_array = np.array(image)

            # YOLO 예측
            results = model(img_array)

            # 결과 이미지 (bounding box 포함)
            result_img = results[0].plot()

        # 결과 출력 (spinner 밖)
        _, col, _ = st.columns(3)
        col.image(
            result_img, 
            caption="예측 결과", 
            width="stretch"
        )

        # 텍스트 결과 출력
        st.markdown("### 📊 탐지 결과")
        with st.container(border=True):
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                # 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                st.write(
                    f"{label} | 신뢰도: {conf:.2f} | 좌표: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                )

# UI가 확정되었다면 이제 구조를 그대로 유지하고 예쁘게 꾸며달라고 요청하기
if predict_button2:
    st.markdown("### 🔍 예측 결과")
    if uploaded_file is None:
        st.warning("먼저 이미지를 업로드해주세요!")
    else:
        detect_image_url = f"{FASTAPI_URL}/detect_image"
        with st.spinner("🔍 예측 중입니다..."):
            # 이미지 가져오기
            
            # f는 이미지를 바이너리 형태로 불러오는 것이다.
            files = {
                "file": (
                    uploaded_file.name, 
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(url=detect_image_url, files=files)
            # if response.status_code == 200:
            st.write(response.json())