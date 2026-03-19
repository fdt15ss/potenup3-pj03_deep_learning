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

# 페이지 설정
st.set_page_config(
    page_title="Object Detection 체험하기",
    page_icon="🔍",
    layout="centered"
)

# ✅ 라이트 테마 + 전체 UI 스타일
st.markdown("""
<style>
/* 전체 배경 */
.main {
    background-color: #f8fafc;
}

/* 메인 타이틀 */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 10px;
}

/* 서브 설명 */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #475569;
    margin-bottom: 30px;
}

/* 카드 공통 */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* 결과 카드 hover */
.card:hover {
    transform: translateY(-3px);
    transition: 0.2s;
}

/* 라벨 스타일 */
.label {
    font-weight: 600;
    color: #2563eb;
}

/* confidence */
.conf {
    color: #16a34a;
    font-weight: 600;
}

/* 좌표 */
.coord {
    color: #d97706;
}

/* 버튼 꾸미기 */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* 업로드 박스 */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5f5;
    padding: 20px;
    border-radius: 15px;
    background-color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# 모델 로드
model = YOLO("../models/yolo26n.pt")

# ✅ 타이틀 (HTML)
st.markdown('<div class="main-title">🔍 Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLO 모델로 이미지 속 객체를 탐지해보세요</div>', unsafe_allow_html=True)

# 모델 설명 카드
with st.expander("📌 모델 설명"):
    st.markdown("""\
    **YOLO(You Only Look Once)**는 이미지에서 객체를 실시간으로 탐지하는 딥러닝 모델입니다.  
    이미지를 한 번만 보고 객체의 위치와 종류를 동시에 예측합니다.
    """)

# 업로드 카드
st.markdown("### 📂 이미지 업로드")

uploaded_file = st.file_uploader(
    "이미지를 업로드 하세요",
    type=["jpg", "jpeg", "png"]
)

# 이미지 표시
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", width="stretch")

# 버튼
predict_button = st.button("🚀 예측하기", width="stretch")

# 예측 실행
if predict_button:
    if uploaded_file is None:
        st.warning("⚠️ 먼저 이미지를 업로드해주세요!")
    else:
        with st.spinner("🔍 AI가 이미지를 분석 중입니다..."):
            img_array = np.array(image)
            results = model(img_array)
            result_img = results[0].plot()

        # 결과 이미지 카드
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ 예측 결과")
        st.image(result_img, width="stretch")
        
        # 결과 리스트
        st.markdown("### 📊 탐지 결과")

        if len(results[0].boxes) == 0:
            st.info("탐지된 객체가 없습니다.")
        else:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                st.markdown(f"""
                <div class="card">
                    <b>#{i+1} 🏷️ {label}</b><br><br>
                    <span class="label">신뢰도:</span> 
                    <span class="conf">{conf:.2f}</span><br>
                    <span class="label">좌표:</span> 
                    <span class="coord">({x1:.1f}, {y1:.1f}) ~ ({x2:.1f}, {y2:.1f})</span>
                </div>
                """, unsafe_allow_html=True)