import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------
# 모델 로드
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("../models/yolo26n-seg.pt")

model = load_model()

# -------------------------
# UI
# -------------------------
st.title("Segmentation 체험하기")

# 모델 설명
with st.expander("📘 모델 설명", expanded=True):
    st.markdown("""
    #### YOLO Segmentation 모델

    - segmentation은 이미지를 입력 받아 객체의 픽셀 단위 영역을 분할하는 모델입니다.  
    - 이미지를 입력하면 객체의 경계(mask)와 클래스 정보를 함께 출력합니다.  
    - object detection이 bounding box로 위치를 찾는다면, segmentation은 객체의 정확한 윤곽을 구분할 수 있습니다.  
    - segmentation을 통해 사람, 동물, 사물의 형태를 정밀하게 분리할 수 있습니다.  
    - segmentation을 학습할 때에는 정밀한 마스크 라벨링과 클래스 불균형 문제를 주의해야 합니다.  
    """)

st.markdown("## YOLO Segmentation 모델")
st.markdown("""
이미지를 업로드하면 segmentation 결과와 마스크를 확인할 수 있습니다.
""")

# -------------------------
# 파일 업로드
# -------------------------
uploaded_file = st.file_uploader(
    label="이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", width="stretch")

# -------------------------
# 예측 버튼
# -------------------------
predict_button = st.button("예측하기")

# -------------------------
# 예측 실행
# -------------------------
if predict_button:
    if image is None:
        st.warning("먼저 이미지를 업로드해주세요.")
    else:
        with st.spinner("예측 중입니다..."):
            img_array = np.array(image)

            results = model(img_array)
            result = results[0]

            # 기본 시각화 이미지
            plotted_img = result.plot()

        st.success("예측 완료!")

        # -------------------------
        # 1. YOLO 기본 결과
        # -------------------------
        st.image(plotted_img, caption="Segmentation 결과", width="stretch")

        # -------------------------
        # 2. masks.data 출력
        # -------------------------
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

            st.markdown("### 마스크 결과")

            for i, mask in enumerate(masks):
                # 0~1 값을 0~255로 변환
                mask_img = (mask * 255).astype(np.uint8)

                st.image(
                    mask_img,
                    caption=f"Mask {i}",
                    width="stretch",
                    clamp=True
                )
        else:
            st.info("검출된 마스크가 없습니다.")