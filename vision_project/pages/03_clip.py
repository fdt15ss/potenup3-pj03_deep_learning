import streamlit as st
import clip
import torch
from PIL import Image

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

model, preprocess = load_model()

# 클래스 (노트북 스타일 유지)
classes = ["a dog", "a cat", "a pig"]

# UI
st.title("💡 CLIP 이미지 유사도 분류")
st.write("이미지를 업로드하면 [개, 고양이, 돼지] 중 확률을 출력합니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="업로드 이미지", width="stretch")

    # 전처리
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        # 🔥 노트북 코드 그대로 사용
        logits_per_image, logits_per_text = model(image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    st.subheader("📊 결과 (확률)")

    # 결과 출력
    for label, prob in zip(classes, probs):
        st.write(f"{label}: {prob:.4f}")

    # 최고 확률
    best_idx = probs.argmax()
    st.success(f"👉 가장 유사한 것: {classes[best_idx]}")