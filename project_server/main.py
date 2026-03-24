# 실행 방법 (project_server 폴더 안에서 실행)
# uvicorn main:app --port 8000 --reload

# ── 기본 라이브러리 ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File
import shutil           # 파일 복사(저장)에 사용
from datetime import datetime

import io               # 메모리 버퍼(파일을 디스크에 저장하지 않고 메모리에서 처리)
import base64           # 바이너리 데이터를 텍스트(문자열)로 변환 (마스크 전송용)
import numpy as np
from PIL import Image   # 이미지 열기 / 변환

# ── 디바이스 설정 ────────────────────────────────────────────────────────────────
# GPU(CUDA)가 있으면 GPU, 없으면 CPU 사용
# 모든 모델(YOLO, CLIP)에 공통으로 적용
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ── YOLO 모델 로드 ───────────────────────────────────────────────────────────────
# ultralytics YOLO 라이브러리 사용
# yolo26n.pt     : 객체 탐지(detection) 전용 모델
# yolo26n-seg.pt : 객체 탐지 + 세그멘테이션(마스크) 모델
from ultralytics import YOLO
yolo_model = YOLO("../models/yolo26n.pt").to(device)      # Object Detection 모델
seg_model  = YOLO("../models/yolo26n-seg.pt").to(device)  # Segmentation 모델
print("YOLO 모델을 불러왔습니다.")

# ── CLIP 모델 로드 ───────────────────────────────────────────────────────────────
# HuggingFace transformers의 CLIP 사용
# CLIPModel    : 이미지-텍스트 유사도를 계산하는 모델
# CLIPProcessor: 이미지와 텍스트를 모델 입력 형식으로 변환(전처리)
from transformers import CLIPProcessor, CLIPModel

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP 모델을 불러왔습니다.")

# CLIP 분류 카테고리 (이 중 하나로 분류)
CLIP_CATEGORIES = ["개", "고양이", "돼지"]

# ── FastAPI 앱 생성 ──────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# ────────────────────────────────────────────────────────────────────────────────
# POST /upload_image
# 역할: 업로드된 이미지를 서버 로컬에 저장한다.
# 입력: 이미지 파일 (multipart/form-data)
# 출력: 저장된 파일 이름, 저장 시각
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    # 저장 시각을 "년월일시분초" 문자열로 변환 (파일명 중복 방지)
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    # 저장 경로: ./images/20240101120000-원본파일명.jpg
    file_name = f"./images/{now}-{file.filename}"

    # 파일을 바이너리 쓰기 모드로 열고 업로드된 내용을 그대로 복사
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": now
    }


# ────────────────────────────────────────────────────────────────────────────────
# POST /object_detection
# 역할: 업로드된 이미지에서 YOLO로 객체를 탐지하고 결과를 반환한다.
# 입력: 이미지 파일 (multipart/form-data)
# 출력: detections 배열
#   - box        : [x1, y1, x2, y2] 바운딩 박스 좌표 (좌상단/우하단 픽셀 좌표)
#   - confidence : 예측 신뢰도 (0.0 ~ 1.0)
#   - label      : 객체 이름 (예: "person", "car")
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/object_detection")
async def object_detection(file: UploadFile = File(...)):
    # 업로드된 파일을 바이트로 읽기
    img_bytes = await file.read()

    # 바이트 데이터를 메모리 버퍼로 감싸서 PIL Image로 변환
    # (디스크에 저장하지 않고 메모리에서 바로 이미지 처리)
    img = Image.open(io.BytesIO(img_bytes))

    # YOLO로 예측 → results는 리스트, 이미지 1장이므로 [0]만 사용
    results = yolo_model.predict(img)
    result  = results[0]

    # 탐지된 객체 정보를 담을 리스트
    detections = []

    # result.names: {0: 'person', 1: 'bicycle', ...} 클래스 인덱스-이름 딕셔너리
    names = result.names

    # result.boxes.data: 탐지된 각 객체의 [x1, y1, x2, y2, confidence, class_index] 텐서
    for x1, y1, x2, y2, conf, label_idx in result.boxes.data:
        detections.append({
            # .item()으로 PyTorch 텐서 → 파이썬 기본 타입(float)으로 변환
            "box": [x1.item(), y1.item(), x2.item(), y2.item()],
            "confidence": conf.item(),
            "label": names[int(label_idx)]  # 클래스 인덱스로 이름 조회
        })

    # 예측에 사용한 이미지를 서버에 저장
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"
    with open(file_name, "wb") as buffer:
        buffer.write(img_bytes)

    return {
        "message": "예측이 완료되었습니다.",
        "filename": file_name,
        "detections": detections
    }


# ────────────────────────────────────────────────────────────────────────────────
# POST /segmentation
# 역할: 업로드된 이미지에서 YOLO-seg로 객체를 탐지하고 세그멘테이션 마스크를 반환한다.
# 입력: 이미지 파일 (multipart/form-data)
# 출력: detections 배열
#   - box        : [x1, y1, x2, y2] 바운딩 박스 좌표
#   - confidence : 예측 신뢰도
#   - label      : 객체 이름
#   - mask       : 마스크 이미지 (PNG → base64 인코딩 문자열)
#                  클라이언트에서 base64 디코딩 후 PNG 이미지로 복원 가능
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/segmentation")
async def segmentation(file: UploadFile = File(...)):
    # 업로드된 파일을 바이트로 읽고 PIL Image로 변환
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # YOLO-seg로 예측
    results = seg_model.predict(img)
    result  = results[0]

    detections = []
    names = result.names   # 클래스 인덱스-이름 딕셔너리
    masks = result.masks   # 세그멘테이션 마스크 객체 (탐지된 객체가 없으면 None)

    # enumerate로 인덱스(i)와 박스 좌표를 함께 순회
    # → i는 masks.data[i]에서 해당 객체의 마스크를 꺼낼 때 사용
    for i, (x1, y1, x2, y2, conf, label_idx) in enumerate(result.boxes.data):
        detection = {
            "box": [x1.item(), y1.item(), x2.item(), y2.item()],
            "confidence": conf.item(),
            "label": names[int(label_idx)],
            "mask": None  # 마스크가 없는 경우 None 유지
        }

        # 마스크가 존재하면 base64 문자열로 변환
        if masks is not None and i < len(masks.data):
            # masks.data[i]: (H, W) 형태의 0/1 이진 텐서
            # .cpu(): GPU 텐서를 CPU로 이동, .numpy(): numpy 배열로 변환
            mask_np = masks.data[i].cpu().numpy()

            # 0/1 이진값 → 0/255 흑백 이미지로 변환 (PIL Image 생성)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

            # 메모리 버퍼에 PNG 형식으로 저장 (디스크 I/O 없이 처리)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")

            # PNG 바이너리 → base64 문자열 인코딩 (JSON으로 전송 가능한 텍스트로 변환)
            detection["mask"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        detections.append(detection)

    # 예측에 사용한 이미지를 서버에 저장
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"
    with open(file_name, "wb") as buffer:
        buffer.write(img_bytes)

    return {
        "message": "세그멘테이션이 완료되었습니다.",
        "filename": file_name,
        "detections": detections
    }


# ────────────────────────────────────────────────────────────────────────────────
# POST /clip
# 역할: 업로드된 이미지가 ["개", "고양이", "돼지"] 중 무엇인지 CLIP으로 분류한다.
# 입력: 이미지 파일 (multipart/form-data)
# 출력:
#   - predicted   : 가장 유사도가 높은 카테고리 이름
#   - similarities: 각 카테고리별 유사도 딕셔너리 (softmax 확률, 합계 = 1.0)
#     예) {"개": 0.95, "고양이": 0.03, "돼지": 0.02}
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/clip")
async def clip_classify(file: UploadFile = File(...)):
    # 업로드된 파일을 바이트로 읽기
    img_bytes = await file.read()

    # PIL Image로 변환 + RGB로 변환
    # CLIP은 RGB 3채널 이미지를 입력으로 받으므로 .convert("RGB") 필수
    # (PNG처럼 RGBA 4채널이거나 흑백 이미지일 경우를 대비)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # CLIPProcessor로 이미지와 텍스트를 모델 입력 형식(텐서)으로 전처리
    # text   : 비교할 카테고리 텍스트 리스트
    # images : 분류할 이미지
    # padding=True: 텍스트 길이가 다를 경우 짧은 쪽을 패딩
    inputs = clip_processor(
        text=CLIP_CATEGORIES,
        images=img,
        return_tensors="pt",  # PyTorch 텐서로 반환
        padding=True
    ).to(device)

    # 추론(예측): torch.no_grad()로 기울기 계산 끄기 → 메모리 절약 + 속도 향상
    with torch.no_grad():
        # logits_per_image: 이미지와 각 텍스트 간의 유사도 점수 (1 x 카테고리 수)
        # softmax(dim=-1): 점수를 확률 분포로 변환 (모든 값의 합 = 1)
        # .cpu().numpy()[0]: GPU → CPU → numpy 배열, [0]으로 1차원 배열 추출
        probs = clip_model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # 카테고리 이름 : 확률 딕셔너리 생성
    similarities = {label: float(prob) for label, prob in zip(CLIP_CATEGORIES, probs)}

    # 가장 확률이 높은 카테고리를 최종 예측 결과로 선택
    predicted = max(similarities, key=similarities.get)

    return {
        "predicted": predicted,
        "similarities": similarities
    }

##################################################################################
# 챗봇 엔드포인트 만들기
##################################################################################
from pydantic import BaseModel
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv("../.env")
client = OpenAI()

def chatbot(user_message):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "당신은 친절한 챗봇입니다."},
            {"role": "user", "content": user_message}
        ]
    )

    return response.output_text

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = chatbot(req.message)

    return {"text": response}