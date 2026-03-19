# (경로가 project_server에 있을 때) uvicorn main:app --port 8080 --reload
from fastapi import FastAPI, UploadFile, File
import shutil
from datetime import datetime

import io
from PIL import Image

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)

# project_server의 상위 폴더 (프로젝트 루트)
ROOT_DIR = BASE_DIR.parent

# models 폴더
MODEL_PATH = ROOT_DIR / "models" / "yolo26n.pt"

# 모델 불러오기
from ultralytics import YOLO

# 모델 로드
model = YOLO(str(MODEL_PATH))
print("모델을 불러왔습니다.")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = IMAGE_DIR / f"{now}-{file.filename}"

    # 파일 저장
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": datetime.now().strftime("%Y%m%d%H%M%S")
    }

@app.post("/detect_image")
async def predict_yolo(file: UploadFile = File(...)):
    # img 읽기
    img = Image.open(io.BytesIO(await file.read()))

    # 예측하기
    results = model.predict(img)
    result = results[0]

    # 데이터 만들기
    detections = []
    names = result.names 
    for x1, y1, x2, y2, conf, label_idx in result.boxes.data:
        detections.append(
            {
                "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": conf.item(),
                "label": names[int(label_idx)]
            }
        )


    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = IMAGE_DIR / f"{now}-{file.filename}"

    # 파일 저장
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": datetime.now().strftime("%Y%m%d%H%M%S"),
        "object_detection": detections
    }# (경로가 project_server에 있을 때) uvicorn main:app --port 8080 --reload

@app.post("/similarity")
async def similarity(file: UploadFile = File(...)):
    
    # 1️⃣ 파일 한 번만 읽기
    contents = await file.read()

    # 2️⃣ PIL 이미지로 변환 (CLIP용)
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 👉 여기서 CLIP 넣으면 됨
    # image_input = preprocess(image).unsqueeze(0).to(device)

    # 3️⃣ 파일 저장
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"images/{now}-{file.filename}"

    with open(file_name, "wb") as f:
        f.write(contents)

    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
    }