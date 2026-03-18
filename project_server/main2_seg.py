# (경로가 project_server에 있을 때)
# uvicorn main:app --port 8080 --reload

from fastapi import FastAPI, UploadFile, File, Form
import shutil
from datetime import datetime
import io
from PIL import Image
from pathlib import Path
import cv2

# -------------------------
# 경로 설정
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

ROOT_DIR = BASE_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "sam3.pt"

# -------------------------
# SAM3 모델 로드
# -------------------------
from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str(MODEL_PATH),
    half=True,
    save=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)
print("SAM3 모델을 불러왔습니다.")

# -------------------------
# FastAPI
# -------------------------
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "SAM3 Server Running"}


# -------------------------
# 이미지 저장 API
# -------------------------
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = IMAGE_DIR / f"{now}-{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "이미지를 저장했습니다.",
        "filename": str(file_path),
        "time": now
    }


# -------------------------
# SAM3 Segmentation API
# -------------------------
@app.post("/segment_image")
async def segment_image(
    file: UploadFile = File(...),
    text: str = Form(...)   # 👉 "person,dog,car"
):
    # -------------------------
    # 이미지 저장
    # -------------------------
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = IMAGE_DIR / f"{now}-{file.filename}"

    contents = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # -------------------------
    # 텍스트 파싱
    # -------------------------
    text_prompts = [t.strip() for t in text.split(",") if t.strip() != ""]

    # -------------------------
    # SAM3 추론
    # -------------------------
    predictor.set_image(str(file_path))
    results = predictor(text=text_prompts)

    result = results[0]

    # -------------------------
    # 결과 이미지 생성
    # -------------------------
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    result_img_pil = Image.fromarray(result_img)

    # 결과 이미지 저장
    result_path = IMAGE_DIR / f"{now}-result-{file.filename}"
    result_img_pil.save(result_path)

    # -------------------------
    # 간단 결과 정보 (옵션)
    # -------------------------
    detections = []
    if hasattr(result, "masks") and result.masks is not None:
        for i in range(len(result.masks.data)):
            detections.append({
                "id": i,
                "label": text_prompts if text_prompts else "unknown"
            })

    return {
        "message": "Segmentation 완료",
        "input_image": str(file_path),
        "result_image": str(result_path),
        "prompts": text_prompts,
        "num_objects": len(detections)
    }