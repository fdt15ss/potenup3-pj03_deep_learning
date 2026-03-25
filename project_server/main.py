# (경로가 project_server에 있을 때) uvicorn main:app --port 포트번호 --reload
from fastapi import FastAPI, UploadFile, File
import shutil
from datetime import datetime

import io
from PIL import Image

# 모델 불러오기
from ultralytics import YOLO
model = YOLO("../models/yolo26n.pt")
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
    file_name = f"./images/{now}-{file.filename}"

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
                "confidence": conf,
                "label": names[int(label_idx)]
            }
        )

    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"
    
    # 파일 저장
    file.file.seek(0)
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": datetime.now().strftime("%Y%m%d%H%M%S"),
        "object_detection": detections
    }

##################################################################################
# 챗봇 엔드포인트 만들기
##################################################################################
from pydantic import BaseModel
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
#---------------------------------------------------
# 챗봇 엔드포인트 만들기 (기본)
#---------------------------------------------------
class ChatRequest(BaseModel):
    message: str

def chatbot(user_message):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "당신은 친절한 챗봇입니다."},
            {"role": "user", "content": user_message}
        ]
    )

    return response.output_text

@app.post("/chat")
async def chat(req: ChatRequest):
    response = chatbot(req.message)

    return {"text": response}
#---------------------------------------------------
# 챗봇 엔드포인트 만들기 (히스토리 반영)
#---------------------------------------------------
from typing import List
# 요청 데이터 
# [
#     {"role": "user", "content": ""},
#     {"role": "ai", "content": ""},
#     {"role": "user", "content": ""},
#     ...
# ]

class Message(BaseModel):
    role: str
    content: str

class ChatHistoryRequest(BaseModel):
    history: List[Message]

def chatbot2(chat_history):
    input_list = [{"role": "system", "content": "당신은 친절한 챗봇입니다."}]
    for chat in chat_history:
        if chat.role == "ai":
            role = "assistant"
        else:
            role = "user"

        input_list.append(
            {"role": role, "content": chat.content}
        )

    response = client.responses.create(
        model="gpt-5-nano",
        input=input_list
    )

    return response.output_text

@app.post("/chat_with_history")
async def chat2(req: ChatHistoryRequest):
    response = chatbot2(req.history)

    return {"text": response}

