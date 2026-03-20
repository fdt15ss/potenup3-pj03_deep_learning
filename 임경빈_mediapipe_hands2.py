## 만약에 웹캠이 켜지지 않는다. 
## uv add opencv-contrib-python 

## mediapipe 라이브러리 설치 uv add mediapipe==0.10.14
import sys 
import cv2 
import mediapipe as mp 
from mediapipe.framework.formats import landmark_pb2
import math

# 모델 불러오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

##########################################################
# 카메라 시작
##########################################################
vcap = cv2.VideoCapture(0)

while True:
    # 카메라 이미지 추출하기 
    ret, frame = vcap.read()

    # 카메라 작동 확인
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()

    # 좌우 반전 
    flipped_frame = cv2.flip(frame, 1)

    #------------------------------------------------------
    # 손 그리기 준비 
    frame.flags.writeable = True 

    # 손 감지하기 
    results = hands.process(flipped_frame)

    # 추출 및 그리기
    if results.multi_hand_landmarks:
        ## 손 하나하나 가져오기 
        for hand_landmarks in results.multi_hand_landmarks:
            print(len(hand_landmarks.landmark))
            
            landmarks = hand_landmarks.landmark
            custom_landmark_list = landmark_pb2.NormalizedLandmarkList()

        for idx, landmark in enumerate(landmarks):
            if idx >=5 and idx <= 12:
                h, w, c = flipped_frame.shape
                point_x = int(landmark.x * w)
                point_y = int(landmark.y * h)

                # 필요하면 직접 그리기
                # cv2.circle(flipped_frame, (point_x, point_y), 5, (0,255,0), 2)

                # ✅ MediaPipe 구조로 추가
                new_landmark = custom_landmark_list.landmark.add()
                new_landmark.x = landmark.x
                new_landmark.y = landmark.y
                new_landmark.z = landmark.z

        # ✅ 이걸 넣어야 정상 작동
        mp_drawing.draw_landmarks(
            flipped_frame,
            custom_landmark_list,
            [
                (0,1), (1,2), (2,3),   # 검지 (5~8)
                (4,5), (5,6), (6,7), # 중지 (9~12)
                (3,7)
            ],  # 연결선 없음 (선 안 그릴거니까)
            # mp_drawing_styles.get_default_hand_landmarks_style(),
        )
    #------------------------------------------------------

    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 화면 끄기 
    key = cv2.waitKey(1)
    if key == 27: # ESC(ASCII Code)
        break

vcap.release()
cv2.destroyAllWindows()