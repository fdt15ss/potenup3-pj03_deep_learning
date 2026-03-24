## 만약에 웹캠이 켜지지 않는다. 
## uv add opencv-contrib-python 

## mediapipe 라이브러리 설치 uv add mediapipe==0.10.14
import sys 
import cv2 
import mediapipe as mp 

# 모델 불러오기
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
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
    flipped_frame = cv2.flip(frame, 1)  # 그릴 도화지 입니다.

    #------------------------------------------------------
    # 포즈 그리기 준비 
    flipped_frame.flags.writeable = True 

    # 전체 포인트 감지하기 
    results = holistic.process(flipped_frame)

    # 추출 및 그리기
    ## 포즈 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            flipped_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )

    ## 얼굴 그리기
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            flipped_frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1,
                    circle_radius=1
                )
        )

    ## 왼손 그리기
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            flipped_frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # 오른손 그리기
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            flipped_frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
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