## 만약에 웹캠이 켜지지 않는다. 
## uv add opencv-contrib-python 

## mediapipe 라이브러리 설치 uv add mediapipe==0.10.14
import sys 
import cv2 
import mediapipe as mp 

# 모델 불러오기
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, # 근거리, 원거리 
    min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True # 디테일한 포인트 추출
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
    # 얼굴 그리기 준비 
    flipped_frame.flags.writeable = True 

    #------------------------------------------------------
    # Face Detection  
    #------------------------------------------------------
    # # 얼굴 감지하기 
    # results = face_detection.process(flipped_frame)

    # # 추출 및 그리기
    # if results.detections:
    #     for detection in results.detections:
    #         mp_drawing.draw_detection(flipped_frame, detection)
    #------------------------------------------------------
    # Face Mesh
    #------------------------------------------------------
    # 얼굴 감지하기 
    results = face_mesh.process(flipped_frame)

    # 추출 및 그리기
    h, w, c = flipped_frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ## 내가 원하는 포인트만 가져오기 
            left_up = 470
            left_down = 472 
            right_up = 476
            right_down = 477

            ## 좌표 추출하기 
            landmarks = face_landmarks.landmark
            print(f"Point {left_up} = {landmarks[left_up].x} | {landmarks[left_up].y} | {landmarks[left_up].z}")
            print(f"Point {left_down} = {landmarks[left_down].x} | {landmarks[left_down].y} | {landmarks[left_down].z}")
            print(f"Point {right_up} = {landmarks[right_up].x} | {landmarks[right_up].y} | {landmarks[right_up].z}")
            print(f"Point {right_down} = {landmarks[right_down].x} | {landmarks[right_down].y} | {landmarks[right_down].z}")
            print("="*100)

            ## 거리 구하기 
            left_up_x = int(landmarks[left_up].x * w)
            left_up_y = int(landmarks[left_up].y * h)
            left_down_x = int(landmarks[left_down].x * w)
            left_down_y = int(landmarks[left_down].y * h)

            left_dist = ((left_up_x - left_down_x)**2 + (left_up_y - left_down_y)**2) ** 0.5

            right_up_x = int(landmarks[right_up].x * w)
            right_up_y = int(landmarks[right_up].y * h)
            right_down_x = int(landmarks[right_down].x * w)
            right_down_y = int(landmarks[right_down].y * h)

            right_dist = ((right_up_x - right_down_x)**2 + (right_up_y - right_down_y)**2) ** 0.5

            ## 거리 표시하기 
            # cv2.putText(
            #     flipped_frame, f"distance: {left_dist}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2
            # )
            # cv2.putText(
            #     flipped_frame, f"distance: {right_dist}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2
            # )
            print(f"left dist: {left_dist} right_dist: {right_dist}")
            ## 조건에 따른 텍스트 표시 
            threshold = 7
            if left_dist < threshold and right_dist < threshold:
                cv2.putText(
                    flipped_frame, f"Don't sleep", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 2
                )

            # 자동 그리기
            # mp_drawing.draw_landmarks(
            #     flipped_frame,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_TESSELATION,
            #     mp_drawing.DrawingSpec(
            #         color=(0,255,0),
            #         thickness=1,
            #         circle_radius=1
            #     )
            # )

    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 화면 끄기 
    key = cv2.waitKey(1)
    if key == 27: # ESC(ASCII Code)
        break

vcap.release()
cv2.destroyAllWindows()