import sys
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math

# ==============================
# MediaPipe 초기화
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# 유틸 함수
# ==============================

def create_two_landmarks(landmarks):
    """4번, 8번 landmark만 추출"""
    custom_list = landmark_pb2.NormalizedLandmarkList()

    for idx in [4, 8]:
        lm = landmarks[idx]
        new_lm = custom_list.landmark.add()
        new_lm.x, new_lm.y, new_lm.z = lm.x, lm.y, lm.z

    return custom_list


def calculate_distance(landmarks, shape):
    """엄지(4) - 검지(8) 거리 계산"""
    h, w, _ = shape

    x1, y1 = int(landmarks[4].x * w), int(landmarks[4].y * h)
    x2, y2 = int(landmarks[8].x * w), int(landmarks[8].y * h)

    distance = math.hypot(x2 - x1, y2 - y1)

    return distance, (x1, y1)


def draw_custom_landmarks(image, landmark_list):
    """4-8 landmark만 MediaPipe 스타일로 그리기"""
    mp_drawing.draw_landmarks(
        image,
        landmark_list,
        [(0, 1)],  # custom index 기준
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )


def draw_distance_text(image, distance, position):
    """거리 텍스트 출력"""
    cv2.putText(
        image,
        f"{int(distance)} px",
        (position[0], position[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# ==============================
# 카메라 시작
# ==============================
vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()

    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()

    flipped_frame = cv2.flip(frame, 1)

    # 성능 최적화 (MediaPipe 권장)
    flipped_frame.flags.writeable = False
    results = hands.process(flipped_frame)
    flipped_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # 1️⃣ landmark 생성
            custom_landmarks = create_two_landmarks(landmarks)

            # 2️⃣ 그리기
            draw_custom_landmarks(flipped_frame, custom_landmarks)

            # 3️⃣ 거리 계산
            distance, pos = calculate_distance(landmarks, flipped_frame.shape)

            # 4️⃣ 텍스트 출력
            draw_distance_text(flipped_frame, distance, pos)

    cv2.imshow("webcam", flipped_frame)

    if cv2.waitKey(1) == 27:
        break

vcap.release()
cv2.destroyAllWindows()