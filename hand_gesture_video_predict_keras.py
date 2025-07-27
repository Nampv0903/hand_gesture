import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# Đường dẫn model, label encoder, video
MODEL_PATH = 'hand_gesture_mlp.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
VIDEO_PATH = 'test.mp4'  # Đổi thành video bạn muốn test
OUTPUT_VIDEO = 'hand_gesture_predicted_keras.mp4'
WIDTH, HEIGHT = 640, 480

mp_hands = mp.solutions.hands

# Load model và label encoder
model = keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)


def get_landmark_vector(landmarks):
    # Flatten 21 điểm (x, y, z) thành 63 chiều
    return np.array([lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [lm.z for lm in landmarks]).reshape(1, -1)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (WIDTH, HEIGHT))
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        label = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if len(hand_landmarks.landmark) == 21:
                    vec = get_landmark_vector(hand_landmarks.landmark)
                    pred_prob = model.predict(vec)
                    pred_idx = np.argmax(pred_prob, axis=1)[0]
                    label = le.inverse_transform([pred_idx])[0]
                    # Vẽ nhãn lên frame
                    cv2.putText(frame, str(label), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        out.write(frame)
    cap.release()
    out.release()
    hands.close()
    print(f"Đã lưu video kết quả vào {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
