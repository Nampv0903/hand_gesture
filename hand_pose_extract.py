import cv2
import mediapipe as mp
import json
import os

# Đường dẫn file json xuất ra
JSON_PATH = 'data/cu_chi_5.json'

if not os.path.exists('data'):
    os.makedirs('data')

mp_hands = mp.solutions.hands

def extract_hand_landmarks(json_path):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    detect_count = 0  # Số frame phát hiện được bàn tay
    results_list = []
    collecting = False

    print("Nhấn 'q' để bắt đầu thu dữ liệu, nhấn 'ESC' để dừng và lưu.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu đang thu, hiển thị số frame đã detect được bàn tay
        if collecting:
            text = f"Detected frames: {detect_count}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam - Nhấn q để bắt đầu, ESC để dừng', frame)
        key = cv2.waitKey(1) & 0xFF
        if not collecting and key == ord('q'):
            print("Bắt đầu thu dữ liệu...")
            collecting = True
        if key == 27:  # ESC
            break
        if collecting:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                frame_landmarks = []
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                results_list.append({'frame': detect_count, 'landmarks': frame_landmarks})
                detect_count += 1

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    with open(json_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"Đã lưu dữ liệu vào {json_path}")
    print(f"Tổng số frame đã phát hiện bàn tay: {detect_count}")

if __name__ == "__main__":
    extract_hand_landmarks(JSON_PATH)
