import json
import cv2
import numpy as np

# Đường dẫn file json xuất ra từ hand_pose_extract.py
JSON_PATH = '/media/DATA1/NAMPV/test_opencv/data/cu_chi_1.json'
OUTPUT_VIDEO = '/media/DATA1/NAMPV/test_opencv/hand_replay.mp4'

# Kích thước khung hình hiển thị
WIDTH, HEIGHT = 640, 480
FPS = 30

# 21 connections của Mediapipe Hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),      # Ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20)  # Ngón út
]

def replay_hand_landmarks(json_path, output_video):
    with open(json_path, 'r') as f:
        data = json.load(f)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, FPS, (WIDTH, HEIGHT))
    for frame_data in data:
        img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        landmarks = frame_data['landmarks']
        if len(landmarks) == 21:
            points = []
            for lm in landmarks:
                x = int(lm['x'] * WIDTH)
                y = int(lm['y'] * HEIGHT)
                points.append((x, y))
            # Vẽ connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(img, points[start], points[end], (0, 0, 255), 2)
            # Vẽ điểm
            for x, y in points:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        out.write(img)
    out.release()
    print(f"Đã lưu video mô phỏng vào {output_video}")

if __name__ == "__main__":
    replay_hand_landmarks(JSON_PATH, OUTPUT_VIDEO)
