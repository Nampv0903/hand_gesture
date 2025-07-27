# Hand Gesture Recognition with Mediapipe & Keras

## Mô tả
Dự án này sử dụng Mediapipe để trích xuất landmark bàn tay từ video/webcam, lưu dữ liệu landmark ra file JSON, huấn luyện mô hình nhận diện cử chỉ bàn tay bằng Keras (MLP), và dự đoán cử chỉ trên video.

## Thư mục chính
- `hand_pose_extract.py`: Thu thập dữ liệu landmark từ webcam/video và lưu ra file JSON.
- `hand_pose_replay.py`: Mô phỏng lại cử động từ file json
- `hand_gesture_train.py`: Huấn luyện mô hình nhận diện cử chỉ bàn tay từ dữ liệu landmark.
- `hand_gesture_video_predict_keras.py`: Dự đoán cử chỉ bàn tay trên video sử dụng model Keras đã huấn luyện.
- `data/`: Chứa các file landmark JSON cho từng cử chỉ.

## Hướng dẫn sử dụng

### 1. Thu thập dữ liệu landmark
- Chạy script để thu landmark từ webcam:
    ```bash
    python hand_pose_extract.py
    ```
- Nhấn `q` để bắt đầu thu, `ESC` để dừng và lưu file JSON vào thư mục `data/`.

### 2. Huấn luyện mô hình
- Chỉnh sửa biến `JSON_LABELS` trong `hand_gesture_train.py` cho đúng tên file và nhãn.
- Chạy:
    ```bash
    python hand_gesture_train.py
    ```
- Model, confusion matrix, log, history sẽ được lưu ra file.

### 3. Dự đoán trên video
- Chỉnh đường dẫn video trong `hand_gesture_video_predict_keras.py`.
- Chạy:
    ```bash
    python hand_gesture_video_predict_keras.py
    ```
- Kết quả sẽ được lưu ra file video mới với nhãn dự đoán trên từng frame.

## Yêu cầu cài đặt
```bash
pip install opencv-python mediapipe tensorflow scikit-learn matplotlib seaborn
```

## Lưu ý
- Mô hình này được chạy trên bộ dữ liệu được thu thập bởi 1 người và không có nhiều mẫu, nên có hiện tượng overfiting nhẹ.
- Để mô hình tổng quát tốt, hãy thu thập dữ liệu đa dạng (nhiều người, nhiều góc, nhiều điều kiện).
- Có thể tăng cường dữ liệu landmark bằng các phép biến đổi (nhiễu, xoay, scale...).
- Nếu chạy trên server không có GUI, hãy thu landmark từ video thay vì webcam.

---

**Liên hệ:** Nếu có thắc mắc, vui lòng liên hệ nampv0903@gmail.com.