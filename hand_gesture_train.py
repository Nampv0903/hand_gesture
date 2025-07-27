import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    f1_score, precision_score, classification_report
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Cấu hình đường dẫn và nhãn cho từng file json
JSON_LABELS = {
    'data/cu_chi_1.json': 'cu_chi_1',
    'data/cu_chi_2.json': 'cu_chi_2',
    'data/cu_chi_3.json': 'cu_chi_3',
    'data/cu_chi_4.json': 'cu_chi_4',
    'data/cu_chi_5.json': 'cu_chi_5',
}

MODEL_PATH = 'hand_gesture_mlp.h5'
LOG_PATH = 'log.txt'
HISTORY_PATH = 'hand_gesture_history.npy'
CONFUSION_MATRIX_IMG = 'confusion_matrix.png'
TRAIN_NPY = 'train_data.npy'
TEST_NPY = 'test_data.npy'

def load_landmark_data(json_labels): 
    X, y = [], []
    for json_path, label in json_labels.items():
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found!")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for frame in data:
            landmarks = frame['landmarks']
            if len(landmarks) == 21:
                X.append([lm['x'] for lm in landmarks] + [lm['y'] for lm in landmarks] + [lm['z'] for lm in landmarks])
                y.append(label)
    return np.array(X), np.array(y)

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix image saved to {save_path}")

def main():
    # Ưu tiên load từ file npy nếu đã tồn tại
    if os.path.exists(TRAIN_NPY) and os.path.exists(TEST_NPY):
        train_data = np.load(TRAIN_NPY, allow_pickle=True).item()
        test_data = np.load(TEST_NPY, allow_pickle=True).item()
        X_train, y_train = train_data['X'], train_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        print(f"Loaded train from {TRAIN_NPY}, test from {TEST_NPY}")
    else:
        X, y = load_landmark_data(JSON_LABELS)
        print(f"Loaded {len(X)} samples, {len(set(y))} classes.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        np.save(TRAIN_NPY, {'X': X_train, 'y': y_train})
        np.save(TEST_NPY, {'X': X_test, 'y': y_test})
        print(f"Saved train to {TRAIN_NPY}, test to {TEST_NPY}")

    # Encode label to integer
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    # One-hot encode
    y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_enc, num_classes)

    # Build MLP model
    model = keras.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        verbose=2
    )

    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Metrics
    acc = accuracy_score(y_test_enc, y_pred)
    recall = recall_score(y_test_enc, y_pred, average='macro')
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    precision = precision_score(y_test_enc, y_pred, average='macro')
    cm = confusion_matrix(y_test_enc, y_pred)
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_)

    # Lưu model và lịch sử
    model.save(MODEL_PATH)
    np.save(HISTORY_PATH, history.history)

    # Vẽ và lưu confusion matrix
    plot_confusion_matrix(cm, le.classes_, CONFUSION_MATRIX_IMG)

    # Lưu log
    with open(LOG_PATH, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    # Hiển thị ngắn gọn
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_IMG}")
    print(f"Saved log to {LOG_PATH}")

if __name__ == "__main__":
    main()
