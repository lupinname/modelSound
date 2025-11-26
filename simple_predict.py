#!/usr/bin/env python
# coding: utf-8
"""
Script đơn giản để sử dụng model gs_model_RF.pkl
"""

import numpy as np
import librosa
import joblib
import os
import sys
import cv2

# ========== CẤU HÌNH ==========
MODEL_PATH = "./Article/15_11_2025__07_36_45/mfcc/RF_SVC_KNN_DTC_Bagging/models/gs_model_RF.pkl"
AUDIO_FILE = "./audioTest/7b0e160e-0505-459e-8ecb-304d7afae9d2-1437486974312-1.7-m-04-dc.wav"  # Thay đổi đường dẫn file audio của bạn ở đây (hoặc truyền qua tham số)

LABELS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# ========== HÀM TRÍCH XUẤT ĐẶC TRƯNG ==========
def generate_spectrogram(aud, Fs):
    """Tạo mel spectrogram từ audio signal"""
    S = librosa.feature.melspectrogram(y=aud, sr=Fs, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = 255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    return S_norm.astype(np.uint8)

def extract_mel_spectrogram_features(audio_path, target_size=(216, 216)):
    """
    Trích xuất đặc trưng mel spectrogram từ file audio
    Model được train với mel spectrogram images (216x216x3 = 139968 features)
    """
    # Load audio
    signal, sr = librosa.load(audio_path, duration=5.0)
    
    # Tạo mel spectrogram
    spectrogram = generate_spectrogram(signal, sr)
    
    # Resize về target_size (216, 216) như trong training
    spectrogram_resized = cv2.resize(spectrogram, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Chuyển sang RGB (3 channels) để có shape (216, 216, 3)
    if len(spectrogram_resized.shape) == 2:
        # Chuyển grayscale sang RGB
        spectrogram_rgb = cv2.cvtColor(spectrogram_resized, cv2.COLOR_GRAY2RGB)
    else:
        spectrogram_rgb = spectrogram_resized
    
    # Chuyển về numpy array và flatten (giống image.img_to_array().flatten() trong training)
    # Đảm bảo dtype là float32 như trong training
    img_array = spectrogram_rgb.astype(np.float32)
    features = img_array.flatten()
    
    return features

# ========== SỬ DỤNG MODEL ==========
if __name__ == "__main__":
    # Lấy đường dẫn file audio từ tham số dòng lệnh nếu có
    audio_file = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
    
    # 1. Load model
    print("Đang load model...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
        exit(1)
    model = joblib.load(MODEL_PATH)
    print("✓ Model đã được load!\n")
    
    # 2. Kiểm tra file audio
    if not os.path.exists(audio_file):
        print(f"❌ Không tìm thấy file: {audio_file}")
        print("\nVui lòng:")
        print("1. Thay đổi biến AUDIO_FILE trong script này")
        print("2. Hoặc chạy với: python simple_predict.py <đường_dẫn_file_audio>")
        exit(1)
    
    # 3. Trích xuất đặc trưng
    print(f"Đang xử lý file: {audio_file}")
    features = extract_mel_spectrogram_features(audio_file)
    features = features.reshape(1, -1).astype('float32')
    print(f"✓ Đã trích xuất đặc trưng! Shape: {features.shape}\n")
    
    # 4. Dự đoán
    prediction = model.predict(features)[0]
    
    # 5. Hiển thị kết quả
    print("=" * 50)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("=" * 50)
    print(f"Loại tiếng khóc: {prediction}")
    
    # Hiển thị xác suất nếu có
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        print("\nXác suất cho từng loại:")
        for i, label in enumerate(LABELS):
            print(f"  {label:15s}: {probabilities[i]:.2%}")
    
    print("=" * 50)

