#!/usr/bin/env python
# coding: utf-8
"""
Script để sử dụng model gs_model_RF.pkl để dự đoán loại tiếng khóc của trẻ sơ sinh
"""

import numpy as np
import librosa
import joblib
import pathlib
import os

# Các nhãn phân loại
LABELS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

def normalize(data):
    """Chuẩn hóa dữ liệu về khoảng [-1, 1]"""
    xmax, xmin = data.max(), data.min()
    if xmax == xmin:
        return data
    return 2 * ((data - xmin) / (xmax - xmin)) - 1

def audio_mfcc(signal, sr, n_mfcc=30):
    """Trích xuất đặc trưng MFCC từ tín hiệu audio"""
    mfcc_signal = np.mean(librosa.feature.mfcc(y=signal, sr=sr, 
                                              n_mfcc=n_mfcc, 
                                              fmin=300., fmax=600.,
                                              center=True, n_mels=20, 
                                              n_fft=1024), axis=0)
    return normalize(mfcc_signal)

def extract_features_from_audio(audio_path, n_mfcc=20):
    """
    Trích xuất đặc trưng từ file audio
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file audio (.wav)
    n_mfcc : int
        Số lượng hệ số MFCC (mặc định 20)
    
    Returns:
    --------
    features : numpy array
        Mảng đặc trưng đã được flatten
    """
    # Load audio file
    signal, sr = librosa.load(audio_path, duration=5.0)
    
    # Trích xuất đặc trưng MFCC
    mfcc_feature = audio_mfcc(signal, sr, n_mfcc=n_mfcc)
    
    # Flatten để có cùng format với dữ liệu training
    features = mfcc_feature.flatten()
    
    return features

def load_model(model_path):
    """
    Load model từ file .pkl
    
    Parameters:
    -----------
    model_path : str
        Đường dẫn đến file model .pkl
    
    Returns:
    --------
    model : sklearn model
        Model đã được load
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Đã load model từ: {model_path}")
    return model

def predict_audio(model, audio_path, n_mfcc=20):
    """
    Dự đoán loại tiếng khóc từ file audio
    
    Parameters:
    -----------
    model : sklearn model
        Model đã được load
    audio_path : str
        Đường dẫn đến file audio
    n_mfcc : int
        Số lượng hệ số MFCC
    
    Returns:
    --------
    prediction : str
        Nhãn dự đoán
    probabilities : dict
        Xác suất cho từng lớp
    """
    # Trích xuất đặc trưng
    features = extract_features_from_audio(audio_path, n_mfcc=n_mfcc)
    
    # Reshape để có shape (1, n_features) cho prediction
    features = features.reshape(1, -1).astype('float32')
    
    # Dự đoán
    prediction = model.predict(features)[0]
    
    # Lấy xác suất (nếu model hỗ trợ)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prob_dict = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        prob_dict = None
    
    return prediction, prob_dict

def predict_from_features_file(model, features_path):
    """
    Dự đoán từ file .npy đã có sẵn đặc trưng
    
    Parameters:
    -----------
    model : sklearn model
        Model đã được load
    features_path : str
        Đường dẫn đến file .npy chứa đặc trưng
    
    Returns:
    --------
    prediction : str
        Nhãn dự đoán
    probabilities : dict
        Xác suất cho từng lớp
    """
    # Load features từ file .npy
    features = np.load(features_path)
    features = features.flatten().reshape(1, -1).astype('float32')
    
    # Dự đoán
    prediction = model.predict(features)[0]
    
    # Lấy xác suất (nếu model hỗ trợ)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prob_dict = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        prob_dict = None
    
    return prediction, prob_dict

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn đến model
    model_path = "./Article/15_11_2025__07_36_45/mfcc/RF_SVC_KNN_DTC_Bagging/models/gs_model_RF.pkl"
    
    # Load model
    try:
        model = load_model(model_path)
        print("Model đã được load thành công!\n")
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        exit(1)
    
    # Ví dụ 1: Dự đoán từ file audio
    print("=" * 60)
    print("VÍ DỤ 1: Dự đoán từ file audio")
    print("=" * 60)
    
    # Thay đổi đường dẫn này thành đường dẫn file audio của bạn
    audio_file = "./dataset/belly_pain/example.wav"  # Thay đổi đường dẫn này
    
    if os.path.exists(audio_file):
        try:
            prediction, probabilities = predict_audio(model, audio_file)
            print(f"File audio: {audio_file}")
            print(f"Dự đoán: {prediction}")
            if probabilities:
                print("\nXác suất cho từng lớp:")
                for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
    else:
        print(f"File audio không tồn tại: {audio_file}")
        print("Vui lòng thay đổi đường dẫn audio_file trong script\n")
    
    # Ví dụ 2: Dự đoán từ file features .npy
    print("\n" + "=" * 60)
    print("VÍ DỤ 2: Dự đoán từ file features .npy")
    print("=" * 60)
    
    # Tìm một file .npy trong thư mục features
    features_dir = "./features/mfcc"
    if os.path.exists(features_dir):
        npy_files = list(pathlib.Path(features_dir).rglob("*.npy"))
        if npy_files:
            features_file = str(npy_files[0])
            try:
                prediction, probabilities = predict_from_features_file(model, features_file)
                print(f"File features: {features_file}")
                print(f"Dự đoán: {prediction}")
                if probabilities:
                    print("\nXác suất cho từng lớp:")
                    for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")
            except Exception as e:
                print(f"Lỗi khi dự đoán: {e}")
        else:
            print(f"Không tìm thấy file .npy trong {features_dir}")
    else:
        print(f"Thư mục features không tồn tại: {features_dir}")
    
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN SỬ DỤNG:")
    print("=" * 60)
    print("""
1. Để dự đoán từ file audio mới:
   - Sử dụng hàm predict_audio(model, audio_path)
   - Ví dụ: prediction, prob = predict_audio(model, "path/to/audio.wav")

2. Để dự đoán từ file features .npy:
   - Sử dụng hàm predict_from_features_file(model, features_path)
   - Ví dụ: prediction, prob = predict_from_features_file(model, "path/to/features.npy")

3. Các nhãn có thể:
   - belly_pain (đau bụng)
   - burping (ợ hơi)
   - discomfort (khó chịu)
   - hungry (đói)
   - tired (mệt mỏi)
    """)

