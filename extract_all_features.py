#!/usr/bin/env python
# coding: utf-8
"""
Script trích xuất tất cả các loại đặc trưng từ audio
- MFCC
- RMS (Root Mean Square)
- ZCR (Zero Crossing Rate)
- Mel Spectrogram
- Các biến thể từ MFCC (GADF, GASF, MTF, RP)
"""

import numpy as np
import librosa
import cv2
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
import pandas as pd

def normalize(data):
    """Chuẩn hóa dữ liệu về khoảng [-1, 1]"""
    xmax, xmin = data.max(), data.min()
    if xmax == xmin:
        return data
    return 2 * ((data - xmin) / (xmax - xmin)) - 1

def extract_mfcc(signal, sr, n_mfcc=20):
    """Trích xuất đặc trưng MFCC"""
    mfcc_signal = np.mean(librosa.feature.mfcc(
        y=signal, sr=sr, 
        n_mfcc=n_mfcc, 
        fmin=300., fmax=600.,
        center=True, n_mels=20, 
        n_fft=1024
    ), axis=0)
    return normalize(mfcc_signal)

def extract_rms(signal, sr):
    """Trích xuất đặc trưng RMS (Root Mean Square)"""
    S, phase = librosa.magphase(librosa.stft(signal))
    rms = librosa.feature.rms(S=S)
    return rms

def extract_zcr(signal, sr):
    """Trích xuất đặc trưng ZCR (Zero Crossing Rate)"""
    zcr = librosa.feature.zero_crossing_rate(signal)
    return zcr

def extract_mel_spectrogram(signal, sr, target_size=(216, 216)):
    """Trích xuất Mel Spectrogram dạng hình ảnh"""
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = 255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    spectrogram = S_norm.astype(np.uint8)
    
    # Resize về target_size
    spectrogram_resized = cv2.resize(spectrogram, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Chuyển sang RGB (3 channels)
    if len(spectrogram_resized.shape) == 2:
        spectrogram_rgb = cv2.cvtColor(spectrogram_resized, cv2.COLOR_GRAY2RGB)
    else:
        spectrogram_rgb = spectrogram_resized
    
    return spectrogram_rgb.astype(np.float32)

def approximate_ts(X, window_size):
    """Piecewise Aggregate Approximation"""
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    return paa.transform(X)

def mfcc_to_image(mfcc_feature, kind="GADF", res_sig_size=90):
    """
    Chuyển đổi MFCC thành hình ảnh
    kind: "GADF", "GASF", "MTF", "RP"
    """
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()
    rp = RecurrencePlot()
    
    # Tính window_size
    x = len(mfcc_feature) // res_sig_size
    if x != 0:
        ts = approximate_ts(mfcc_feature.reshape(1, -1), x)
        ts = ts.reshape(-1, 1)
    else:
        ts = mfcc_feature.reshape(-1, 1)
    
    if kind == "GADF":
        img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "GASF":
        img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "MTF":
        img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "RP":
        img = rp.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "RGB_GAF":
        gasf_img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
        gadf_img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gasf_img, gadf_img, np.zeros(gadf_img.shape)))
    else:
        raise ValueError(f"Unknown kind: {kind}")
    
    return img

def extract_all_features(audio_path, n_mfcc=20, include_images=True, target_size=(216, 216)):
    """
    Trích xuất tất cả các loại đặc trưng từ file audio
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file audio
    n_mfcc : int
        Số lượng hệ số MFCC
    include_images : bool
        Có bao gồm các đặc trưng dạng hình ảnh không
    target_size : tuple
        Kích thước cho hình ảnh (width, height)
    
    Returns:
    --------
    features_dict : dict
        Dictionary chứa tất cả các đặc trưng
    """
    # Load audio
    signal, sr = librosa.load(audio_path, duration=5.0)
    
    features = {}
    
    # 1. MFCC
    mfcc_feature = extract_mfcc(signal, sr, n_mfcc=n_mfcc)
    features['mfcc'] = mfcc_feature.flatten()
    
    # 2. RMS
    rms_feature = extract_rms(signal, sr)
    features['rms'] = rms_feature.flatten()
    
    # 3. ZCR
    zcr_feature = extract_zcr(signal, sr)
    features['zcr'] = zcr_feature.flatten()
    
    # 4. Mel Spectrogram
    if include_images:
        mel_spec = extract_mel_spectrogram(signal, sr, target_size=target_size)
        features['mel_spectrogram'] = mel_spec.flatten()
        
        # 5. Các biến thể từ MFCC (GADF, GASF, MTF, RP)
        for kind in ["GADF", "GASF", "MTF", "RP"]:
            try:
                img = mfcc_to_image(mfcc_feature, kind=kind, res_sig_size=90)
                features[f'mfcc_{kind.lower()}'] = img.flatten()
            except Exception as e:
                print(f"Warning: Không thể tạo {kind}: {e}")
    
    # Thông tin về shape
    features_info = {k: v.shape for k, v in features.items()}
    
    return features, features_info

def combine_features(features_dict, feature_types=None):
    """
    Kết hợp các đặc trưng lại với nhau
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary chứa các đặc trưng
    feature_types : list
        Danh sách các loại đặc trưng muốn kết hợp
        Nếu None, sẽ kết hợp tất cả
    
    Returns:
    --------
    combined_features : numpy array
        Đặc trưng đã được kết hợp (concatenate)
    """
    if feature_types is None:
        feature_types = list(features_dict.keys())
    
    combined = []
    for ft_type in feature_types:
        if ft_type in features_dict:
            combined.append(features_dict[ft_type])
        else:
            print(f"Warning: Không tìm thấy đặc trưng '{ft_type}'")
    
    if len(combined) == 0:
        raise ValueError("Không có đặc trưng nào để kết hợp!")
    
    return np.concatenate(combined)

# Ví dụ sử dụng
if __name__ == "__main__":
    audio_file = "./audioTest/7b0e160e-0505-459e-8ecb-304d7afae9d2-1437486974312-1.7-m-04-dc.wav"
    
    print("Đang trích xuất tất cả đặc trưng...")
    features, info = extract_all_features(audio_file, include_images=True)
    
    print("\n" + "="*60)
    print("THÔNG TIN CÁC ĐẶC TRƯNG ĐÃ TRÍCH XUẤT")
    print("="*60)
    for name, shape in info.items():
        print(f"{name:20s}: shape={shape}, size={np.prod(shape)}")
    
    print("\n" + "="*60)
    print("VÍ DỤ KẾT HỢP ĐẶC TRƯNG")
    print("="*60)
    
    # Kết hợp chỉ các đặc trưng số (MFCC, RMS, ZCR)
    numeric_features = combine_features(features, ['mfcc', 'rms', 'zcr'])
    print(f"\n1. Kết hợp MFCC + RMS + ZCR:")
    print(f"   Shape: {numeric_features.shape}")
    print(f"   Total features: {numeric_features.size}")
    
    # Kết hợp tất cả đặc trưng
    all_features = combine_features(features)
    print(f"\n2. Kết hợp TẤT CẢ đặc trưng:")
    print(f"   Shape: {all_features.shape}")
    print(f"   Total features: {all_features.size}")
    
    # Kết hợp MFCC + Mel Spectrogram
    mfcc_mel = combine_features(features, ['mfcc', 'mel_spectrogram'])
    print(f"\n3. Kết hợp MFCC + Mel Spectrogram:")
    print(f"   Shape: {mfcc_mel.shape}")
    print(f"   Total features: {mfcc_mel.size}")


