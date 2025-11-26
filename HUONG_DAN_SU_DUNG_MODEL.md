# Hướng Dẫn Sử Dụng Model gs_model_RF.pkl

## Tổng Quan

Model `gs_model_RF.pkl` là một Random Forest classifier được huấn luyện để phân loại tiếng khóc của trẻ sơ sinh thành 5 loại:
- **belly_pain**: Đau bụng
- **burping**: Ợ hơi
- **discomfort**: Khó chịu
- **hungry**: Đói
- **tired**: Mệt mỏi

## Cách Sử Dụng

### Cách 1: Sử dụng script đơn giản (Khuyến nghị)

1. Mở file `simple_predict.py`
2. Thay đổi đường dẫn file audio:
   ```python
   AUDIO_FILE = "đường/dẫn/đến/file/audio.wav"
   ```
3. Chạy script:
   ```bash
   python simple_predict.py
   ```

Hoặc truyền đường dẫn file audio qua tham số dòng lệnh:
```bash
python simple_predict.py path/to/audio.wav
```

### Cách 2: Sử dụng script đầy đủ

Chạy file `predict.py` để xem các ví dụ chi tiết:
```bash
python predict.py
```

### Cách 3: Sử dụng trong code Python của bạn

```python
import numpy as np
import librosa
import joblib

# Load model
model = joblib.load("./Article/15_11_2025__07_36_45/mfcc/RF_SVC_KNN_DTC_Bagging/models/gs_model_RF.pkl")

# Trích xuất đặc trưng từ file audio
def normalize(data):
    xmax, xmin = data.max(), data.min()
    if xmax == xmin:
        return data
    return 2 * ((data - xmin) / (xmax - xmin)) - 1

def extract_mfcc_features(audio_path, n_mfcc=20):
    signal, sr = librosa.load(audio_path, duration=5.0)
    mfcc_signal = np.mean(librosa.feature.mfcc(
        y=signal, sr=sr, 
        n_mfcc=n_mfcc, 
        fmin=300., fmax=600.,
        center=True, n_mels=20, 
        n_fft=1024
    ), axis=0)
    return normalize(mfcc_signal).flatten()

# Dự đoán
audio_file = "path/to/your/audio.wav"
features = extract_mfcc_features(audio_file)
features = features.reshape(1, -1).astype('float32')

prediction = model.predict(features)[0]
print(f"Dự đoán: {prediction}")

# Lấy xác suất (nếu cần)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(features)[0]
    print("Xác suất:", probabilities)
```

## Yêu Cầu

- Python 3.x
- Các thư viện cần thiết:
  ```bash
  pip install numpy librosa scikit-learn joblib
  ```

## Lưu Ý

1. **Format file audio**: Model được huấn luyện với file WAV. Nếu file của bạn ở format khác, hãy chuyển đổi sang WAV trước.

2. **Độ dài audio**: Model xử lý 5 giây đầu tiên của file audio. Nếu file dài hơn, chỉ 5 giây đầu sẽ được sử dụng.

3. **Đặc trưng**: Model sử dụng đặc trưng MFCC với:
   - `n_mfcc=20`
   - `fmin=300`, `fmax=600`
   - `n_mels=20`
   - `n_fft=1024`

4. **Format dữ liệu**: Đặc trưng phải được chuẩn hóa và có kiểu `float32`.

## Ví Dụ Kết Quả

```
==================================================
KẾT QUẢ DỰ ĐOÁN
==================================================
Loại tiếng khóc: hungry

Xác suất cho từng loại:
  belly_pain     : 5.23%
  burping        : 2.15%
  discomfort     : 8.42%
  hungry         : 82.10%
  tired          : 2.10%
==================================================
```

## Xử Lý Lỗi

### Lỗi: FileNotFoundError
- Kiểm tra đường dẫn đến file model và file audio
- Đảm bảo đường dẫn đúng (sử dụng `/` hoặc `\\` tùy hệ điều hành)

### Lỗi: ImportError
- Cài đặt các thư viện cần thiết: `pip install -r requirements.txt`

### Lỗi khi dự đoán
- Kiểm tra file audio có đúng format WAV không
- Đảm bảo file audio không bị hỏng

## Hỗ Trợ

Nếu gặp vấn đề, vui lòng kiểm tra:
1. Đường dẫn file model và audio
2. Các thư viện đã được cài đặt đầy đủ
3. Format và chất lượng file audio

