import os
import numpy as np

# Đường dẫn tới thư mục MFCC
base_dir = "features/mfcc"
output_features = []
output_labels = []

# Duyệt qua từng thư mục con (tức là mỗi loại tiếng khóc)
for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            feature_path = os.path.join(label_dir, file)
            feature = np.load(feature_path)
            output_features.append(feature)
            output_labels.append(label)

print(f"Tổng số mẫu: {len(output_features)}")

# Chuyển sang numpy array
X = np.array(output_features)
y = np.array(output_labels)

# Lưu lại
np.save(os.path.join(base_dir, "features.npy"), X)
np.save(os.path.join(base_dir, "labels.npy"), y)

print("✅ Đã lưu features.npy và labels.npy trong", base_dir)
