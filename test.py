import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# 1. Tải mô hình đã huấn luyện
MODEL_PATH = "ResNet50_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy file mô hình: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("-> Đã tải mô hình ResNet50 thành công!")

# Tự động xác định số lượng class đầu ra của mô hình
num_model_classes = model.output_shape[-1]
print(f"-> Số lượng đầu ra (classes) của mô hình: {num_model_classes}")

# Danh sách 3 nhãn chuẩn khớp hoàn toàn với lúc train mô hình thực tế
CLASS_NAMES = [
    'backwardbadposture', 
    'forwardbadposture', 
    'goodposture'
][:num_model_classes]

print(f"-> Danh sách nhãn đồng bộ: {CLASS_NAMES}")

# 2. Từ điển ánh xạ nhãn COCO sang nhãn mô hình
COCO_TO_MODEL_MAP = {
    'leaning_backward': 'backwardbadposture',
    'leaning_forward': 'forwardbadposture',
    'slouch': 'forwardbadposture',          
    'good_posture': 'goodposture',
    'People-Sitting-Posture': 'goodposture',
    'backwardbadposture': 'backwardbadposture',
    'forwardbadposture': 'forwardbadposture',
    'goodposture': 'goodposture'
}

# 3. Đọc file COCO JSON từ thư mục test
TEST_DIR = "test"
ANNOTATION_FILE = os.path.join(TEST_DIR, "_annotations.coco.json")

if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"Không tìm thấy file annotation COCO tại: {ANNOTATION_FILE}")

with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
images_dict = {img['id']: img['file_name'] for img in coco_data['images']}

image_to_cat = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    cat_id = ann['category_id']
    image_to_cat[img_id] = cat_id

total_images = 0
correct_predictions = 0

print("\n--- BẮT ĐẦU ĐÁNH GIÁ (TESTING) ---")

for img_id, file_name in images_dict.items():
    if img_id not in image_to_cat:
        continue
        
    cat_id = image_to_cat[img_id]
    coco_label = categories.get(cat_id, "")
    
    true_label = COCO_TO_MODEL_MAP.get(coco_label, coco_label)
        
    img_path = os.path.join(TEST_DIR, file_name)
    if not os.path.exists(img_path):
        continue
        
    total_images += 1
    
    # Tiền xử lý ảnh về kích thước 224x224 chuẩn đầu vào mô hình
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Dự đoán
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    
    if pred_idx < len(CLASS_NAMES):
        predicted_label = CLASS_NAMES[pred_idx]
    else:
        predicted_label = f"Class_{pred_idx}"
        
    confidence = float(preds[pred_idx]) * 100.0
    
    # Ngưỡng tin cậy 50%
    if confidence < 50.0:
        pred_str = f"Unknown ({confidence:.2f}%)"
        is_correct = False
    else:
        pred_str = f"{predicted_label} ({confidence:.2f}%)"
        is_correct = (predicted_label == true_label)
        
    if is_correct:
        correct_predictions += 1
        
    status_str = "[ĐÚNG]" if is_correct else "[SAI]"
    print(f"Ảnh: {file_name} | Thật: {true_label} --> AI đoán: {pred_str} | {status_str}")

# Tổng kết kết quả
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100.0
    print("\n==========================================")
    print(f" Tổng số ảnh kiểm thử hợp lệ: {total_images}")
    print(f" Số ảnh đoán đúng: {correct_predictions}")
    print(f" ĐỘ CHÍNH XÁC (TEST ACCURACY): {accuracy:.2f}%")
    print("==========================================")
else:
    print("⚠️ Không có ảnh nào khớp với nhãn ánh xạ.")