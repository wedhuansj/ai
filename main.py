import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder=".")
CORS(app)

# 1. Khai báo danh sách 7 nhãn chuẩn từ Roboflow Dataset
CLASS_NAMES = [
    'Unlabeled',
    'backwardbadposture',
    'backwardbadposture forwardbadposture',
    'backwardbadposture goodposture',
    'forwardbadposture',
    'forwardbadposture goodposture',
    'goodposture'
]

# Danh sách tên nhãn hiển thị thân thiện trên UI
FRIENDLY_NAMES = {
    'Unlabeled': 'Chưa xác định',
    'backwardbadposture': 'Ngả lưng ra sau (Tư thế sai)',
    'backwardbadposture forwardbadposture': 'Sai tư thế tổng hợp (Ngả sau / Cúi tới)',
    'backwardbadposture goodposture': 'Ngả lưng nhẹ (Gần chuẩn)',
    'forwardbadposture': 'Cúi đầu / Rướn người về trước (Tư thế sai)',
    'forwardbadposture goodposture': 'Cúi nhẹ (Gần chuẩn)',
    'goodposture': 'Tư thế ngồi chuẩn (Ngồi tốt)'
}

MODEL_PATH = "ResNet50_model.keras"
model = None

# 2. Tải mô hình ResNet50 khi khởi chạy Backend
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("-> Đã tải thành công mô hình ResNet50_model.keras")
    except Exception as e:
        print(f"-> Lỗi khi tải mô hình: {e}")
else:
    print(f"-> KHÔNG TÌM THẤY TỆP MÔ HÌNH: {MODEL_PATH}")

def preprocess_image(image: Image.Image):
    """Tiền xử lý ảnh về kích thước 224x224 chuẩn đầu vào ResNet50"""
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False, 
            'error': 'Mô hình AI chưa được tải thành công trên Server. Vui lòng kiểm tra lại tệp ResNet50_model.keras!'
        }), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy tệp ảnh gửi lên!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn tệp ảnh nào!'}), 400

    try:
        # Đọc ảnh từ request
        image = Image.open(file.stream)
        processed_img = preprocess_image(image)

        # Mô hình dự đoán
        predictions = model.predict(processed_img)[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx]) * 100.0

        raw_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"
        friendly_label = FRIENDLY_NAMES.get(raw_label, raw_label)

        # Trả về kết quả dự đoán chi tiết
        return jsonify({
            'success': True,
            'label_raw': raw_label,
            'label_friendly': friendly_label,
            'confidence': round(confidence, 2),
            'all_predictions': {
                FRIENDLY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]): round(float(predictions[i]) * 100.0, 2)
                for i in range(min(len(predictions), len(CLASS_NAMES)))
            }
        })

    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Lỗi trong quá trình xử lý hình ảnh: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Khởi chạy ứng dụng Web StudyCheck AI...")
    app.run(host='0.0.0.0', port=5000, debug=True)