import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder=".")
CORS(app)

# 1. Khai báo danh sách nhãn chuẩn theo đúng thứ tự lúc train mô hình
CLASS_NAMES = [
    'backwardbadposture',
    'forwardbadposture',
    'goodposture'
]

# Danh sách tên nhãn hiển thị thân thiện trên UI
FRIENDLY_NAMES = {
    'backwardbadposture': 'Ngả ra sau (Sai tư thế)',
    'forwardbadposture': 'Cúi về trước (Sai tư thế)',
    'goodposture': 'Tư thế ngồi chuẩn (Ngồi đúng)'
}

MODEL_PATH = "ResNet50_model.keras"
model = None

# Ngưỡng độ tin cậy tối thiểu để AI kết luận (75%)
CONFIDENCE_THRESHOLD = 75.0

# Thư mục lưu trữ các mẫu feedback (hard cases)
HARD_CASES_DIR = "hard_cases"
os.makedirs(HARD_CASES_DIR, exist_ok=True)

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

        # Kiểm tra ngưỡng tin cậy theo yêu cầu
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': True,
                'is_uncertain': True,
                'label_raw': raw_label,
                'label_friendly': "Chưa chắc chắn - hãy điều chỉnh góc camera thấy rõ lưng và đầu",
                'confidence': round(confidence, 2),
                'all_predictions': {
                    FRIENDLY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]): round(float(predictions[i]) * 100.0, 2)
                    for i in range(min(len(predictions), len(CLASS_NAMES)))
                }
            })

        return jsonify({
            'success': True,
            'is_uncertain': False,
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

@app.route('/feedback', methods=['POST'])
def feedback():
    """Lưu ảnh và thông tin phản hồi của người dùng vào thư mục hard_cases"""
    try:
        user_feedback = request.form.get('feedback') # 'correct' hoặc 'incorrect'
        predicted_label = request.form.get('label_raw')
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Không tìm thấy tệp ảnh để lưu feedback!'}), 400
            
        file = request.files['file']
        
        timestamp = int(time.time())
        filename = f"{timestamp}_{user_feedback}_{predicted_label}.jpg"
        filename = filename.replace(" ", "_")
        
        file_path = os.path.join(HARD_CASES_DIR, filename)
        file.seek(0)
        file.save(file_path)
        
        print(f"[FEEDBACK SAVED] Đã lưu mẫu phản hồi: {user_feedback} | Nhãn AI: {predicted_label} tại {file_path}")
        
        return jsonify({'success': True, 'message': 'Đã lưu phản hồi và hình ảnh thành công!'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi khi lưu feedback: {str(e)}'}), 500

if __name__ == '__main__':
    print("Khởi chạy ứng dụng Web StudyCheck AI...")
    app.run(host='0.0.0.0', port=5000, debug=True)