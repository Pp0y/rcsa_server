import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

# 📂 โฟลเดอร์หลักสำหรับเก็บภาพ
BASE_FOLDER = "processed_images"
os.makedirs(BASE_FOLDER, exist_ok=True)

def ensure_folder_exists(folder_path):
    """สร้างโฟลเดอร์ถ้ายังไม่มี"""
    os.makedirs(folder_path, exist_ok=True)

def generate_unique_filename(folder_path, base_filename, ext=".jpg"):
    """สร้างชื่อไฟล์ที่ไม่ซ้ำในโฟลเดอร์"""
    filename = f"{base_filename}{ext}"
    file_path = os.path.join(folder_path, filename)
    
    counter = 1
    while os.path.exists(file_path):
        filename = f"{base_filename}_{counter}{ext}"
        file_path = os.path.join(folder_path, filename)
        counter += 1

    return filename, file_path

def process_image(image_data):
    """ประมวลผลภาพ (ทำให้เป็นขาวดำและแปลง Perspective)"""
    np_image = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("❌ Invalid Image Data")

    img_original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = biggest_contour(contours)

    if biggest.size != 0:
        cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

        points = biggest.reshape(4, 2)
        input_points = np.zeros((4, 2), dtype="float32")

        points_sum = points.sum(axis=1)
        input_points[0] = points[np.argmin(points_sum)]
        input_points[3] = points[np.argmax(points_sum)]

        points_diff = np.diff(points, axis=1)
        input_points[1] = points[np.argmin(points_diff)]
        input_points[2] = points[np.argmax(points_diff)]

        converted_points = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])
        matrix = cv2.getPerspectiveTransform(input_points, converted_points)
        img_output = cv2.warpPerspective(img_original, matrix, (1920, 1080))

        return img_output

    return img

def biggest_contour(contours):
    """ค้นหา Contour ที่ใหญ่ที่สุด (ใช้เพื่อแปลง Perspective)"""
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

@app.route('/process', methods=['POST'])
def process():
    """รับภาพจาก Unity, ประมวลผล, และบันทึกไฟล์แบบไม่ซ้ำ"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_data = file.read()

    try:
        processed_image = process_image(image_data)
    except ValueError:
        return jsonify({"error": "Invalid Image Data"}), 400

    # 📌 ตรวจสอบชื่อไฟล์ เพื่อเลือกโฟลเดอร์บันทึก
    original_filename = file.filename.lower()  # แปลงเป็นตัวพิมพ์เล็กเพื่อป้องกันปัญหา case-sensitive
    base_filename, file_extension = os.path.splitext(original_filename)

    if base_filename.startswith("air"):
        folder_name = "air"
    elif base_filename.startswith("earth"):
        folder_name = "earth"
    else:
        folder_name = "others"  # สำหรับไฟล์ที่ไม่ใช่ air หรือ earth

    # 📂 กำหนดเส้นทางโฟลเดอร์
    save_folder = os.path.join(BASE_FOLDER, folder_name)
    ensure_folder_exists(save_folder)  # ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี

    # 🕒 สร้างชื่อไฟล์ใหม่ให้ไม่ซ้ำ
    unique_filename, save_path = generate_unique_filename(save_folder, base_filename, ".jpg")

    # 💾 บันทึกภาพลงโฟลเดอร์ที่เลือก
    cv2.imwrite(save_path, processed_image)

    return jsonify({
        "message": "Image processed successfully!",
        "category": folder_name,
        "file_name": unique_filename,
        "file_path": save_path
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
