import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file, Response
from io import BytesIO

app = Flask(__name__)

def process_image(image_data):
    # อ่านภาพจาก BytesIO
    np_image = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # สำเนาภาพต้นฉบับ
    img_original = img.copy()

    # แปลงภาพเป็น Grayscale และใช้ Bilateral Filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    # ค้นหา Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = biggest_contour(contours)

    if biggest.size != 0:
        cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

        points = biggest.reshape(4, 2)
        input_points = np.zeros((4, 2), dtype="float32")

        # จัดเรียงจุด
        points_sum = points.sum(axis=1)
        input_points[0] = points[np.argmin(points_sum)]
        input_points[3] = points[np.argmax(points_sum)]

        points_diff = np.diff(points, axis=1)
        input_points[1] = points[np.argmin(points_diff)]
        input_points[2] = points[np.argmax(points_diff)]

        # แปลง Perspective
        converted_points = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])
        matrix = cv2.getPerspectiveTransform(input_points, converted_points)
        img_output = cv2.warpPerspective(img_original, matrix, (1920, 1080))

        return img_output

    # ถ้าไม่พบ Contours ส่งภาพต้นฉบับกลับไป ส่งกลับไป
    return img

def biggest_contour(contours):
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
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # อ่านข้อมูลไฟล์ภาพ
    image_data = file.read()

    # ประมวลผลภาพ
    processed_image = process_image(image_data)

    # แปลงภาพเป็น BytesIO เพื่อตอบกลับ
    _, buffer = cv2.imencode('.jpg', processed_image)
    response = BytesIO(buffer)

    # ส่งภาพกลับไป
    return Response(response.getvalue(), mimetype='image/jpeg')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
