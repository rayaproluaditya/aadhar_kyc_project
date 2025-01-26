from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to decode base64 image
def decode_image(image_data):
    image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

# Function to compare two images using OpenCV
def compare_faces(img1, img2):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

    if len(faces1) == 0 or len(faces2) == 0:
        return False

    (x, y, w, h) = faces1[0]
    face1 = gray1[y:y+h, x:x+w]
    
    (x, y, w, h) = faces2[0]
    face2 = gray2[y:y+h, x:x+w]

    # Resize to same size
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))

    # Compute pixel difference
    difference = cv2.absdiff(face1, face2)
    if np.mean(difference) < 50:
        return True
    else:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    aadhaar_number = request.form['aadhaar']
    aadhaar_photo = request.files['aadhaarPhoto']
    live_photo_data = request.form['livePhoto']

    # Save Aadhaar image
    aadhaar_filename = secure_filename(aadhaar_number + '.jpg')
    aadhaar_path = os.path.join(app.config['UPLOAD_FOLDER'], aadhaar_filename)
    aadhaar_photo.save(aadhaar_path)

    # Decode the live photo
    live_photo = decode_image(live_photo_data)

    # Load the uploaded Aadhaar photo
    aadhaar_img = cv2.imread(aadhaar_path)

    if aadhaar_img is None:
        return jsonify({"message": "Error loading Aadhaar image"}), 400

    if compare_faces(live_photo, aadhaar_img):
        return jsonify({"message": "Authentication Successful"})
    else:
        return jsonify({"message": "Authentication Failed"}), 400

if __name__ == '__main__':
    app.run(debug=True)