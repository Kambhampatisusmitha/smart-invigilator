import os
import random
import string
import base64
import time
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_mail import Mail, Message
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# MongoDB setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client["ExamSecure"]
users_collection = db["users"]

# Flask-Mail configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("EMAIL_USER")
app.config["MAIL_PASSWORD"] = os.getenv("EMAIL_PASS")
mail = Mail(app)

# Global variables
otp_store = {}
alerts = []
model = YOLO("yolov8_trained_model.pt")
print("Model classes:", model.names)

save_directory = os.path.join("static", "cheating_frames")
log_file = os.path.join("static", "detection_log.csv")
os.makedirs(save_directory, exist_ok=True)

confidence_threshold = 0.4
cooldown_time = 5
last_detection_time = {}

df = pd.read_csv(log_file) if os.path.exists(log_file) else pd.DataFrame(columns=["ID", "Class", "Confidence", "Timestamp"])
detection_id = len(df) + 1

# Helper functions
def generate_otp():
    return "".join(random.choices(string.digits, k=6))

def get_user_role_from_db(email):
    user = users_collection.find_one({"email": email})
    return user["role"] if user else None

def otp_is_valid(email, otp):
    return otp_store.get(email) == otp

# Authentication routes
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email, password, role = data.get("email"), data.get("password"), data.get("role")

    if not email or not password or not role:
        return jsonify({"error": "Email, password, and role are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email is already registered"}), 400

    if len(password) < 8 or not any(c.isupper() for c in password) or not any(c.islower() for c in password) or not any(c.isdigit() for c in password):
        return jsonify({"error": "Password must be at least 8 characters with uppercase, lowercase, and a number"}), 400

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({"email": email, "password": hashed_password, "role": role})

    otp = generate_otp()
    otp_store[email] = otp
    msg = Message("Your OTP for Registration", sender=app.config["MAIL_USERNAME"], recipients=[email])
    msg.body = f"Your OTP is: {otp}. It will expire in 10 minutes."
    mail.send(msg)

    return jsonify({"message": "User registered. Please check your email for OTP"}), 200

@app.route("/verify-registration-otp", methods=["POST"])
def verify_registration_otp():
    data = request.json
    email, otp = data.get("email"), data.get("otp")

    if otp_is_valid(email, otp):
        del otp_store[email]
        return jsonify({"message": "OTP verified successfully"}), 200
    return jsonify({"error": "Invalid OTP"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email, password, role = data.get("email"), data.get("password"), data.get("role")

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password) or user["role"] != role:
        return jsonify({"error": "Invalid credentials"}), 400

    otp = generate_otp()
    otp_store[email] = otp
    msg = Message("Your OTP for Authentication", sender=app.config["MAIL_USERNAME"], recipients=[email])
    msg.body = f"Your OTP is: {otp}. It will expire in 10 minutes."
    mail.send(msg)

    return jsonify({"message": "Credentials verified. OTP sent"}), 200

@app.route('/verify-login-otp', methods=['POST'])
def verify_login_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")

    if otp_is_valid(email, otp):
        user_role = get_user_role_from_db(email)
        return jsonify({"message": "OTP verified", "role": user_role}), 200
    return jsonify({"error": "Invalid OTP"}), 400

@app.route("/send-otp-for-password", methods=["POST"])
def send_otp_for_password():
    data = request.json
    email = data.get("email")

    if not users_collection.find_one({"email": email}):
        return jsonify({"error": "Email not found"}), 404

    otp = generate_otp()
    otp_store[email] = otp
    msg = Message("Your OTP for Password Reset", sender=app.config["MAIL_USERNAME"], recipients=[email])
    msg.body = f"Your OTP is: {otp}. It will expire in 10 minutes."
    mail.send(msg)

    return jsonify({"message": "OTP sent successfully"}), 200

@app.route("/verify-otp-for-password", methods=["POST"])
def verify_otp_for_password():
    data = request.json
    email, otp = data.get("email"), data.get("otp")

    if otp_is_valid(email, otp):
        return jsonify({"message": "OTP verified successfully"}), 200
    return jsonify({"error": "Invalid OTP"}), 400

@app.route("/update-password", methods=["POST"])
def update_password():
    data = request.json
    email, new_password, confirm_password, otp = data.get("email"), data.get("newPassword"), data.get("confirmNewPassword"), data.get("otp")

    if not otp_is_valid(email, otp):
        return jsonify({"error": "Invalid OTP"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    if len(new_password) < 8 or not any(c.isupper() for c in new_password) or not any(c.islower() for c in new_password) or not any(c.isdigit() for c in new_password):
        return jsonify({"error": "Password must be at least 8 characters with uppercase, lowercase, and a number"}), 400

    hashed_password = generate_password_hash(new_password)
    users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})
    del otp_store[email]

    return jsonify({"message": "Password reset successfully"}), 200

# Render HTML templates
@app.route('/')
def index(): return render_template('index.html')
@app.route('/login')
def login_page(): return render_template('sign in.html')
@app.route('/register')
def register_page(): return render_template('signup.html')
@app.route('/administrator')
def administrator(): return render_template('administrator.html')
@app.route('/staff')
def staff(): return render_template('staff.html')
@app.route('/watchlive')
def watchlive(): return render_template('live_stream.html')
@app.route('/anomalies')
def anomalies(): return render_template('anomalies.html')
@app.route('/examsched')
def examsched(): return render_template('exam-sched.html')
@app.route('/staffexamsched')
def staffexamsched(): return render_template('staff_exam_schedule.html')
@app.route('/signout')
def signout(): return render_template('index.html')
@app.route('/reset')
def reset(): return render_template('forgot-password.html')

# Frame processing endpoint
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global detection_id, alerts
    if not request.json or 'frame' not in request.json:
        return jsonify({"error": "No frame data provided"}), 400

    try:
        frame_data = request.json['frame'].split(',')[1] if 'data:image' in request.json['frame'] else request.json['frame']
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            app.logger.error("Decoded frame is empty or invalid")
            return jsonify({"error": "Invalid image data"}), 400

        annotated_frame = frame.copy()
        results = model(frame)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if class_name != "normal" and confidence > confidence_threshold:
                current_time = time.time()
                if (class_name not in last_detection_time or (current_time - last_detection_time[class_name]) > cooldown_time):
                    last_detection_time[class_name] = current_time
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(save_directory, filename)
                    cv2.imwrite(filepath, frame[y1:y2, x1:x2])

                    alerts.append(f"Cheating detected: {class_name} with confidence {confidence:.2f}")
                    detections.append({"class_name": class_name, "confidence": confidence, "bbox": [x1, y1, x2, y2], "timestamp": timestamp})

        success, buffer = cv2.imencode('.jpg', annotated_frame)
        if not success:
            raise Exception("Could not encode annotated frame")

        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"status": "success", "detections": detections, "annotated_frame": encoded_frame, "timestamp": datetime.now().isoformat()})

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/alerts')
def get_alerts():
    global alerts
    clear = request.args.get('clear', '').lower() == 'true'
    response_alerts = alerts.copy()
    if clear:
        alerts = []
    return jsonify(response_alerts)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
