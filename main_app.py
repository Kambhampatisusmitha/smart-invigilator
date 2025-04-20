import cv2
import os
import time
import pandas as pd
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("C:/Users/rhyds/OneDrive/Desktop/smart/Smart Invigilator/yolov8_trained_model.pt")
print("Model classes:", model.names)

# Setup directories
save_directory = "cheating_frames"
log_file = "detection_log.csv"
os.makedirs(save_directory, exist_ok=True)


# Confidence threshold
confidence_threshold = 0.4

# Cooldown time to avoid redundant detections
cooldown_time = 5
last_detection_time = {}

# Load existing log or create a new one
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
else:
    df = pd.DataFrame(columns=["ID", "Date", "Time", "Category", "Details", "Image Path"])

# Detection ID counter
detection_id = len(df) + 1

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def generate_frames():
    global detection_id  # Access the detection_id from the global scope
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

            if class_name != "normal" and confidence > confidence_threshold:
                current_time = time.time()
                if class_name not in last_detection_time or (current_time - last_detection_time[class_name]) > cooldown_time:
                    last_detection_time[class_name] = current_time

                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(save_directory, filename)
                    cv2.imwrite(filepath, frame[y1:y2, x1:x2])  # Save cropped image

                    # Log data
                    date, current_time_str = time.strftime("%B %d, %Y"), time.strftime("%I:%M %p")
                    details = f"{class_name} detected with {confidence:.2f} confidence"
                    
                    df.loc[len(df)] = [detection_id, date, current_time_str, class_name, details, f"cheating_frames/{filename}"]
                    df.to_csv(log_file,index=False)
                    detection_id += 1


        # Draw bounding box
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = box.conf[0].item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG and send to frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('live_stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)

