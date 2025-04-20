import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8_trained_model.pt").to('cuda')

# Print class names
print("Model classes:", model.names)

# Setup directories
save_directory = "cheating_frames"
log_file = "detection_log.csv"
os.makedirs(save_directory, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
confidence_threshold = 0.4  

# Cooldown time to avoid redundant detections
cooldown_time = 2  
last_detection_time = {}


# Load existing log or create a new one
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
else:
    df = pd.DataFrame(columns=["ID", "Date", "Time", "Category", "Details", "Image Path"])

# Detection ID counter
detection_id = len(df) + 1  

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
                date, current_time = time.strftime("%B %d, %Y"), time.strftime("%I:%M %p")
                details = f"{class_name} detected with {confidence:.2f} confidence"
            
                df.loc[len(df)] = [detection_id, date, current_time, class_name, details, f"cheating_frames/{filename}"]
                detection_id += 1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show detection frame
    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save logs to CSV
df.to_csv(log_file, index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()
