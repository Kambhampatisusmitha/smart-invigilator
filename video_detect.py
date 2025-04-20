import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
model = YOLO("cheating_yolov8x.pt").to('cuda')

# Print class names
print("Model classes:", model.names)

# Setup directories
save_directory = "cheating_frames"
log_file = "detection_log.csv"
os.makedirs(save_directory, exist_ok=True)

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

# Load video
video_path =  "C:\\Users\\lonym\\Videos\\vlc-record-2025-02-20-19h50m10s-exam.mp4-.avi"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer for output video with detections
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process frames from the video
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Run YOLO inference on the frame
    results = model(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

        if class_name != "Normal" and confidence > confidence_threshold:
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

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Optionally, show the frame with detections
    cv2.imshow('Detected Frame', frame)
    
    # Press 'q' to quit early if desired
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the log to CSV
df.to_csv(log_file, index=False)

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")
