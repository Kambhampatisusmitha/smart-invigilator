import cv2
from ultralytics import YOLO
import os
import time

# Step 1: Load the trained YOLOv8 model
model = YOLO("yolov8_trained_model.pt")  # Load your trained YOLOv8 model

# Print class names of the model
print("Model classes:", model.names)

# Step 2: Initialize the webcam or video stream (0 for default webcam)
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam or replace with a video file path

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 3: Define a threshold for cheating detection 
confidence_threshold = 0.4  

# Step 4: Define the folder to save images
save_directory = "cheating_frames"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)  # Create the folder if it doesn't exist

# Step 5: Loop to capture frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Step 6: Run inference on the current frame
    results = model(frame)

    # Step 7: Debugging - Show raw results to check what's being detected
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # Extract bounding box coordinates
        confidence = result.conf[0].item()  # Confidence score
        class_id = int(result.cls[0].item())  # Class ID
        class_name = model.names[class_id]  # Get the class name
        
        # Debugging output for every detected object
        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        # If the detected object is a "person" and confidence is above threshold
        if class_name != "normal" and confidence > confidence_threshold:
            # Mark that cheating is detected
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # Timestamp for unique filename
            filename = os.path.join(save_directory, f"{class_name}_{timestamp}.jpg")  # Full path

            # Save the frame to disk
            cv2.imwrite(filename, frame)
            print(f"Cheating detected! Frame saved as {filename}")
              # Exit the loop after saving the first detected frame (optional)

    # Step 8: Draw bounding boxes around detected objects
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        class_id = int(result.cls[0].item())
        class_name = model.names[class_id]
        confidence = result.conf[0].item()

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 9: Display the frame with bounding boxes (for visual feedback)
    cv2.imshow('YOLOv8 Live Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 10: Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
