# step 1: Import the ultralytics

from ultralytics import YOLO

# Step 2: Define paths of your dataset

dataset_path = "Dataset" 

# Step 3: Initialize the YOLOv8 model
model = YOLO("yolov8x.pt")  

# Step 4: Train the model
results = model.train(
    data=f"{dataset_path}/data.yaml", 
    epochs=50,                         
    imgsz=640,                         
    batch=16,                         
)

# Save the model after training
model.save("cheating_yolov8x.pt")


