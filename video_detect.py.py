import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.h5')

# Define constants
IMG_SIZE = 48
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Ask the user for the video file path
video_path = input("Enter the path to the video file: ")

# Initialize the video capture with the given video path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the video is finished

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = gray_frame[y:y + h, x:x + w]
        
        # Resize the face to 48x48 for model prediction
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # Normalize the pixel values and reshape for model input
        face_resized = face_resized / 255.0
        face_resized = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Reshape to (1, 48, 48, 1)

        # Make a prediction
        prediction = model.predict(face_resized)
        max_index = np.argmax(prediction[0])
        predicted_emotion = EMOTIONS[max_index]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the predictions
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
