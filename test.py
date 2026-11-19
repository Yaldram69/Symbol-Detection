import cv2
import numpy as np
import tensorflow as tf
from time import sleep

# Load the model and label encoder
model = tf.keras.models.load_model("mobilenet_logo_detection_model.h5")
label_encoder = np.load("label_encoder.npy", allow_pickle=True)

# Parameters for display and detection
CONFIDENCE_THRESHOLD = 0.7
BUFFER_FRAMES = 10  # Frames to confirm a detection
FRAME_HEIGHT, FRAME_WIDTH = 128, 128  # Model input size

# Initialize a buffer to store recent predictions
buffer = []


# Preprocess the frame to match model input
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


# Function to add bounding box and label
def draw_detection(frame, label, confidence):
    # Draw bounding box (around the entire frame here for simplicity)
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 255, 0), 2)
    # Display label and confidence
    text = f"{label}: {confidence * 100:.2f}%"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    input_frame = preprocess_frame(frame)
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Store high-confidence predictions in buffer
    if confidence > CONFIDENCE_THRESHOLD:
        buffer.append(predicted_class)
    else:
        buffer.append(None)

    # Maintain buffer size
    if len(buffer) > BUFFER_FRAMES:
        buffer.pop(0)

    # Confirm detection if majority in buffer are the same logo
    if buffer.count(predicted_class) > (BUFFER_FRAMES // 2):
        logo_name = label_encoder[predicted_class]
        frame = draw_detection(frame, logo_name, confidence)

        # Optional: Uncomment to add audio feedback (requires `playsound` library)
        # from playsound import playsound
        # playsound('path/to/sound_file.mp3')  # Add audio feedback for detection

    # Display the video feed
    cv2.imshow("Enhanced Logo Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
