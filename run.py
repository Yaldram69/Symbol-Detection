import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
from sklearn.preprocessing import LabelEncoder
class_name = ['Thumbs Up','Thumbs Down','Peace','Super','You','Call Me','Rock','Love You','High Five','None']
mp_holistic = mp.solutions.holistic
mp_drawings = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable= False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

le = LabelEncoder()
left_hand_model = tf.keras.models.load_model("D:\PythonProjects\Symbol Detection\hand-sign-detection-main\left_hand_gesturev3.h5")
right_hand_model = tf.keras.models.load_model("D:\PythonProjects\Symbol Detection\hand-sign-detection-main\handgesture_rightv2.h5")
label_names = []
for i in range(10):
    name = le.inverse_transform([i])
    label_names.append(name[0])
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, ) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        # Get the frame and results from Mediapipe
        image, results = mediapipe_detection(frame, holistic)

        if results.left_hand_landmarks:
            mp_drawings.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawings.DrawingSpec(color=(200, 22, 76), thickness=2, circle_radius=2),
                                       mp_drawings.DrawingSpec(color=(100, 44, 100), thickness=2, circle_radius=1))
            left_arr = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            left_arr_reshaped = np.expand_dims(left_arr, 0)
            predicted_left = left_hand_model.predict(left_arr_reshaped)
            left_gesture = label_names[np.argmax(predicted_left)]
            print(f"LEFT: {left_gesture}")

        if results.right_hand_landmarks:
            mp_drawings.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawings.DrawingSpec(color=(166, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawings.DrawingSpec(color=(188, 66, 77), thickness=2, circle_radius=1))
            right_arr = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            right_arr_reshaped = np.expand_dims(right_arr, 0)
            predicted_right = right_hand_model.predict(right_arr_reshaped)
            right_gesture = label_names[np.argmax(predicted_right)]
            print(f"RIGHT: {right_gesture}")

        image = cv2.flip(image, 1)
        if results.left_hand_landmarks:
            cv2.putText(image, left_gesture, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (188, 66, 77), 3, cv2.LINE_AA)
        if results.right_hand_landmarks:
            cv2.putText(image, right_gesture, (400, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (188, 66, 77), 3, cv2.LINE_AA)
        if not results.left_hand_landmarks and results.right_hand_landmarks:
            cv2.putText(image, " ", (15, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow('sample', image)

        # Breaking Gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
