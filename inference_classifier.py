import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if necessary

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label mapping (A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}  # 0 → 'A', 1 → 'B', ..., 25 → 'Z'

detected_text = ""  # Store detected letters

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't access the webcam. Try a different index.")
        break  # Exit if webcam isn't working

    H, W, _ = frame.shape  # Get frame dimensions

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect hand landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize the landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize X
                data_aux.append(y - min(y_))  # Normalize Y

        # Bounding box around hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Debugging: Check feature length before prediction
        print(f"Extracted Features: {len(data_aux)} (Expected: 84)")

        # Ensure correct feature count before prediction
        if len(data_aux) not in [42, 84]:
            print("Skipping prediction due to unexpected feature length.")
            continue  

        # If only one hand is detected, duplicate the features
        if len(data_aux) == 42:
            data_aux *= 2  # Duplicate features to match 84

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = int(prediction[0])

        # Get predicted character safely
        predicted_character = labels_dict.get(predicted_label, f"Unknown({predicted_label})")
        print(f"Predicted: {predicted_character}")

        # Update detected text
        detected_text += predicted_character  # Append new character

        # Display prediction on screen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display detected letters in a box
    cv2.rectangle(frame, (50, 50), (W - 50, 120), (255, 255, 255), -1)  # White box
    cv2.putText(frame, detected_text, (60, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show the output frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
