from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json

app = Flask(__name__)
app.secret_key = "12345"  # You can manually set your secret key

# Load trained sign language model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize OpenCV & MediaPipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label mapping (A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}

# User database
USER_DATA_FILE = "users.json"

# Load user data
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user data
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)

def detect_sign():
    """ Generator function to process video frames and return detections """
    while True:
        data_aux, x_, y_ = [], [], []
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) == 42:
                data_aux *= 2  # Ensure 84 features for model input

            if len(data_aux) == 84:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_label = int(prediction[0])
                predicted_character = labels_dict.get(predicted_label, "Unknown")

            # Display predicted sign
            cv2.putText(frame, predicted_character, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        users = load_users()

        if email in users:
            return render_template('sign_up.html', error="User already exists! Try logging in.")

        users[email] = {"password": password}
        save_users(users)
        return redirect(url_for('login'))

    return render_template('sign_up.html', error=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')  # Use .get() to avoid KeyError
        password = request.form.get('password')

        if not email or not password:
            return "Error: Missing email or password", 400

        # Load users from JSON file
        users = load_users()

        # Check if user exists and password matches
        if email in users and users[email]['password'] == password:
            session['user'] = email  # Set session
            return redirect(url_for('home'))  # Redirect to home
        else:
            return "Invalid credentials", 401

    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', user=session['user'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/sign_detection')
def sign_detection():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('sign_detection.html')

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(detect_sign(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign_representation', methods=['GET', 'POST'])
def sign_representation():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        text = request.form.get('text_input')  # Get the input text
        return render_template('sign_representation.html', text=text)
    return render_template('sign_representation.html', text="")

if __name__ == '__main__':
    app.run(debug=True)
