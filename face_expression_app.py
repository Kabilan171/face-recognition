import streamlit as st
import cv2
from fer import FER
import datetime
import numpy as np

# Initialize FER detector
detector = FER()

# App state
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'start'
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# Set page layout
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Face Expression Detection</h1>", unsafe_allow_html=True)

def reset_to_home():
    st.session_state.app_state = 'start'
    st.session_state.run_camera = False

def run_expression_detection():
    st.session_state.app_state = 'detect'
    st.session_state.run_camera = True

# Start page with large Play button
if st.session_state.app_state == 'start':
    st.markdown("<div style='display:flex; justify-content:center; align-items:center; height:70vh;'>"
                "<button onclick='window.location.reload();' style='font-size:50px; padding:20px 40px; border:none; border-radius:15px; background-color:#4CAF50; color:white;'>▶ Start</button>"
                "</div>", unsafe_allow_html=True)
    st.button("Start", on_click=run_expression_detection)

# Detection page (live webcam feed)
elif st.session_state.app_state == 'detect':
    col1, col2 = st.columns([3, 1])
    expression_log = col2.empty()
    stop_button = col2.button("Stop", on_click=reset_to_home)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        col1.error("Could not open webcam.")
    else:
        stframe = col1.empty()
        result_log = []
        last_logged_time = ""

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        while st.session_state.run_camera:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                emotions = detector.detect_emotions(rgb_face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if emotions:
                    emotion_scores = emotions[0]["emotions"]
                    top_expression = max(emotion_scores, key=emotion_scores.get)
                    top_score = emotion_scores[top_expression]

                    current_time = datetime.datetime.now().strftime('%H:%M:%S')

                    # Log only if timestamp has changed (i.e., per second)
                    if current_time != last_logged_time:
                        last_logged_time = current_time
                        display_text = f"{current_time} - {top_expression} ({top_score:.2f})"
                        result_log.append(display_text)
                        if len(result_log) > 10:
                            result_log.pop(0)

                        expression_log.markdown("### Expression Log\n" + "\n\n".join(result_log[::-1]))

                    cv2.putText(frame, f"{top_expression} ({top_score:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_container_width=True)

        video_capture.release()
