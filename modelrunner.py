from keras.preprocessing.image import img_to_array
import cv2
import imutils
import dlib
from keras.models import load_model
import numpy as np
from scipy.spatial import distance as dist

# Paths
video_file_name = "sample/live_vid2.mp4"
video_emotion_model_path = 'models/model_num.hdf5'
shape_predictor_path = 'models/shape_predictor_68_face_landmarks .dat'

# Config
use_live_video = True

# Load models
emotion_classifier = load_model(video_emotion_model_path, compile=False)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# EAR Calculation for blink detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Draw emotion bars directly next to the face with improved colors
def draw_emotion_overlay(frame, preds, emotions, x, y, w, h):
    bar_x = x + w + 10
    bar_y = y
    bar_height = 20
    max_width = 150

    for (emotion, prob) in zip(emotions, preds):
        bar_width = int(prob * max_width)

        # Soft blue bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 150, 255), -1)

        # Text with slight black background for readability
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_width, bar_y + bar_height), (0, 0, 0, 100), -1)
        cv2.putText(frame, f"{emotion}: {prob*100:.1f}%", (bar_x + 5, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        bar_y += 25

# Video capture
if use_live_video:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_file_name)

# Blink counter variables
blink_counter = 0
total_blinks = 0
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3

# Main loop
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) == 0:
            cv2.putText(frame, "Not-Attentive (No Face Detected)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for face in faces:
            landmarks = predictor(gray, face)

            # Eyes for blink detection
            left_eye = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42) ]
            right_eye = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48) ]

            # EAR calculation
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    total_blinks += 1
                blink_counter = 0

            # Face box coordinates
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # Extract face ROI for emotion detection
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            preds = emotion_classifier.predict(face_roi)[0]
            label = EMOTIONS[preds.argmax()]

            # Draw face rectangle (soft blue)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 255), 2)

            # Draw main label (Attentive with emotion)
            label_text = f"Attentive ({label})"
            cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 0, 0), -1)  # black background
            cv2.putText(frame, label_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw emotion bars beside face
            draw_emotion_overlay(frame, preds, EMOTIONS, x, y, w, h)

            # Draw facial landmarks (yellow dots)
            for n in range(68):
                x_lm, y_lm = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x_lm, y_lm), 2, (0, 255, 255), -1)

        # Draw blink counter (light green)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Student Attention Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()
