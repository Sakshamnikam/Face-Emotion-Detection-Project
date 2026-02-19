import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tkinter import filedialog
from collections import deque
emotion_buffer = deque(maxlen=5)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------- LOAD MODEL ----------------
model = load_model("model/emotion_model.h5")

emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]

CONFIDENCE_THRESHOLD = 0.30
FACE_CONFIDENCE_THRESHOLD = 0.6

# ---------------- LOAD DNN FACE DETECTOR ----------------
face_net = cv2.dnn.readNetFromCaffe(
    "face_dnn/deploy.prototxt",
    "face_dnn/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = None
running = False


# ---------------- EMOTION PREDICTION ----------------
def predict_emotion(face):
    prediction = model.predict(face, verbose=0)[0]
    emotion_index = np.argmax(prediction)

    emotion_buffer.append(emotion_index)

    # Majority vote smoothing
    final_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

    return emotion_labels[final_emotion]




# ---------------- CAMERA FUNCTIONS ----------------
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()


def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    video_label.configure(image="")


def update_frame():
    global cap, running

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    display_frame = process_frame(frame, use_equalization=True)

    img = Image.fromarray(display_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)


# ---------------- IMAGE UPLOAD ----------------
def upload_image():
    global running, cap

    running = False
    if cap:
        cap.release()

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)
    image = cv2.resize(image, (700, 450))
    processed = process_frame(image, use_equalization=True)

    img = Image.fromarray(processed)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


# ---------------- CORE PROCESSING (DNN + EMOTION) ----------------
def process_frame(frame, use_equalization=True):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > FACE_CONFIDENCE_THRESHOLD:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            margin = 30
            startX = max(0, startX - margin)
            startY = max(0, startY - margin)
            endX = min(w, endX + margin)
            endY = min(h, endY + margin)

            face = frame[startY:endY, startX:endX]


            if face.size == 0:
                continue

            # Convert to grayscale (since model expects 1 channel)
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            if use_equalization:
                face_gray = cv2.equalizeHist(face_gray)

            face_gray = cv2.resize(face_gray, (48, 48))

            face_gray = face_gray / 255.0
            face_gray = np.reshape(face_gray, (1, 48, 48, 1))

            emotion = predict_emotion(face_gray)

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            cv2.putText(frame, emotion,
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------- GUI SETUP ----------------
app = ctk.CTk()
app.title("Face Emotion Detection - DNN Version")
app.geometry("900x650")

title = ctk.CTkLabel(app,
                     text="Face Emotion Detection System (DNN)",
                     font=("Segoe UI", 24, "bold"))
title.pack(pady=20)

video_label = ctk.CTkLabel(app, text="")
video_label.pack()

btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=20)

ctk.CTkButton(btn_frame,
              text="Start Camera",
              width=150,
              command=start_camera).grid(row=0, column=0, padx=20)

ctk.CTkButton(btn_frame,
              text="Stop Camera",
              width=150,
              fg_color="red",
              command=stop_camera).grid(row=0, column=1, padx=20)

ctk.CTkButton(btn_frame,
              text="Upload Image",
              width=150,
              command=upload_image).grid(row=1, column=0, columnspan=2, pady=15)

app.mainloop()
