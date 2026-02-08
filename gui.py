import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tkinter import filedialog

# ---------------- SETTINGS ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------- LOAD MODEL ----------------
model = load_model("model/emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = None
running = False

CONFIDENCE_THRESHOLD = 0.6

# ---------------- FUNCTIONS ----------------
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
    video_label.configure(image="", text="")

def predict_emotion(face):
    prediction = model.predict(face, verbose=0)[0]
    confidence = np.max(prediction)
    emotion_index = np.argmax(prediction)

    if confidence >= CONFIDENCE_THRESHOLD:
        return emotion_labels[emotion_index]
    else:
        return "Uncertain"

def update_frame():
    global cap, running

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (500, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        pad = int(0.1 * w)
        face = gray[
            max(0, y-pad):y+h+pad,
            max(0, x-pad):x+w+pad
        ]

        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        emotion = predict_emotion(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame, emotion, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,0), 2
        )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

    video_label.after(10, update_frame)

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

    img = cv2.imread(file_path)
    img = cv2.resize(img, (500, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        video_label.configure(text="No face detected", image="")
        return

    for (x, y, w, h) in faces:
        pad = int(0.1 * w)
        face = gray[
            max(0, y-pad):y+h+pad,
            max(0, x-pad):x+w+pad
        ]

        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        emotion = predict_emotion(face)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            img, emotion, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,0), 2
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="")

# ---------------- GUI ----------------
app = ctk.CTk()
app.title("Face Emotion Detection")
app.geometry("900x600")
app.resizable(False, False)

title = ctk.CTkLabel(
    app,
    text="Face Emotion Detection System",
    font=("Segoe UI", 26, "bold")
)
title.pack(pady=20)

video_label = ctk.CTkLabel(app, text="")
video_label.pack()

btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=20)

start_btn = ctk.CTkButton(
    btn_frame,
    text="Start Camera",
    width=150,
    command=start_camera
)
start_btn.grid(row=0, column=0, padx=20)

stop_btn = ctk.CTkButton(
    btn_frame,
    text="Stop Camera",
    width=150,
    fg_color="red",
    hover_color="#aa0000",
    command=stop_camera
)
stop_btn.grid(row=0, column=1, padx=20)

upload_btn = ctk.CTkButton(
    btn_frame,
    text="Upload Image",
    width=150,
    command=upload_image
)
upload_btn.grid(row=1, column=0, columnspan=2, pady=15)

app.mainloop()
