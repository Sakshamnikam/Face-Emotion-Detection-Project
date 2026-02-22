import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

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
    video_label.configure(image="")

def update_frame():
    global cap, running

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

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

app.mainloop()