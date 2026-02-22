import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# ---------------- SETTINGS ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------- LOAD MODEL ----------------
model = load_model(resource_path("model/emotion_model.h5"))

face_cascade = cv2.CascadeClassifier(
    resource_path("haarcascade/haarcascade_frontalface_default.xml")
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = None
running = False

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

    process_and_display(frame)
    video_label.after(10, update_frame)

# ---------------- IMAGE UPLOAD FUNCTION ----------------
def upload_image():
    global running
    running = False

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    frame = cv2.imread(file_path)
    process_and_display(frame)

# ---------------- COMMON PROCESS FUNCTION ----------------
def process_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face)
        max_index = np.argmax(prediction)
        emotion = emotion_labels[max_index]
        confidence = prediction[0][max_index] * 100

        label_text = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    # Convert BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    img = Image.fromarray(frame)

    # 🔥 Resize to fit GUI
    img = img.resize((700, 450))   # Change size as you like

    # 🔥 Use CTkImage instead of ImageTk
    ctk_image = ctk.CTkImage(light_image=img,
                             dark_image=img,
                             size=(700, 450))

    video_label.configure(image=ctk_image)
    video_label.image = ctk_image

# ---------------- GUI ----------------
app = ctk.CTk()
app.title("Face Emotion Detection")
app.geometry("900x650")
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
start_btn.grid(row=0, column=0, padx=15)

stop_btn = ctk.CTkButton(
    btn_frame,
    text="Stop Camera",
    width=150,
    fg_color="red",
    hover_color="#aa0000",
    command=stop_camera
)
stop_btn.grid(row=0, column=1, padx=15)

upload_btn = ctk.CTkButton(
    btn_frame,
    text="Upload Image",
    width=150,
    fg_color="#1f6aa5",
    command=upload_image
)
upload_btn.grid(row=0, column=2, padx=15)

app.mainloop()