🎭 Face Emotion Detection System

Face Emotion Detection is a real-time AI-based computer vision project that detects human emotions using a webcam.
It uses Deep Learning (CNN) and OpenCV to analyze facial expressions and classify emotions accurately.

✨ Features
🎥 Real-Time Emotion Detection
Detects face using webcam
Predicts emotion in real time
Displays emotion label on screen

🧠 Deep Learning Model
CNN-based architecture
Trained using FER-2013 dataset
High accuracy emotion classification

📸 Face Detection
Uses Haar Cascade classifier
Detects frontal faces
Works smoothly in real time

🗂️ Project Structure
Face_Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── model/
│   └── emotion_model.h5
│
├── train_model.py
├── main.py
├── requirements.txt
└── README.md

⚙️ Requirements
🧩 System Requirements
Windows OS
Python 3.10
Webcam

📦 Python Libraries

Install all dependencies using:
pip install -r requirements.txt

📄 requirements
tensorflow==2.10.0
opencv-python
numpy
matplotlib

⚙️ How It Works

Webcam captures live video
Face detected using Haar Cascade
Image converted to grayscale
Image resized to 48×48
CNN predicts emotion
Emotion displayed on screen

🧠 CNN Architecture
Convolution Layer (32 filters)
Max Pooling
Convolution Layer (64 filters)
Max Pooling
Convolution Layer (128 filters)
Flatten Layer
Dense Layer
Dropout

Softmax Output Layer

▶️ How to Run the Project
Step 1️⃣ Train the Model
python train_model.py


This will generate:
model/emotion_model.h5

Step 2️⃣ Run Emotion Detection
python main.py


✔ Webcam opens
✔ Face detected
✔ Emotion displayed
❌ Press Q to exit

😄 Emotions Detected

Happy
Sad
Angry
Surprise
Fear
Neutral
Disgust

🚀 Applications
Human Computer Interaction
Mental Health Analysis
Smart Classroom Systems
AI-based Surveillance
Emotion Recognition Systems

⚠️ Limitations
Requires good lighting
Works best with frontal faces
Accuracy depends on dataset
Requires Python 3.10

🚀 Future Enhancements
Face recognition integration
Emotion-based music player
Mobile application
Emotion analytics dashboard
Improved CNN accuracy

👨‍💻 Author

Saksham Nikam
Face Emotion Detection Project

⭐ Support
If you like this project, don’t forget to ⭐ the repository!