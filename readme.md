# 🎭 Face Emotion Detection System

Face Emotion Detection is a real-time AI-based computer vision project that detects human emotions using a webcam. It leverages **Deep Learning (CNN)** and **OpenCV** to analyze facial expressions and classify emotions accurately.

---

## ✨ Features

* 🎥 **Real-Time Emotion Detection**

  * Detects faces using a webcam
  * Predicts emotions in real time
  * Displays emotion labels on the screen

* 🧠 **Deep Learning Model**

  * CNN-based architecture
  * Trained on the **FER-2013** dataset
  * High-accuracy emotion classification

* 📸 **Face Detection**

  * Uses Haar Cascade Classifier
  * Detects frontal faces
  * Works smoothly in real time

---

## 🗂️ Project Structure

```text
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
```

---

## ⚙️ Requirements

### 🧩 System Requirements

* Windows OS
* Python 3.10
* Webcam

### 📦 Python Libraries

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```text
tensorflow==2.10.0
opencv-python
numpy
matplotlib
```

---

## ⚙️ How It Works

1. Webcam captures live video
2. Face detected using Haar Cascade
3. Image converted to grayscale
4. Image resized to **48 × 48**
5. CNN predicts emotion
6. Emotion label displayed on screen

---

## 🧠 CNN Architecture

* Convolution Layer (32 filters)
* Max Pooling Layer
* Convolution Layer (64 filters)
* Max Pooling Layer
* Convolution Layer (128 filters)
* Flatten Layer
* Dense Layer
* Dropout Layer
* Softmax Output Layer

---

## ▶️ How to Run the Project

### Step 1️⃣ Train the Model

```bash
python train_model.py
```

This will generate:

```
model/emotion_model.h5
```

### Step 2️⃣ Run Emotion Detection

```bash
python main.py
```

✔ Webcam opens
✔ Face detected
✔ Emotion displayed
❌ Press **Q** to exit

---

## 😄 Emotions Detected

* Happy
* Sad
* Angry
* Surprise
* Fear
* Neutral
* Disgust

---

## 🚀 Applications

* Human Computer Interaction
* Mental Health Analysis
* Smart Classroom Systems
* AI-based Surveillance
* Emotion Recognition Systems

---

## ⚠️ Limitations

* Requires good lighting conditions
* Works best with frontal faces
* Accuracy depends on the dataset quality
* Requires Python 3.10

---

## 🚀 Future Enhancements

* Face recognition integration
* Emotion-based music player
* Mobile application
* Emotion analytics dashboard
* Improved CNN accuracy

---

## 👨‍💻 Author

**Saksham Nikam**
Face Emotion Detection Project

---

## ⭐ Support

If you like this project, don’t forget to ⭐ the repository!
