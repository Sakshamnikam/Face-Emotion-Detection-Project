# 🎭 Face Emotion Detection System

Face Emotion Detection is a real-time AI-based computer vision project that detects human emotions using a webcam. It leverages **Deep Learning (CNN)** and **OpenCV** to analyze facial expressions and classify emotions accurately.

---
## 🛠️ Tech Stack

- **Programming Language:** Python 3.10
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **GUI Framework:** CustomTkinter
- **Image Processing:** NumPy, PIL
- **Model Type:** Convolutional Neural Network (CNN)
- **Dataset:** FER-2013
- **Packaging:** PyInstaller (Windows EXE)


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

## 📁 Project Structure

```text
Face_Emotion_Detection/
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
├── train_model.py        # Train CNN model
├── gui.py                # GUI-based emotion detector
├── main.py               # CLI-based emotion detector
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
opencv-python==4.8.0.76
numpy==1.23.5
matplotlib==3.7.2
Pillow==9.5.0
customtkinter==5.2.1
```

---

## 📥 Dataset Download

This project uses the **FER-2013** dataset.

### Steps to Download and Setup Dataset

1. Download the dataset from Kaggle:

   👉 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

2. Extract the downloaded ZIP file.

3. Copy the extracted **fer2013** folder into the `dataset/` directory of this project.

After copying, the structure should look like:

```text
dataset/
├── train/
└── test/
```

⚠️ **Note:** Make sure the folder names and structure remain unchanged, otherwise the training script may not work correctly.

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
python gui.py
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
## NOTE:• First run may be slow due to model loading
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
