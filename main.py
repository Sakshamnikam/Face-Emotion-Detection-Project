import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

CONFIDENCE_THRESHOLD = 0.45

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(40, 40)
    )

    for (x,y,w,h) in faces:

        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        # Convert to grayscale (VERY IMPORTANT)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face = cv2.resize(face, (48,48))
        face = face / 255.0

        # Reshape to match model input
        face = np.reshape(face, (1,48,48,1))


        prediction = model.predict(face, verbose=0)[0]
        confidence = np.max(prediction)
        emotion_index = np.argmax(prediction)

        if confidence > CONFIDENCE_THRESHOLD:
            emotion = emotion_labels[emotion_index]
        else:
            emotion = "Uncertain"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
