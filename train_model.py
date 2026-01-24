import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size
IMG_SIZE = 48
BATCH_SIZE = 64

train_path = "dataset/train"
test_path = "dataset/test"

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

test_data = datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

# Save model
model.save("model/emotion_model.h5")

print("✅ Model training completed & saved successfully!")
