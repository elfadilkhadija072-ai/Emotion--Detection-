import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger modèle
model = load_model("fer2013_emotion_model.h5")

# Labels émotions
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# Charger image
image_path = "test.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize
image = cv2.resize(image, (48, 48))

# Normalisation
image = image.astype("float32") / 255.0

# Reshape
image = np.reshape(image, (1, 48, 48, 1))

# Prediction
prediction = model.predict(image)
emotion = emotion_labels[np.argmax(prediction)]
print("Predicted Emotion :", emotion)