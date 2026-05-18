import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter
import mediapipe as mp
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Charger le modèle
model = load_model("fer2013_emotion_model.h5")

# Labels émotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Historique des émotions
emotion_history = []

# Détecteur de visage Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Mains
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
# Initialisation de l'objet Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter et afficher les graphiques")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des mains
    result_hands = hands.process(rgb_frame)
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Détection Face Mesh
    result_face = face_mesh.process(rgb_frame)
    if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Détection émotion
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = prediction[0][emotion_index] * 100

        emotion_history.append(emotion)

        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion + Face Mesh + Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Graphiques
if emotion_history:
    emotion_counts = Counter(emotion_history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    ax1.bar(emotions, counts, color='steelblue')
    ax1.set_title('Émotions détectées')
    ax1.set_xlabel('Émotion')
    ax1.set_ylabel('Nombre de détections')
    ax1.tick_params(axis='x', rotation=45)

    ax2.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Distribution des émotions')

    plt.tight_layout()
    plt.savefig("emotion_stats.png")
    plt.show()
    print("Graphiques sauvegardés !")
else:
    print("Aucune émotion détectée.")