import mediapipe as mp

def verify_mediapipe():
    try:
        # Test de l'accès aux solutions
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        print("✅ Mediapipe est bien installé !")
        print(f"Version de Mediapipe : {mp.__version__}")
        
        # Test de l'initialisation de l'objet
        if hands:
            print("✅ L'objet 'hands' a été créé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")

if __name__ == "__main__":
    verify_mediapipe()