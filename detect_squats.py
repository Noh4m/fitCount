import cv2
import numpy as np
import pygame

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Vérifier si la capture vidéo est réussie
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

squats_count = 0
series_count = 0

# Initialisation de Pygame pour jouer le son
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav") 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ajustez ce seuil en fonction de la position de la tête
threshold_down = 300 

# Suivre la direction de mouvement de la tête
moving_up = False

# Définir la largeur et la hauteur souhaitées
desired_width = 640  # Remplacez par la largeur souhaitée
desired_height = 480  # Remplacez par la hauteur souhaitée

# Définir la largeur et la hauteur de la fenêtre
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True:
    # Capture d'une trame vidéo
    ret, frame = cap.read()

    # Vérifier si la capture de la trame est réussie
    if not ret:
        print("Erreur: Impossible de lire la trame.")
        break

    # Détecte les visages
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Vérifie si la tête est suffisamment haute pour compter le squat
        if y < threshold_down:
            # Vérifier si le visage est droit 
            if h > 0.7 * w:
                if not moving_up:
                    squats_count += 1

                    # Série de 10 squats est accomplie
                    if squats_count % 10 == 0:
                        series_count += 1
                        print(f"Série accomplie : {series_count}")

                        # Alerte sonore
                        alert_sound.play()

                    # Nombre de squats effectués
                    print(f"Squats comptés : {squats_count}")

                moving_up = True
            else:
                moving_up = False
        else:
            moving_up = False

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Affiche le score des squats et des séries sur l'écran
    cv2.putText(frame, f"Squats: {squats_count} | Series: {series_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Affiche l'image avec les rectangles autour des visages
    cv2.imshow('Face Detection', frame)

    # appuyez sur la touche 'q' pour quitter l'application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
