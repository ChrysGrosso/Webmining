import webbrowser 
import speech_recognition as sr
import cv2

# Initialisation de la reconnaissance vocale
r = sr.Recognizer()

# Initialisation de la webcam
cap = cv2.VideoCapture(0)
while True : 
    # Définition du codec et de la création du fichier vidéo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    # Boucle d'écoute des commandes vocales
    while True:
        # Enregistrement audio
        with sr.Microphone() as source:
            print("Parlez...")
            audio = r.listen(source)

        # Reconnaissance de la parole
        try:
            command = r.recognize_google(audio, language="fr-FR")
            print("Commande : " + command)

            # Commande pour démarrer la webcam
            if "open" in command:
                cap.open(0)
                print("Webcam démarrée.")

            # Commande pour démarrer l'enregistrement
            elif "start" in command:
                out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
                print("Enregistrement démarré.")

            # Commande pour arrêter l'enregistrement
            elif "end" in command:
                out.release()
                out = None
                print("Enregistrement arrêté.")

            # Commande pour arrêter la webcam
            elif "close" in command:
                cap.release()
                print("Webcam arrêtée.")
                
            # Commande pour quitter l'application
            elif "quitter" in command:
                break

        # En cas d'erreur de reconnaissance vocale
        except sr.UnknownValueError:
            print("Commande vocale non reconnue.")
            
        # Lecture de la webcam et enregistrement vidéo
        ret, frame = cap.read()
        if out is not None:
            out.write(frame)
        cv2.imshow('Webcam', frame)
        
        # Sortie de la boucle d'écoute si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fermeture de la fenêtre et relâchement de la webcam
    cv2.destroyAllWindows()
    cap.release()



