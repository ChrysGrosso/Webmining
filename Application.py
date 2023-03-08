# Importation des librairie et d'un module model.py
import cv2
import argparse
import pickle
import face_recognition
import numpy as np
import streamlit as st
from model import ERModel
import pandas as pd

st.title('Reconnaissance faciale et identification sur des vidéos WEBCAM')

# Récupération de la fonction pour récupérer les coordonnées des visages

exec(open("recup_donnees_visage.py").read())

FRAME_WINDOW = st.image([])

start = st.button("Allumer la caméra")
stop = st.button("Eteindre la caméra")

# Initialisation de la reconnaissance vocale

if start : 
    ##########"Allumer la caméra##########################################
    # Chargement des modèles pré-entrainés
    model = ERModel("model.json", "model_weights.h5")
    data = pickle.loads(open('face_enc','rb').read())
    
    faces_encodings = data["encodings"]
    faces_names = data["names"]
    
    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    
    args = parser.parse_args()
    
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Homme', 'Femme']
    
    ageNet = cv2.dnn.readNet(ageProto,ageModel)
    genderNet = cv2.dnn.readNet(genderProto,genderModel)
    faceNet = cv2.dnn.readNet(faceProto,faceModel)
    
    if args.device == "cpu":
        ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
        
    elif args.device == "gpu":
        ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
        genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
        genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(args.input if args.input else 0)
    padding = 20
        
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
  
    out = cv2.VideoWriter('C:/Users/laura/.env/webmining/video_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_width,frame_height))
    face_names = []
    
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frame = cv2.flip(frame,1)
        
        if not hasFrame:
            cv2.waitKey()
            break
        
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameFace, bboxes = getFaceBox(faceNet, frame)
        
        faces = face_recognition.face_locations(gray)
        
        
        for bbox in faces:
            face=frame[max(bbox[0]-padding,0):min(bbox[2]+padding,frame.shape[0]),max(0,bbox[3]-padding):min(bbox[1]+padding,frame.shape[1])]

            ########################################################
            # Détection des visages 
            
            encodings = face_recognition.face_encodings(rgb,[bbox])
            matches = face_recognition.compare_faces (data["encodings"], encodings[0],tolerance=0.6)
            
            name = "Inconnu"
            face_distances = face_recognition.face_distance(data["encodings"], encodings[0])
            
            best_match_index = np.argmin(face_distances)
            
    
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            
            if name not in face_names : 
                face_names.append(name)
                print(face_names)
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            ####################################################
            # Prédiction genre
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ####################################################
            # Prédiction age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            ####################################################
            # Prédiction sentiments
            roi_gray = gray[max(bbox[0]-padding,0):min(bbox[2]+padding,frame.shape[0]),max(0,bbox[3]-padding):min(bbox[1]+padding,frame.shape[1])]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            pred = model.predict_emotion(cropped_img)
    
    
            label = "{},{},{},{}".format(gender, age,name,pred)
            ## Création du carré
            cv2.rectangle(frameFace, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), int(round(frameFace.shape[0]/150)), 8)
            
            font = cv2.FONT_HERSHEY_DUPLEX

            k = 0
            
            for i, line in enumerate(label.split(',')):
                k = k+25
                
                try:
                    ## Sentiment et couelur associé
                    if line in ["Enerve", "Dégoute", "Peureux", "Heureux", "Neutre", "Triste","Surpris"]:
                        cv2.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 255, 255), 1)
                    elif line.startswith("("):
                        cv2.putText(frameFace, line+" "+str(round(max(agePreds[0].tolist())*100,2)), (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (0, 0, 0), 1)
                    elif line == "Male":
                        cv2.putText(frameFace, line+" "+str(round(max(genderPreds[genderPreds[0].argmax()].tolist())*100,2)), (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 255, 255), 1)
                    elif line == "Female":
                        cv2.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 255, 255), 1)
                    else:
                        cv2.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (0, 0, 0), 1)
                except:
                    pass
                
        frame_ok = cv2.cvtColor(frameFace, cv2.COLOR_RGB2BGR)
    
        # Affichage de l'image sur l'application
        FRAME_WINDOW.image(frame_ok)
        
        out.write(cv2.cvtColor(frame_ok, cv2.COLOR_RGB2BGR))
        
        with open('noms_output.txt', 'w') as f:
            f.writelines('\n'.join(face_names))           

        
        
        if stop :
            
                
            break 
            cap.release()
            out.release()
            

cv2.destroyAllWindows()
