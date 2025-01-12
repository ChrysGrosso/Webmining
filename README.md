# Reconnaissance faciale et vocale d'identification sur des vidéos WEBCAM - Master SISE // Challenge 

* BOUTONNET Laura
* DHIFLAOUI Nawres
* GROSSO Chrystelle
* PONSON Robin

# Installation de Cmake

Avant de pouvoir tester notre application il vous faut installer cmake, prérequis pour la librairie dlib, utilisée dans notre application. 

Pour celà, cliquer sur ce lien. 

https://cmake.org/download/

Ensuite, cliquer sur installer la version Windows *64 comme ci-dessous. 

![image](https://user-images.githubusercontent.com/83652394/224253280-d193e404-1adc-45c4-8866-cc1f5aeff606.png)

# Utilisation de l'application

1. Télécharger le dossier.zip et extrayez le dans le répertoire de votre choix. 
2. Installer l'environnement conda "fac_voc_reco.yml" et l'activer.

Via une console anaconda prompt :
```
cd "chemin/ou/se/situe/votre/fichier/fac_voc_reco.yml

conda env create -f fac_voc_reco.yml 

conda activate fac_voc_reco
```

3. Executer  "App.py" en ligne de commande
```
streamlit run App.py
```

# Aperçu de l'application

## Page d'accueil 

![image](https://user-images.githubusercontent.com/83652394/224171278-fab90270-ca39-4225-b8cb-7d5b209b8d04.png)

Vous pouvez voir un court descriptif de l'application ainsi qu'une photo de demonstration.

## Reconnaisance faciale + vocale 

Vous touverez ci-dessous une image de notre page de reconnaissance faciale + vocale. 

![image](https://user-images.githubusercontent.com/83652394/224178169-042e8ce7-033e-4693-8550-38d7c4773ccf.png) 

Arriver sur cette page, vous pouvez dire à voix haute : "Allume la caméra". 

Le message : Vous avez cliqué sur Allumer la caméra s'affiche

Au bout d'un petit moment, la caméra va s'ouvrir et vous pourrez apprécier les différentes fonctionnalités de reconnaissance faciale.

Pour arrêter la webcam, cliquer succcesivement deux fois sur eteindre la caméra.

Si vous voulez relancer la caméra, cliquer sur "Cliquez ici si vous voulez recommencer" et dîtes, "allume la caméra" (vous pouvez auusi cliquer sur ce bouton pour l'allumer, un message va alors s'afficher pour vous notifier que la caméra va s'allumer et vous dire par quel moyen la caméra s'est lancée).

Quand vous voudrez arrêter la webcam, cliquer sur "Eteindre la caméra", si vous voulez quitter l'application et voir l'enregistrement que vous avez généré, vous pouvez fermer la page et le prompt d'Anaconda. 

Vous pourrez trouver la vidéo dans votre réperoire courant sous le nom "vidéo.mp4" (voir ci-dessous).

![image](https://user-images.githubusercontent.com/83652394/224183211-f66a2155-e86b-4d67-8943-a9aeba3d838b.png)

# Actualisation de la base de données d'entraînement 

Si vous souhaitez ajouter un ou plusieurs individus dans les known_faces (veillez à ce que la photo soi au format ".jpeg"), mettez la photo dans ce répertoire. 

Ensuite, via une console comme le terminal de Visual Studio Code ou un Anaconda Prompt, éxécutez la commande suivante : 

```
python encoding_train.py
```
Ce fichier actualisera les données présentes dans le fichier "encoded_face". Celui-ci qui contient les caractéristiques des visages de chacun des individu présent dans la base de données.

Bonne découverte !


