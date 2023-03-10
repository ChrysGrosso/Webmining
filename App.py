import streamlit as st
from PIL import Image


image = Image.open('Image.png')

st.title("Application de reconnaissance faciale et vocale")

st.write("Cette application a été réalisée par : Laura Boutonnet, Nawres Dhiflaoui, Chrystelle Grosso et Robin Ponson")

st.write("Elle a été réalisée dans le cadre d'un de WebMining, nous disposions de 47 heures.")

st.write("")

st.write("Son but est de permettre la détection de visage présents dans une vidéo (webcam). Nous devions identifier les visages présents sur la webcam mais aussi leurs donner une tranche d'âge, un genre et une émotion.")

st.write("")

st.write("L'application se présente en deux onglets : Accueil et Reconnaissance faciale + vocale")

st.image(image, caption='Exemple de notre application')
