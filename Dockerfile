# Utilisez une image de base avec Python installé
FROM python:3.8.16

# Ajouter les dépendances système pour la webcam et le micro
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Créez le répertoire de travail
WORKDIR /app

# Copiez les fichiers de l'application vers le conteneur
COPY . /app

# Installez les dépendances python de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Ouvrez le port nécessaire pour que Streamlit fonctionne correctement
EXPOSE 8501

# Exécutez Streamlit
CMD ["streamlit", "run", "test.py"]