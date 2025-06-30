import streamlit as st
from face_detection import detect_faces

def app():
    st.title("🧠 Face Detection App - Viola-Jones")

    st.markdown("""
    ## 👋 Instructions
    1. Cliquez sur **"Detect Faces"** pour lancer votre webcam.
    2. Appuyez sur `q` pour quitter la détection.
    3. Appuyez sur `s` pendant la détection pour **sauvegarder une image** avec les visages détectés.
    """)

    # Color picker
    color = st.color_picker("🎨 Choisissez la couleur des rectangles", "#00FF00")  # vert par défaut

    # Sliders for scaleFactor and minNeighbors
    scale_factor = st.slider("🔍 Ajuster le paramètre `scaleFactor`", 1.1, 2.0, 1.3, 0.1)
    min_neighbors = st.slider("👥 Ajuster le paramètre `minNeighbors`", 3, 10, 5)

    if st.button("📷 Detect Faces"):
        st.warning("La détection a été lancée. Appuyez sur 'q' pour quitter. Appuyez sur 's' pour sauvegarder une image.")
        detect_faces(scale_factor, min_neighbors, color)

if __name__ == "__main__":
    app()
