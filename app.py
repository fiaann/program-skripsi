import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd 

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/model.h5')

model = load_model()

# Preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))  # Sesuaikan dengan ukuran input model
    image = np.array(image) / 255.0   # Normalisasi (jika model dilatih dengan ini)
    image = np.expand_dims(image, axis=0)
    return image

# UI Streamlit
st.title("Klasifikasi Tingkat Kematangan Buah Pisang")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)


    st.write("Memprediksi...")
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    classes = ['freshbanana', 'rawbanana', 'rottenbanana']
    
    # Buat DataFrame dengan label huruf
    prediction_df = pd.DataFrame(prediction, columns=classes)
    st.write("Hasil Prediksi:")
    st.dataframe(prediction_df)

    predicted_class = classes[np.argmax(prediction)]
    st.write(f"Model memprediksi: {predicted_class}")