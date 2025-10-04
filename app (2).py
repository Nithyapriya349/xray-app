import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Load model only once and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('xray_model.hdf5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """
         )

# Define your class names (adjust to your model's output)
#class_names = ["Normal", "Tumor"]

# File uploader
file = st.file_uploader("Please upload a chest scan file", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # updated
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]  # add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    st.write("Prediction Scores:", score.numpy())
    st.write(
        f"This image most likely belongs to **{class_names[np.argmax(score)]}** "
        f"with a **{100 * np.max(score):.2f}%** confidence."
    )
