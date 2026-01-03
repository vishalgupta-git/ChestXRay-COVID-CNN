import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)

CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']
IMAGE_SIZE = (128, 128)

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_model.keras")

with st.spinner("Loading model..."):
    model = load_model()

# ------------------ FUNCTIONS ------------------
def preprocess_image(image):
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image = np.asarray(image) / 255.0
    return np.expand_dims(image, axis=0)

# ------------------ UI ------------------
st.title("🫁 COVID-19 Chest X-Ray Detector")

st.info(
    "This application classifies chest X-ray images into:\n"
    "- Covid-19\n"
    "- Normal\n"
    "- Viral Pneumonia\n\n"
    "⚠️ For educational purposes only."
)

file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded X-Ray", width=350)

    if st.button("Analyze"):
        img = preprocess_image(image)
        prediction = model.predict(img)

        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100
        label = CLASS_NAMES[class_index]

        st.subheader("Prediction Result")

        if label == "Covid":
            st.error(f"🚨 COVID-19 ({confidence:.2f}%)")
        elif label == "Viral Pneumonia":
            st.warning(f"⚠️ Viral Pneumonia ({confidence:.2f}%)")
        else:
            st.success(f"✅ Normal ({confidence:.2f}%)")

        st.markdown("### Class Probabilities")
        st.bar_chart(dict(zip(CLASS_NAMES, prediction[0])))
