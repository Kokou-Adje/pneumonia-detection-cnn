"""
Streamlit web app for the chest X-ray pneumonia classifier.
Author: Kokou Adje

Upload a chest X-ray image and the trained DenseNet-121 model predicts
Normal or Pneumonia with a confidence score. Run with:  streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
from PIL import Image

# Training used flow_from_dataframe with class_mode="binary", which assigns
# labels alphabetically: NORMAL=0, PNEUMONIA=1. The sigmoid output is therefore
# the probability of pneumonia.
CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = (224, 224)

# The model is saved next to the training script in src/, or at the repo root.
_HERE = os.path.dirname(__file__)
_CANDIDATES = [
    os.path.join(_HERE, "src", "pneumonia_cnn_model.keras"),
    os.path.join(_HERE, "pneumonia_cnn_model.keras"),
]
MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained Keras model once and cache it across reruns."""
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the training pipeline exactly: RGB, 224x224, DenseNet preprocess_input.

    DenseNet121 was trained on 3-channel RGB images run through
    keras.applications.densenet.preprocess_input (not a plain /255 rescale),
    so the app must apply the same transform or predictions will be wrong.
    """
    from tensorflow.keras.applications.densenet import preprocess_input

    image = image.convert("RGB").resize(IMG_SIZE)          # 3 channels, 224x224
    array = np.asarray(image, dtype="float32")             # 0..255 floats
    array = preprocess_input(array)                        # DenseNet normalization
    return np.expand_dims(array, axis=0)                   # add batch dimension


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray and a DenseNet-121 model (transfer learning) predicts "
    "whether it shows signs of pneumonia. Trained on the Chest X-Ray Images dataset."
)

st.warning(
    "This is a student project for learning purposes, not a medical device. "
    "It must not be used for real diagnosis.",
    icon="⚠️",
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Train it in the Kaggle/Colab notebook, download "
        "`pneumonia_cnn_model.keras`, and place it in this app's `src/` folder, "
        "then reload this page."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            prob_pneumonia = float(model.predict(preprocess(image), verbose=0)[0][0])
        label_idx = int(prob_pneumonia > 0.5)
        label = CLASS_NAMES[label_idx]
        confidence = prob_pneumonia if label_idx == 1 else 1 - prob_pneumonia

        if label == "Pneumonia":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.caption(f"Model output (probability of pneumonia): {prob_pneumonia:.3f}")
        st.progress(prob_pneumonia)

    st.info(
        "The model reaches about 88% accuracy and 0.95 ROC-AUC on the test set, "
        "with balanced sensitivity and specificity after class-weighting. Still, "
        "treat any result as a flag to review, not a diagnosis."
    )
else:
    st.info("Upload an image above to get a prediction.")
