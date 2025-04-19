import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os

# Model path and download link
MODEL_FILE = "model.h5"
MODEL_LINK = "https://github.com/soumik12345/Traffic-Sign-Classifier/releases/download/v1.0/model.h5"

# Function to download model if not present
def fetch_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading the model...")
        response = requests.get(MODEL_LINK)
        with open(MODEL_FILE, "wb") as file:
            file.write(response.content)

# Load model with caching
@st.cache_resource
def initialize_model():
    fetch_model()
    return load_model(MODEL_FILE)

# Class labels for predictions
CLASS_NAMES = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
    "Speed Limit 80", "End of Speed Limit 80", "Speed Limit 100", "Speed Limit 120",
    "No Passing", "No Passing for Vehicles over 3.5 tons", "Right-of-way at Intersection",
    "Priority Road", "Yield", "Stop", "No Vehicles", "Vehicles Over 3.5 Tons Prohibited",
    "No Entry", "General Caution", "Dangerous Curve Left", "Dangerous Curve Right",
    "Double Curve", "Bumpy Road", "Slippery Road", "Road Narrows on the Right",
    "Road Work", "Traffic Signals", "Pedestrians", "Children Crossing", "Bicycles Crossing",
    "Beware of Ice/Snow", "Wild Animals Crossing", "End of All Restrictions", "Turn Right Ahead",
    "Turn Left Ahead", "Ahead Only", "Go Straight or Right", "Go Straight or Left",
    "Keep Right", "Keep Left", "Roundabout Mandatory", "End of No Passing",
    "End of No Passing by Vehicles Over 3.5 Tons"
]

# Function to prepare image for prediction
def process_image(img: Image.Image) -> np.ndarray:
    img = img.resize((32, 32))
    array = np.asarray(img) / 255.0
    return np.expand_dims(array, axis=0)

# Streamlit app interface
st.set_page_config("Traffic Sign Recognition", page_icon="🚗")
st.title("🚗 Traffic Sign Classifier")
st.caption("Developed by Chandra")

# Image uploader
image_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

# Load model
model = initialize_model()

# Prediction trigger
if image_file:
    uploaded_image = Image.open(image_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    input_data = process_image(uploaded_image)
    prediction = model.predict(input_data)
    top_class = np.argmax(prediction)
    probability = float(np.max(prediction))

    st.markdown("### 🧠 Prediction Result")
    st.success(f"**Traffic Sign:** {CLASS_NAMES[top_class]}")
    st.info(f"**Confidence:** {probability:.2%}")
