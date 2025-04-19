import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os
import h5py

MODEL_FILE = "model.h5"
MODEL_URL = "https://github.com/soumik12345/Traffic-Sign-Classifier/releases/download/v1.0/model.h5"

# Function to check if the downloaded file is a valid HDF5
def is_valid_h5(filepath):
    try:
        with h5py.File(filepath, "r"):
            return True
    except Exception:
        return False

def download_model():
    if not os.path.exists(MODEL_FILE) or not is_valid_h5(MODEL_FILE):
        st.info("Downloading a valid model file...")
        response = requests.get(MODEL_URL, stream=True)
        if "html" in response.headers.get("Content-Type", ""):
            st.error("Failed to download model: Received HTML page instead of binary file.")
            return
        with open(MODEL_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

@st.cache_resource
def initialize_model():
    download_model()
    if not is_valid_h5(MODEL_FILE):
        st.error("Model file is not valid. Please check the model URL or content.")
        st.stop()
    return load_model(MODEL_FILE)

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

def process_image(img: Image.Image) -> np.ndarray:
    img = img.resize((32, 32))
    array = np.asarray(img) / 255.0
    return np.expand_dims(array, axis=0)

st.set_page_config("Traffic Sign Recognition", page_icon="🚗")
st.title("🚗 Traffic Sign Classifier")
st.caption("Developed by Chandra")

image_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

model = initialize_model()

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
