# app.py

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_loader import load_spiral_model, load_wave_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Parkinson's Prediction App",
    page_icon="🧠",
    layout="wide"
)

# --- Load Models ---
# Using st.cache_resource ensures the models are loaded only once
@st.cache_resource
def get_models():
    spiral_model = load_spiral_model()
    wave_model = load_wave_model()
    return spiral_model, wave_model

spiral_model, wave_model = get_models()

# --- Prediction Function ---
def predict(image, model):
    """Runs prediction on a single image and returns the Parkinson's probability."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        return probabilities[1].item() # Return prob of 'Parkinson' class

# --- Web App Interface ---
st.title("Parkinson's Disease Prediction via Hand Drawings")
st.write("Upload both a **spiral** and a **wave** drawing to get an integrated prediction.")

col1, col2 = st.columns(2)

with col1:
    st.header("Spiral Drawing")
    spiral_image_file = st.file_uploader("Upload a Spiral Image", type=["png", "jpg", "jpeg"], key="spiral")
    if spiral_image_file:
        st.image(spiral_image_file, caption="Uploaded Spiral")

with col2:
    st.header("Wave Drawing")
    wave_image_file = st.file_uploader("Upload a Wave Image", type=["png", "jpg", "jpeg"], key="wave")
    if wave_image_file:
        st.image(wave_image_file, caption="Uploaded Wave")

if st.button("Analyze Drawings", use_container_width=True):
    if spiral_image_file and wave_image_file:
        with st.spinner('Analyzing... Please wait.'):
            # Open images
            spiral_image = Image.open(spiral_image_file).convert("RGB")
            wave_image = Image.open(wave_image_file).convert("RGB")

            # Get predictions
            prob_spiral = predict(spiral_image, spiral_model)
            prob_wave = predict(wave_image, wave_model)
            final_prob = (prob_spiral + prob_wave) / 2.0

            # Display results
            st.header("Analysis Complete")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Spiral Model Confidence (Parkinson)", f"{prob_spiral:.2%}")
                st.metric("Wave Model Confidence (Parkinson)", f"{prob_wave:.2%}")
            
            with res_col2:
                st.metric("Combined Confidence (Parkinson)", f"{final_prob:.2%}")
                if final_prob > 0.5:
                    st.error("Final Prediction: Parkinson's Detected")
                else:
                    st.success("Final Prediction: Healthy")
    else:
        st.warning("Please upload both a spiral and a wave image.")