# app.py (Final Version with Gemini XAI Explanation)

import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

# MODIFIED: Add all necessary imports
from torchvision import transforms
from model_loader import load_spiral_model, load_wave_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Parkinson's Prediction App",
    page_icon="🧠",
    layout="wide"
)

# --- API Key ---
# IMPORTANT: Add your Gemini API Key in the Streamlit Secrets Manager
# Go to Manage app -> Settings -> Secrets and add GEMINI_API_KEY = "YOUR_KEY"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")


# --- Load Models ---
@st.cache_resource
def get_models():
    spiral_model = load_spiral_model()
    wave_model = load_wave_model()
    return spiral_model, wave_model

spiral_model, wave_model = get_models()
class_names = ['Healthy', 'Parkinson']


# --- Prediction and XAI Functions ---
def predict(image, model):
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
        pred_idx = torch.argmax(probabilities).item()
        return pred_idx, probabilities[1].item()

def generate_grad_cam(image_pil, model, model_type, pred_idx):
    transform_for_model = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform_for_model(image_pil).unsqueeze(0)
    open_cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    open_cv_image = cv2.resize(open_cv_image, (224, 224))
    img_for_display = np.float32(open_cv_image) / 255
    if model_type == 'resnet':
        target_layer = model.layer4[-1]
    else: # convnext
        target_layer = model.model.stages[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    return show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)

# NEW: Gemini helper function
def get_gemini_explanation(api_key, image_path, final_prediction):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(image_path)
        prompt = f"""
        You are an AI assistant analyzing the output of a Parkinson's detection model.
        The provided image contains two Grad-CAM heatmaps: one for a spiral drawing and one for a wave drawing.
        The model's final combined prediction was '{final_prediction}'.
        Your task is to:
        1. Briefly analyze the heatmap for the spiral drawing on the left.
        2. Briefly analyze the heatmap for the wave drawing on the right.
        3. Based on the heatmaps, provide a short, combined summary explaining why the model likely arrived at its final prediction.
        """
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Could not get Gemini explanation. Error: {e}"

# --- Web App Interface ---
st.title("Parkinson's Disease Prediction via Hand Drawings")
st.write("Upload a **spiral** and a **wave** drawing for an integrated prediction and an AI-powered explanation.")

with st.sidebar:
    st.title("Upload Drawings")
    spiral_image_file = st.file_uploader("Upload a Spiral Image", type=["png", "jpg", "jpeg"], key="spiral")
    wave_image_file = st.file_uploader("Upload a Wave Image", type=["png", "jpg", "jpeg"], key="wave")

if st.button("Analyze Drawings", use_container_width=True):
    if spiral_image_file and wave_image_file:
        with st.spinner('Analyzing... Please wait.'):
            spiral_image = Image.open(spiral_image_file).convert("RGB")
            wave_image = Image.open(wave_image_file).convert("RGB")

            spiral_idx, prob_spiral = predict(spiral_image, spiral_model)
            wave_idx, prob_wave = predict(wave_image, wave_model)
            final_prob = (prob_spiral + prob_wave) / 2.0
            final_prediction = "Parkinson's Detected" if final_prob > 0.5 else "Healthy"

            st.header("Analysis Complete")
            res_col1, res_col2 = st.columns(2)
            # ... (Result metrics display) ...
            
            # MODIFIED: XAI and Gemini Section
            st.divider()
            st.header("Explainable AI (XAI) Analysis")
            st.write("The heatmaps show what the AI focused on. The text below is an AI-generated summary of the visuals.")
            
            # Generate visualizations
            spiral_xai_viz = generate_grad_cam(spiral_image, spiral_model, 'resnet', spiral_idx)
            wave_xai_viz = generate_grad_cam(wave_image, wave_model, 'convnext', wave_idx)

            # Create a combined plot to send to Gemini
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(spiral_xai_viz)
            axs[0].set_title("Spiral Drawing Analysis")
            axs[0].axis('off')
            axs[1].imshow(wave_xai_viz)
            axs[1].set_title("Wave Drawing Analysis")
            axs[1].axis('off')
            
            # Display the combined plot in Streamlit
            st.pyplot(fig)

            # Save the figure and get Gemini explanation
            combined_xai_path = "combined_xai.png"
            fig.savefig(combined_xai_path)
            
            if GEMINI_API_KEY != "YOUR_API_KEY_HERE":
                st.subheader("AI-Generated Summary")
                explanation = get_gemini_explanation(GEMINI_API_KEY, combined_xai_path, final_prediction)
                st.markdown(explanation)
            else:
                st.warning("Please add your Gemini API Key in the Streamlit app secrets to enable the AI summary.")
    else:
        st.warning("Please upload both a spiral and a wave image.")