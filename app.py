import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# All necessary imports
from torchvision import transforms
from model_loader import load_spiral_model, load_wave_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Page Configuration ---
st.set_page_config(
    page_title="Parkinson's Prediction App",
    page_icon="🧠",
    layout="wide"
)

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
            with res_col1:
                st.metric("Spiral Model Confidence (Parkinson)", f"{prob_spiral:.2%}")
                st.metric("Wave Model Confidence (Parkinson)", f"{prob_wave:.2%}")
            with res_col2:
                st.metric("Combined Confidence (Parkinson)", f"{final_prob:.2%}")
                if final_prob > 0.5:
                    st.error(f"Final Prediction: {final_prediction}")
                else:
                    st.success(f"Final Prediction: {final_prediction}")
            
            st.divider()
            st.header("Explainable AI (XAI) Analysis")

            # NEW: Static explanation paragraph
            st.markdown("""
            The heatmaps below show the areas of the drawings the AI model focused on most. **Red ('hot') areas** were critically important to the model's decision, while **blue ('cold') areas** were largely ignored.

            - For a **'Parkinson's'** prediction, the model often highlights areas of **tremor (shakiness)**, **inconsistent curves**, or **abnormally small/flat shapes (micrographia)**.
            - For a **'Healthy'** prediction, the model tends to focus on **smooth lines**, **consistent shapes**, and **uniform wave heights**.
            """)
            
            xai_col1, xai_col2 = st.columns(2)
            with xai_col1:
                spiral_xai_viz = generate_grad_cam(spiral_image, spiral_model, 'resnet', spiral_idx)
                st.image(spiral_xai_viz, caption="Spiral Drawing Analysis", use_container_width=True)

            with xai_col2:
                wave_xai_viz = generate_grad_cam(wave_image, wave_model, 'convnext', wave_idx)
                st.image(wave_xai_viz, caption="Wave Drawing Analysis", use_container_width=True)
    else:
        st.warning("Please upload both a spiral and a wave image.")