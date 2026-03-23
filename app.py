import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import sys

# Project imports
try:
    from xray.logger import logging
    from xray.exception import XRayException
    from xray.ml.model.arch import Net
except ImportError as e:
    st.error(f"Project modules not found: {e}. Run from Lungs-Disease-Diagnosis/.")
    st.stop()

# -----------------------------
# Page Config (UI Settings)
# -----------------------------
st.set_page_config(
    page_title="Lung Disease Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Aesthetic UI
# -----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #0b0f16 0%, #121827 40%, #0a1018 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(180deg, #0b0f16 0%, #121827 40%, #0a1018 100%);
        color: white;
    }
    h1 {
        text-align: center;
        color: #ffffff;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5em;
        border: none;
        text-decoration: none;
        border-bottom: none;
    }
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        height: 3em;
        width: 200px;
        margin: 0 auto;
        font-size: 1.2em;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 2em;
        margin: 1em 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2em;
        margin: 1em 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .predictor-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5em;
        margin: 0.5em 0 1.5em;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .sidebar {
        background: linear-gradient(180deg, #1f3040 0%, #152734 100%);
        color: #f2f7ff;
        padding: 1em;
        border-radius: 10px;
        border: 1px solid rgba(210, 225, 255, 0.3);
    }
    .sidebar-box {
        background: rgba(25, 47, 69, 0.7);
        border-radius: 12px;
        padding: 0.8em;
        margin-bottom: 1em;
        border: 1px solid rgba(158, 188, 223, 0.35);
    }
    hr {
        display: none !important;
        border: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .element-container {
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.header("🩺 Lung Disease Predictor")
    st.markdown("<h5 style='color: #b2dfff; margin-bottom: 0.5em;'>Predictor Overview</h5>", unsafe_allow_html=True)
    st.write("This predictor classifies chest X-rays for COVID-19, Pneumonia, and Normal conditions.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.header("ℹ️ About")
    st.write("This AI-powered application uses deep learning to analyze chest X-ray images and detect lung diseases including COVID-19, Pneumonia, and Normal cases.")
    
    st.header("📊 Model Performance")
    st.write("• **Accuracy**: ~91%")
    st.write("• **Classes**: COVID19, NORMAL, PNEUMONIA")
    st.write("• **Architecture**: Custom CNN")
    
    st.header("🔧 How to Use")
    st.write("1. Upload a chest X-ray image")
    st.write("2. Click 'Analyze Image'")
    st.write("3. View the prediction results")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Main Content
# -----------------------------
st.markdown('<h1>🫁 Lung Disease Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Chest X-ray Analysis for COVID-19, Pneumonia & Normal Detection</p>', unsafe_allow_html=True)

# -----------------------------
# Load Model and Transforms (cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        logging.info("Loading model and transforms...")
        
        # Load trained model from notebook directory
        model_path = os.path.join("notebook", "xray_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run the Experiment.ipynb notebook first.")
        model = Net()  # Initialize the model architecture
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Define transforms as in the notebook
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        logging.info("Artifacts loaded successfully.")
        return model, transform
    except Exception as e:
        raise XRayException(e, sys)

model, transform = load_artifacts()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    try:
        logging.info("Starting prediction...")
        
        # Preprocess image using pipeline transforms
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        label_str = ["COVID19", "NORMAL", "PNEUMONIA"][predicted.item()]
        logging.info(f"Prediction: {label_str}, Confidence: {confidence.item():.4f}")
        
        return label_str, confidence.item()
    except Exception as e:
        raise XRayException(e, sys)

# -----------------------------
# Upload Image
# -----------------------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("🖼️ Preview")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)
    else:
        st.subheader("🖼️ Preview")
        st.info("Upload an image to see preview")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="display: flex; justify-content: center; align-items: flex-end; height: 110px;">', unsafe_allow_html=True)
        if st.button("🔍 Analyze Image", key="predict"):
            try:
                with st.spinner("🧠 AI is analyzing the image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    label, confidence = predict(image)
                
                st.success("✅ Analysis Complete!")
                
                # Display results with better formatting
                if label == "NORMAL":
                    st.markdown(f"<h2 style='color: #28a745;'>🟢 {label}</h2>", unsafe_allow_html=True)
                    st.write(f"**Confidence**: {confidence*100:.2f}%")
                    st.write("The X-ray appears normal with no signs of lung disease.")
                elif label == "COVID19":
                    st.markdown(f"<h2 style='color: #ffc107;'>🟡 {label}</h2>", unsafe_allow_html=True)
                    st.write(f"**Confidence**: {confidence*100:.2f}%")
                    st.write("⚠️ COVID-19 indicators detected. Please consult a healthcare professional.")
                else:  # PNEUMONIA
                    st.markdown(f"<h2 style='color: #dc3545;'>🔴 {label}</h2>", unsafe_allow_html=True)
                    st.write(f"**Confidence**: {confidence*100:.2f}%")
                    st.write("⚠️ Pneumonia indicators detected. Please seek medical attention.")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

