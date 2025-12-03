import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Venus Volcano Detector", layout="centered")

# --- Load Model ---
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_trained_model():
    return tf.keras.models.load_model('venus_volcano_model.keras')

model = load_trained_model()

# --- Manual Feature Extractor (Must match training exactly) ---
def process_image(pil_image):
    # 1. Convert to grayscale and resize to training size (64x64)
    img = pil_image.convert('L') 
    img = img.resize((64, 64))
    img_array = np.array(img)
    
    # 2. Normalize
    norm_img = img_array / 255.0
    
    # 3. Extract Features (COPY FROM TRAINING CODE)
    flat = norm_img.flatten()
    mean_val = np.mean(norm_img)
    std_val = np.std(norm_img)
    max_val = np.max(norm_img)
    
    img_uint8 = (norm_img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, 100, 200)
    edge_density = np.sum(edges) / 255.0
    
    features = np.concatenate([flat, [mean_val, std_val, max_val, edge_density]])
    
    # 4. Reshape for model (1, num_features)
    return features.reshape(1, -1), img_array

# --- UI Layout ---
st.title("🌋 Venus Volcano Detector")
st.write("Upload a SAR image patch to detect if it contains a volcano.")

uploaded_file = st.file_uploader("Choose a .png, .jpg or .sdt file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    if st.button("Analyze Image"):
        # Process and Predict
        features, processed_img = process_image(image)
        
        # Display the processed view (what the computer sees)
        st.write("Processed Input (64x64 Grayscale):")
        st.image(processed_img, width=128, clamp=True)
        
        prediction = model.predict(features)
        score = prediction[0][0]
        
        st.divider()
        
        # Results
        if score > 0.5:
            st.success(f"**Volcano Detected!** (Confidence: {score:.2%})")
            st.balloons()
        else:
            st.error(f"**No Volcano Detected.** (Confidence: {1-score:.2%})")

# Go to anaconda prompt, navigate to this folder and type:
# streamlit run app.py 
