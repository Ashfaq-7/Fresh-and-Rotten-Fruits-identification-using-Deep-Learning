import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = r"C:\Users\ashfa\Desktop\fruits\dataset\fruit_classification_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Streamlit UI
st.title("🍏🍌🍊 Fresh vs Rotten Fruit Classifier")
st.write("Upload an image of an apple, banana, or orange to check if it's fresh or rotten!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image_disp = Image.open(uploaded_file)
    st.image(image_disp, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image_disp.resize((224, 224))  # Ensure it matches model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    
    # Display result
    st.success(f"🔍 Prediction: **{predicted_class}**")

