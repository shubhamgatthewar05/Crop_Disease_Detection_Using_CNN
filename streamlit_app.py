import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    img = image.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # If the image has an alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices_rev):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices_rev.get(predicted_class_index, "Unknown")
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

def main():
    st.title("🌾 Plant Disease Detection App")
    st.write("Upload an image of a crop leaf, and the app will predict the disease.")

    # Load the trained model
    model_path = 'plant_disease_prediction_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the model is trained and saved.")
        return

    @st.cache_resource  # Cache the model to prevent reloading on each run
    def load_trained_model():
        return load_model(model_path)

    model = load_trained_model()

    # Load class indices
    class_indices_path = 'class_indices.json'
    if not os.path.exists(class_indices_path):
        st.error(f"Class indices file '{class_indices_path}' not found. Please ensure it exists.")
        return

    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    # Reverse the class_indices dictionary to map indices to class names
    class_indices_rev = {int(k): v for k, v in class_indices.items()}

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Prediction
            predicted_class, confidence = predict_image_class(model, image, class_indices_rev)
            st.write(f"### Prediction: {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()
