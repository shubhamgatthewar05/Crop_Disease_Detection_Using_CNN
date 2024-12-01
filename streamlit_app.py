import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import google.generativeai as genai
from translate import Translator

# Configure the Gemini AI model
genai.configure(api_key="AIzaSyBWlYpnB7qyHX74pJS4mnlmRyhu_cMYqis")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the Google Translator
def get_translator(target_language):
    return Translator(to_lang=target_language)

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
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices_rev):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices_rev.get(predicted_class_index, "Unknown")
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

# Function to query the Gemini AI model for information about the disease
def get_disease_info(disease_name):
    prompt = (f"Provide detailed information about the '{disease_name}' disease, including:\n"
              f"- The actual name of the plant affected\n"
              f"- Description of the disease\n"
              f"- Regions in India where it is mostly found\n"
              f"- Weather conditions it requires\n"
              f"- Solutions or treatments for the disease.")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

def translate_text(text, target_language):
    try:
        # Initialize the translator with the target language
        translator = get_translator(target_language)
        # Translate the text
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"Error translating text: {e}"

def main():
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Set a custom background
    page_bg_img = f"""
    <style>
    body {{
        background-image: url("pixel.jpg");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main {{
         background: #228B22;
        border-radius: 15px;
        padding: 20px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Title and subtitle
    st.title("🌾 Plant Disease Detection App")
    st.subheader("Upload an image of a crop leaf, and the app will predict the disease.")
    
    with st.sidebar:
        st.markdown("### 🌱 Features")
        st.markdown("- **Predict Plant Diseases**")
        st.markdown("- **Disease Information**")
        st.markdown("- **Multilingual Solutions**")
        st.markdown("- **Visual Confidence Scores**")
        st.markdown("### 👨‍💻 About")
        st.markdown("This app uses machine learning and Gemini AI to analyze and solve plant health issues.")

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
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    # Language selection for translation
    target_language = st.selectbox("Select a language for the solution:", ["en", "hi", "te", "ta", "bn", "ml", "mr"])

    if uploaded_file is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Classifying...")

            # Prediction
            predicted_class, confidence = predict_image_class(model, image, class_indices_rev)
            st.success(f"### Prediction: {predicted_class}")
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Get additional information from the Gemini AI model
            disease_info = get_disease_info(predicted_class)
            st.write("### Disease Information:")
            st.info(disease_info)

            # Translate the disease information into the selected language
            translated_info = translate_text(disease_info, target_language)
            st.write("### Translated Solution:")
            st.markdown(f"**{translated_info}**")

        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()
