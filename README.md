```markdown
# 🌿 Plant Disease Detection App

This project uses a convolutional neural network (CNN) model to detect plant diseases from leaf images. The app predicts the disease based on the uploaded image of a plant's leaf and provides the disease name along with the confidence level of the prediction.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Instructions](#instructions)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)

## Overview
This app helps farmers and agricultural researchers detect diseases in crops by uploading images of leaves. The app predicts various plant diseases using a pre-trained model, and displays the prediction with a confidence score.

## Technologies Used
- Python
- TensorFlow/Keras
- Streamlit (for building the app interface)
- PIL (Python Imaging Library)
- NumPy
- JSON (for class mapping)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/plant-disease-detection-app.git
cd plant-disease-detection-app
```

### 2. Create and Activate a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

## Instructions

### Step 1: Preprocess the Dataset

1. First, ensure that you have a plant disease dataset (e.g., PlantVillage dataset).
2. Run the `dataset.py` file to preprocess the dataset.
   ```bash
   python dataset.py
   ```
   This script will prepare the images for model training by resizing, normalizing, and saving them.

### Step 2: Train the Model

1. After preprocessing the dataset, run `main.py` to train the CNN model on the dataset.
   ```bash
   python main.py
   ```
   Once the model is trained, it will save the model as `plant_disease_prediction_model.h5` in your project directory.

### Step 3: Run the Streamlit App

1. After training the model, run the `streamlit_app.py` to start the Streamlit web app.
   ```bash
   streamlit run streamlit_app.py
   ```
2. Upload an image of a leaf, and the app will predict the disease and display the confidence level.

## Dataset
The dataset used for this project contains images of leaves affected by various diseases. If you do not have the dataset, you can download the PlantVillage dataset, which is widely available online.

Ensure the dataset is properly organized before running `dataset.py`:
- Each disease type should be in its own folder, with images inside.
  
Example directory structure:
```
dataset/
    Apple___Apple_scab/
        img1.jpg
        img2.jpg
        ...
    Tomato___Bacterial_spot/
        img1.jpg
        img2.jpg
        ...
```

## Model Training
The CNN model is trained using TensorFlow/Keras, and it classifies the leaf images into one of the 38 disease/healthy classes. The model is saved as `plant_disease_prediction_model.h5` after training.

The class indices are stored in `class_indices.json`, mapping each class index to its respective class name.

## Running the App
Once you run the app using Streamlit, you'll be able to upload images and see the predictions. The app will display:
- The name of the predicted disease (or healthy if the plant is healthy)
- The confidence level of the prediction

## Project Structure
```bash
.
├── dataset.py               # Script to preprocess dataset
├── main.py                  # Script to train the model
├── streamlit_app.py          # Streamlit web application
├── plant_disease_prediction_model.h5  # Trained CNN model
├── class_indices.json        # Mapping of class indices to class names
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

## Future Improvements
- **Enhance Model**: Fine-tune the model for better accuracy.
- **More Disease Types**: Add support for additional plant diseases.
- **Deployment**: Deploy the app to a cloud service like Heroku or AWS.




### Key Points:
1. **Project Overview**: Provides a brief description of the purpose and features of the app.
2. **Technologies Used**: Lists the major tools and frameworks used.
3. **Installation**: Instructions to install dependencies and clone the repository.
4. **Instructions**: Step-by-step guide for running `dataset.py`, `main.py`, and `streamlit_app.py` in sequence.
5. **Dataset**: Describes the structure of the dataset.
6. **Model Training**: Notes on how the model is trained and saved.
7. **App Execution**: Explains how to run the Streamlit app.
8. **Future Improvements**: Offers suggestions for enhancing the project.

Once you've made any necessary adjustments (such as adding your GitHub URL), you can save this as `README.md` in your project directory.
