import os
import json
from zipfile import ZipFile

# Load Kaggle credentials from the local kaggle.json file
with open("kaggle.json", "r") as file:
    kaggle_credentials = json.load(file)

# Set Kaggle credentials as environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]

# Download the dataset from Kaggle
# This command will run the Kaggle API to download the dataset
os.system('kaggle datasets download -d abdallahalidev/plantvillage-dataset')

# Unzip the downloaded dataset
with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

print("Download and extraction complete.")
