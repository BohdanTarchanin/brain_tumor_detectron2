import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
import torch
from io import BytesIO
import tempfile

# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Define function to load model from URL
def load_model_from_url(url):
    response = requests.get(url)
    model_bytes = BytesIO(response.content)
    return model_bytes

# Replace 'GOOGLE_DRIVE_LINK' with the actual shareable link obtained from Google Drive
google_drive_link = 'https://drive.google.com/uc?id=14TmpFeOoMXSt7VYMtDrL3q98spZSNWo2'

# Load the model.pth file from the Google Drive link
model_bytes = load_model_from_url(google_drive_link)

# Save the BytesIO object to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(model_bytes.read())
    temp_file_path = temp_file.name

# Load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = temp_file_path  # Use the file path
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)
