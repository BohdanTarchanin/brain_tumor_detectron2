import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile

from util import visualize

# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Replace 'GOOGLE_DRIVE_LINK' with the actual shareable link obtained from Google Drive
google_drive_link = 'https://drive.google.com/file/d/14TmpFeOoMXSt7VYMtDrL3q98spZSNWo2/view?usp=sharing'

# Download the model.pth file from the Google Drive link
response = requests.get(google_drive_link)
model_bytes = BytesIO(response.content)

# Save the BytesIO object to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(model_bytes.read())
    temp_file_path = temp_file.name

# load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = temp_file_path  # Use the file path
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)
