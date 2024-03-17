import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
from io import BytesIO

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
