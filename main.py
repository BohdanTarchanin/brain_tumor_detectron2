import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import os
import torch
from io import BytesIO
import tempfile
import gdown
from util import visualize

# set title
st.title('Розпізнавання і виявлення патологічних утворень головного мозку')

# set header
st.write('Ця програма дозволяє завантажувати зображення МРТ для виявлення пухлини мозку')

# upload file
file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

url = "https://drive.google.com/uc?id=1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA"
output = "model.pth"

# Download model file if it doesn't exist
if not os.path.exists(output):
    with st.spinner('Downloading model file...'):
        gdown.download(url, output)
