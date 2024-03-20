import streamlit as st
#from detectron2.config import get_cfg
#from detectron2.engine import DefaultPredictor
#from detectron2 import model_zoo
from PIL import Image
import numpy as np
import os
import torch
from io import BytesIO
import tempfile
from util import visualize

# Function to download model file if it doesn't exist
def download_model_if_not_exists(url, output):
    if not os.path.exists(output):
        with st.spinner('Downloading model file...'):
            gdown.download(url, output)
    return output

# set title
st.title('Розпізнавання і виявлення патологічних утворень головного мозку')

# set header
st.write('Ця програма дозволяє завантажувати зображення МРТ для виявлення пухлини мозку')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

url = "https://drive.google.com/uc?id=1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA"
output = "model.pth"
download_model_if_not_exists(url, output)


