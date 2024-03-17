import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
import os
import time
import torch
from io import BytesIO
import tempfile

# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

checkFile = "model.pth"
if os.path.exists(checkFile) == False:
    print('I miss :', checkFile)
    msg = st.warning("🚩 Models need to be downloaded... ")
    try:
        with st.spinner('Initiating...'):
            time.sleep(3)
            url_pth = "https://drive.google.com/file/d/1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA/uc?export=download"

            r_pth = requests.get(url_pth, allow_redirects=True)

            open("model.pth", 'wb').write(r_pth.content)                   
            del r_pth
            msg.success("Download was successful ✅")
    except:
        msg.error("Error downloading model files...😥")


