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
from util import visualize, set_background

set_background('./bg.jpg')

@st.cache_resource 
def load_model():
    url = "https://drive.google.com/uc?id=1T9MVlOqeOtrFLX1vOv1U-kWHoPy7J5aX"
    output = "model.pt"

    # Download model file if it doesn't exist
    if not os.path.exists(output):
        with st.spinner('Downloading model file...'):
            gdown.download(url, output)
    
    # load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = 'model.pt'
    cfg.MODEL.DEVICE = 'cpu'

    predictor = DefaultPredictor(cfg)
    return predictor



