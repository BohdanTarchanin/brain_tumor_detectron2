import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
import os
import time
import gdown
import torch
from io import BytesIO
import tempfile

# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

url = "https://drive.google.com/uc?id=1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA"
output = "model.pth"
gdown.download(url, output)

# load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './model/model.pth'
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)
