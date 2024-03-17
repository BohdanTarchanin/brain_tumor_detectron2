import streamlit as st
#from detectron2.config import get_cfg
#from detectron2.engine import DefaultPredictor
#from detectron2 import model_zoo
from PIL import Image
import numpy as np
import requests
import os
import time
import gdown
import torch
from io import BytesIO
import tempfile
from util import visualize

# set title
st.title('МРТ головного мозку виявлення пухлини')

# set header
st.header('Будь ласка, завантажте зображення')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

