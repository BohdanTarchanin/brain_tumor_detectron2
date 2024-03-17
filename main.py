import streamlit as st
#from detectron2.config import get_cfg
#from detectron2.engine import DefaultPredictor
#from detectron2 import model_zoo
from PIL import Image
import numpy as np

from util import visualize


# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])
