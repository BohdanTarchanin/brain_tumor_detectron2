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
from util import visualize

# Function to download model file if it doesn't exist
@st.cache(allow_output_mutation=True)
def load_model(url, output):
    if not os.path.exists(output):
        st.write("Downloading model file...")
        with st.spinner('Downloading model file...'):
            gdown.download(url, output)
        st.write("Model file downloaded successfully.")
    else:
        st.write("Using existing model file.")
    return output

# set title
st.title('МРТ головного мозку виявлення пухлини')

# set header
st.header('Будь ласка, завантажте зображення')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

url = "https://drive.google.com/uc?id=1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA"
output = "model.pth"
output = load_model(url, output)

# load model
@st.cache(allow_output_mutation=True)
def load_detector(output):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = output
    cfg.MODEL.DEVICE = 'cpu'
    return DefaultPredictor(cfg)

predictor = load_detector(output)

# load image
if file:
    image = Image.open(file).convert('RGB')

    image_array = np.asarray(image)

    # detect objects
    outputs = predictor(image_array)

    threshold = 0.5

    # Display predictions
    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    bboxes_ = []
    for j, bbox in enumerate(bboxes):
        bbox = bbox.tolist()

        score = scores[j]
        pred = preds[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            bboxes_.append([x1, y1, x2, y2])

    # visualize
    visualize(image, bboxes_)
