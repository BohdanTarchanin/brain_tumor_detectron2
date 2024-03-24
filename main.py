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
    url = "https://drive.google.com/uc?id=1XTevverAgBxlZXRzpRdzR9gYM4YvoKgA"
    output = "model.pth"

    # Download model file if it doesn't exist
    if not os.path.exists(output):
        with st.spinner('Downloading model file...'):
            gdown.download(url, output)
    
    # load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = 'model.pth'
    cfg.MODEL.DEVICE = 'cpu'

    predictor = DefaultPredictor(cfg)
    return predictor

# set title
st.title('Розпізнавання і виявлення патологічних утворень головного мозку')

# set header
st.write('Ця програма дозволяє завантажувати зображення МРТ для виявлення паталогічних утворень головного мозку')

st.info('👇 Завантажуйте зображення МРТ')

# upload file
file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

# load model
predictor = load_model()

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



