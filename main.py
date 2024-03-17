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

# load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = 'model.pth'
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

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
