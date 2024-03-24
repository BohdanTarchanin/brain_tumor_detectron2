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


