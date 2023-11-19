from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

predict_image = model('bus.jpg')
cv2.imshow(" ",predict_image)