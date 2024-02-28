import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from PIL import Image
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_feature = model.roi_heads.box_predictor.cls_score.in_feature
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes)

