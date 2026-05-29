#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_WEIGHTS_PATH, MODEL_INPUT_SIZE, NUM_CLASSES


def setup_model(weights_path=None):
    """Setup and load the ResNet50 model for ESC-50 classification."""
    if weights_path is None:
        weights_path = MODEL_WEIGHTS_PATH
    
    # Create backbone
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    
    # Load state dict
    state = torch.load(weights_path, map_location='cuda')
    if next(iter(state)).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    backbone.load_state_dict(state)
    
    # Create model with softmax
    model = nn.Sequential(backbone, nn.Softmax(dim=1)).cuda()
    model.eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    # Wrap in DataParallel
    model = nn.DataParallel(model)
    
    return model


def get_model_predictions(model, input_tensor):
    """Get model predictions for input tensor."""
    with torch.no_grad():
        predictions, classes = torch.max(model(input_tensor.cuda()), dim=1)
    return predictions, classes
