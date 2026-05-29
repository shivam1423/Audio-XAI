import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
from src.utils import MODEL_WEIGHTS_PATH, MODEL_INPUT_SIZE, NUM_CLASSES
from src.models.base import ModelInputType
from src.preprocessor import ResNetPreprocessor


def ResNetModel(weights_path=None):
    """Setup and load the ResNet50 model for ESC-50 classification."""
    if weights_path is None:
        weights_path = MODEL_WEIGHTS_PATH

    # Create backbone
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)

    # Load state dict with proper device handling
    # state = torch.load(weights_path, map_location=device)
    # if next(iter(state)).startswith('module.'):
    #     state = {k.replace('module.', ''): v for k, v in state.items()}
    # backbone.load_state_dict(state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint without unpickling custom modules
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch < 2.0 has no weights_only
        state = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state, dict) and not isinstance(state.get('model_state_dict', state), type(None)):
        state = state.get('model_state_dict', state.get('state_dict', state))
    if next(iter(state)).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    backbone.load_state_dict(state, strict=False)

    # Create model with softmax
    model = nn.Sequential(backbone, nn.Softmax(dim=1))
    model = model.to(device)
    model.eval()

    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    # Wrap in DataParallel if CUDA available
    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    # Add predict method to the model
    class ModelWithPredict:
        def __init__(self, model):
            self.model = model
            # Model metadata
            self.input_type = ModelInputType.SPECTROGRAM
            self.sample_rate = 22050  # ResNet expects mel specs from 22050 Hz audio
            self.preprocessor = ResNetPreprocessor()  # Model-specific preprocessing

        def predict(self, input_tensor):
            """Predict method that returns logits and probabilities."""
            with torch.no_grad():
                # Handle DataParallel wrapper
                if isinstance(self.model, nn.DataParallel):
                    # Get the actual model from DataParallel
                    actual_model = self.model.module
                    # Get backbone (first part without softmax)
                    backbone = actual_model[0]
                    logits = backbone(input_tensor)
                    probs = self.model(input_tensor)  # Full model with softmax
                else:
                    # No DataParallel wrapper
                    backbone = self.model[0]
                    logits = backbone(input_tensor)
                    probs = self.model(input_tensor)
                return logits, probs

        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)

        def parameters(self):
            return self.model.parameters()

        def eval(self):
            self.model.eval()
            return self

    return ModelWithPredict(model)


def get_model_predictions(model, input_tensor):
    """Get model predictions for input tensor."""
    with torch.no_grad():
        predictions, classes = torch.max(model(input_tensor), dim=1)
    return predictions, classes