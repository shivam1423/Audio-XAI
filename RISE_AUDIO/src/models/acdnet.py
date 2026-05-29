import torch
import torch.nn as nn
import sys
import os
from src.utils import NUM_CLASSES
from src.models.base import ModelInputType
from src.preprocessor import RawAudioPreprocessor

# Import ACDNet model
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ACDNet/torch'))
from resources.models import GetACDNetModel


def ACDNetModel(weights_path=None, sample_rate=20000, input_length=30225):
    """Setup and load the ACDNet model for ESC-50 classification."""

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights_path, map_location=device)

    # DEBUG: Print checkpoint keys
    print(f"DEBUG: Checkpoint keys: {state.keys()}")
    if 'config' in state:
        print(f"DEBUG: Config type: {type(state['config'])}")
        print(f"DEBUG: Config content: {state['config']}")
    if 'weight' in state:
        print(f"DEBUG: Has 'weight' key")
    if 'model_state_dict' in state:
        print(f"DEBUG: Has 'model_state_dict' key")

    if 'weight' in state:
        # Original ACDNet format (ESC50)
        weight = state['weight']
        config = state['config']  # This is channel_config (list)
    elif 'model_state_dict' in state:
        # New ACDNet_UrbanSound8K format
        weight = state['model_state_dict']
        # The 'config' here is training config (dict), not channel config
        # Use None to let ACDNet use default channel configuration
        config = None
    else:
        # Fallback: assume it's a direct state dict
        weight = state
        config = None
    # Create model
    model = GetACDNetModel(input_length, NUM_CLASSES, sample_rate, config).to(device)
    model.load_state_dict(weight)
    model.eval()

    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    # Wrap with DataParallel if CUDA available
    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    # Wrap with predict method
    class ModelWithPredict:
        def __init__(self, model):
            self.model = model
            # Model metadata
            self.input_type = ModelInputType.RAW_AUDIO
            self.sample_rate = sample_rate  # ACDNet uses 20kHz
            self.input_length = input_length  # Expected input length (30225 for 1.5s @ 20kHz)
            self.preprocessor = RawAudioPreprocessor(target_sample_rate=sample_rate, fixed_length=input_length)

        def predict(self, input_tensor):
            """Predict method that returns logits and probabilities."""
            with torch.no_grad():
                # Handle DataParallel wrapper
                if isinstance(self.model, nn.DataParallel):
                    actual_model = self.model.module
                else:
                    actual_model = self.model

                # ACDNet expects (batch, 1, 1, samples) - 4D input for Conv2D layers
                # It treats 1D audio as 2D image: (batch, channels=1, height=1, width=samples)
                if input_tensor.dim() == 2:
                    # (batch, samples) -> (batch, 1, 1, samples)
                    input_tensor = input_tensor.unsqueeze(1).unsqueeze(1)
                elif input_tensor.dim() == 1:
                    # (samples,) -> (1, 1, 1, samples)
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                elif input_tensor.dim() == 3:
                    # (batch, 1, samples) -> (batch, 1, 1, samples)
                    input_tensor = input_tensor.unsqueeze(2)

                # Get probabilities (ACDNet already has softmax)
                probs = self.model(input_tensor)

                # Calculate logits (inverse softmax)
                logits = torch.log(probs + 1e-10)

                return logits, probs

        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)

        def parameters(self):
            return self.model.parameters()

        def eval(self):
            self.model.eval()
            return self

    return ModelWithPredict(model)