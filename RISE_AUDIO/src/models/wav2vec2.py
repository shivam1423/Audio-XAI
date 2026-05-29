import torch
import torch.nn as nn
import sys
import os
from src.utils import NUM_CLASSES
from src.models.base import ModelInputType
from src.preprocessor import RawAudioPreprocessor

# Add path to custom Wav2Vec2 model
wav2vec2_path = os.path.join(os.path.dirname(__file__), '../../../Wav2Vec2')
if wav2vec2_path not in sys.path:
    sys.path.insert(0, wav2vec2_path)

from model.wav2vec2_classifier import Wav2Vec2Classifier


def Wav2Vec2Model(weights_path=None):
    """
    Setup and load the Wav2Vec2 model for ESC-50 classification.
    
    Uses the CUSTOM Wav2Vec2Classifier architecture (same as training):
    - Wav2Vec2Model (facebook/wav2vec2-base)
    - 2-layer classifier: Dropout → Linear(768→384) → ReLU → Dropout → Linear(384→50)
    
    This ensures checkpoint compatibility with the trained model.
    """
    if weights_path is None:
        weights_path = 'checkpoints/best_model_wav2vec2.pt'
    
    # Create model using CUSTOM Wav2Vec2Classifier (same as training)
    print("Loading custom Wav2Vec2Classifier architecture...")
    model = Wav2Vec2Classifier(
        model_name="facebook/wav2vec2-base",
        num_classes=NUM_CLASSES,
        dropout_rate=0.1
    )
    
    # Load state dict with proper device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading checkpoint from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        if next(iter(state)).startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        
        model.load_state_dict(state)
        print("✓ Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {weights_path}, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    # Wrap in DataParallel if CUDA available
    # if torch.cuda.is_available():
    #     model = nn.DataParallel(model)
    # Wrap in DataParallel if CUDA available
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            # Use both GPUs if available
            device_ids = [0, 1]
            model = nn.DataParallel(model, device_ids=device_ids)
            print(f"✓ Using DataParallel on GPUs: {device_ids}")
        else:
            model = nn.DataParallel(model)
            print(f"✓ Using DataParallel on GPU: 0")
    
    # Add predict method to the model
    class ModelWithPredict:
        def __init__(self, model):
            self.model = model
            # Model metadata
            self.input_type = ModelInputType.RAW_AUDIO
            self.sample_rate = 16000  # wav2vec2 expects 16kHz audio
            self.preprocessor = RawAudioPreprocessor(target_sample_rate=16000)  # Model-specific preprocessing

        def predict(self, input_tensor):
            """
            Predict method that returns logits and probabilities.
            
            Note: Custom Wav2Vec2Classifier returns logits directly (not wrapped in outputs object)
            """
            with torch.no_grad():
                # Handle DataParallel wrapper
                if isinstance(self.model, nn.DataParallel):
                    # Get the actual model from DataParallel
                    actual_model = self.model.module
                    logits = actual_model(input_tensor)  # Returns logits directly
                else:
                    # No DataParallel wrapper
                    logits = self.model(input_tensor)  # Returns logits directly
                
                probs = torch.softmax(logits, dim=-1)
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
        logits, probs = model.predict(input_tensor)
        predictions, classes = torch.max(probs, dim=1)
    return predictions, classes

