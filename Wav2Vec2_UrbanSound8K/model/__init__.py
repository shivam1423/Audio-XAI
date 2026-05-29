"""
Wav2Vec2 model architecture for UrbanSound8K classification
"""
from .wav2vec2_classifier import Wav2Vec2Classifier, create_model

__all__ = ['Wav2Vec2Classifier', 'create_model']
