"""
Wav2Vec2-based classifier for ESC-50 sound classification
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional
from config import Config


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2-based classifier for environmental sound classification
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 50,
        dropout_rate: float = 0.1,
        freeze_feature_extractor: bool = False
    ):
        """
        Initialize Wav2Vec2 classifier
        
        Args:
            model_name: Pre-trained Wav2Vec2 model name
            num_classes: Number of output classes
            dropout_rate: Dropout rate for classifier head
            freeze_feature_extractor: Whether to freeze Wav2Vec2 feature extractor
        """
        super(Wav2Vec2Classifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Get hidden size from Wav2Vec2 config
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_values: Input audio tensor of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get Wav2Vec2 outputs
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(hidden_states, dim=1)  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_feature_extractor_output(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get feature extractor output (useful for analysis)
        
        Args:
            input_values: Input audio tensor
            
        Returns:
            Feature extractor output
        """
        with torch.no_grad():
            features = self.wav2vec2.feature_extractor(input_values)
        return features
    
    def freeze_wav2vec2(self):
        """Freeze Wav2Vec2 parameters"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def unfreeze_wav2vec2(self):
        """Unfreeze Wav2Vec2 parameters"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True


class Wav2Vec2ForSequenceClassification(nn.Module):
    """
    Alternative implementation using HuggingFace's sequence classification approach
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 50,
        dropout_rate: float = 0.1
    ):
        super(Wav2Vec2ForSequenceClassification, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_classes)
        )
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        
        # Use pooled output (mean of last hidden state)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(pooled_output)
        
        return logits


def create_model(
    model_name: str = "facebook/wav2vec2-base",
    num_classes: int = 50,
    dropout_rate: float = 0.1,
    freeze_feature_extractor: bool = False
) -> Wav2Vec2Classifier:
    """
    Factory function to create Wav2Vec2 classifier
    
    Args:
        model_name: Pre-trained model name
        num_classes: Number of classes
        dropout_rate: Dropout rate
        freeze_feature_extractor: Whether to freeze feature extractor
        
    Returns:
        Wav2Vec2Classifier instance
    """
    return Wav2Vec2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_feature_extractor=freeze_feature_extractor
    )

