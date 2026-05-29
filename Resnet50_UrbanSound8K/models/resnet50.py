"""
ResNet50 Model for UrbanSound8K Classification

This wrapper keeps the architecture as close as possible to the ESC-50
ResNet50 used elsewhere in the project:

- Standard torchvision ResNet50 backbone (3‑channel input, 224×224 images)
- Final fully‑connected layer changed to 10 outputs for UrbanSound8K
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50AudioClassifier(nn.Module):
    """
    ResNet50 model adapted for audio classification.

    Differences from vanilla ImageNet ResNet50:
    - Final fully connected layer outputs `num_classes` (10 for UrbanSound8K).
    - Input remains 3‑channel 224×224 images, matching ESC-50 ResNet50.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Initialize ResNet50 audio classifier.

        Args:
            num_classes: Number of output classes (default: 10 for UrbanSound8K)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
        """
        super(ResNet50AudioClassifier, self).__init__()

        self.num_classes = num_classes

        # Load standard torchvision ResNet50 backbone
        if pretrained:
            # Use weights parameter for newer torchvision versions
            try:
                self.resnet = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V1
                )
            except Exception:
                # Fallback for older torchvision versions
                self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet50(pretrained=False)

        # Replace final fully connected layer with dataset‑specific head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            logits: Output logits (batch_size, num_classes)
        """
        return self.resnet(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            features: Feature vector (batch_size, 2048)
        """
        # Forward through all layers except the final fc layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def create_model(config):
    """
    Create ResNet50 model based on configuration.

    Args:
        config: Configuration object with model parameters

    Returns:
        model: ResNet50AudioClassifier model
    """
    model = ResNet50AudioClassifier(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
    )

    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_summary(model):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model
    """
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    print("Model: ResNet50 Audio Classifier")

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")

    print("\nLayer structure:")
    print("  - Input: (batch, 3, 224, 224) - 3‑channel mel spectrogram image")
    print("  - ResNet Backbone: 4 residual blocks")
    print("  - Global Average Pooling")
    print(f"  - FC: 2048 -> {model.num_classes} classes")
    print("=" * 70)


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNet50 Audio Classifier...")

    # Create a dummy config class for testing
    class DummyConfig:
        num_classes = 10
        pretrained = True

    config = DummyConfig()

    # Create model
    model = create_model(config)
    print_model_summary(model)

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 spectrogram images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    # Test feature extraction
    print("\nTesting feature extraction...")
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")

    print("\nAll tests passed!")
