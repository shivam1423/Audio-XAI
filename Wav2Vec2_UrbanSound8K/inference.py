#!/usr/bin/env python3
"""
Inference script for Wav2Vec2 UrbanSound8K model
"""
import argparse
import os
import sys
import torch
import torchaudio
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model.wav2vec2_classifier import Wav2Vec2Classifier
from utils.helpers import get_device


def load_model(checkpoint_path: str, device: torch.device) -> Wav2Vec2Classifier:
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if isinstance(checkpoint.get('config'), dict):
        config_dict = checkpoint['config']
        model_name = config_dict.get('MODEL_NAME', 'facebook/wav2vec2-base')
        num_classes = config_dict.get('NUM_CLASSES', 10)
        dropout_rate = config_dict.get('DROPOUT_RATE', 0.1)
    else:
        # Fallback to defaults
        model_name = 'facebook/wav2vec2-base'
        num_classes = 10
        dropout_rate = 0.1
    
    # Create model
    model = Wav2Vec2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_audio(
    file_path: str, 
    target_sample_rate: int = 16000,
    max_duration: float = 4.0
) -> torch.Tensor:
    """
    Preprocess audio file for inference
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate
        max_duration: Maximum audio duration in seconds
        
    Returns:
        Preprocessed audio tensor
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to 1D
    waveform = waveform.squeeze(0)
    
    # Pad or truncate to max_duration
    max_length = int(max_duration * target_sample_rate)
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        padding = max_length - len(waveform)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform


def predict_single_file(
    model: Wav2Vec2Classifier,
    file_path: str,
    class_names: List[str],
    device: torch.device,
    target_sample_rate: int = 16000,
    max_duration: float = 4.0
) -> Tuple[str, float, List[float]]:
    """
    Predict class for a single audio file
    
    Args:
        model: Trained model
        file_path: Path to audio file
        class_names: List of class names
        device: Device to run inference on
        target_sample_rate: Target sample rate
        max_duration: Maximum audio duration
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Preprocess audio
    audio = preprocess_audio(file_path, target_sample_rate, max_duration)
    audio = audio.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        logits = model(audio)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    predicted_class = class_names[predicted_class_idx]
    all_probabilities = probabilities[0].cpu().numpy().tolist()
    
    return predicted_class, confidence, all_probabilities


def predict_batch(
    model: Wav2Vec2Classifier,
    file_paths: List[str],
    class_names: List[str],
    device: torch.device,
    target_sample_rate: int = 16000,
    max_duration: float = 4.0
) -> List[Tuple[str, str, float, List[float]]]:
    """
    Predict classes for multiple audio files
    
    Args:
        model: Trained model
        file_paths: List of audio file paths
        class_names: List of class names
        device: Device to run inference on
        target_sample_rate: Target sample rate
        max_duration: Maximum audio duration
        
    Returns:
        List of tuples (file_path, predicted_class, confidence, all_probabilities)
    """
    results = []
    
    for file_path in file_paths:
        try:
            predicted_class, confidence, all_probabilities = predict_single_file(
                model, file_path, class_names, device, target_sample_rate, max_duration
            )
            results.append((file_path, predicted_class, confidence, all_probabilities))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append((file_path, "ERROR", 0.0, []))
    
    return results


def print_top_predictions(probabilities: List[float], class_names: List[str], top_k: int = 5):
    """
    Print top-k predictions with probabilities
    
    Args:
        probabilities: List of probabilities for all classes
        class_names: List of class names
        top_k: Number of top predictions to show
    """
    # Get top-k indices (but limit to actual number of classes)
    top_k = min(top_k, len(probabilities))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    print(f"\nTop {top_k} predictions:")
    print("-" * 40)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {probabilities[idx]:.4f}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run inference with trained Wav2Vec2 model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--audio_file", type=str,
                       help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str,
                       help="Directory containing audio files")
    parser.add_argument("--output_file", type=str,
                       help="Output file to save predictions")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top predictions to show")
    
    args = parser.parse_args()
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Get class names
    config = Config()
    class_names = config.CLASS_NAMES
    
    # Determine input files
    if args.audio_file:
        file_paths = [args.audio_file]
    elif args.audio_dir:
        if not os.path.exists(args.audio_dir):
            print(f"Error: Directory {args.audio_dir} does not exist")
            sys.exit(1)
        
        # Get all audio files from directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        file_paths = []
        for root, dirs, files in os.walk(args.audio_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print(f"No audio files found in {args.audio_dir}")
            sys.exit(1)
        
        print(f"Found {len(file_paths)} audio files")
    else:
        print("Error: Please provide either --audio_file or --audio_dir")
        sys.exit(1)
    
    # Run inference
    print("Running inference...")
    results = predict_batch(
        model=model,
        file_paths=file_paths,
        class_names=class_names,
        device=device,
        target_sample_rate=config.SAMPLE_RATE,
        max_duration=config.MAX_DURATION
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    
    for file_path, predicted_class, confidence, all_probabilities in results:
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        if all_probabilities:
            print_top_predictions(all_probabilities, class_names, args.top_k)
    
    # Save results to file if specified
    if args.output_file:
        import csv
        with open(args.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Predicted_Class', 'Confidence'] + class_names)
            
            for file_path, predicted_class, confidence, all_probabilities in results:
                row = [os.path.basename(file_path), predicted_class, confidence] + all_probabilities
                writer.writerow(row)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
