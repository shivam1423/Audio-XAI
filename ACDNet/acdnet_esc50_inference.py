#!/usr/bin/env python3
"""
ACDNet inference on individual ESC-50 audio files
"""

import os
import sys
import torch
import numpy as np
import librosa
import pandas as pd
from pathlib import Path

# Add ACDNet to path
sys.path.append('/beegfs/work_fast/pandey/Thesis/RISE_dev/ACDNet/torch')
from resources.models import GetACDNetModel

# Configuration
SAMPLE_RATE = 20000
INPUT_LENGTH = 30225  # ~1.5 seconds at 20kHz
NUM_CLASSES = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ESC-50 class labels
ESC50_LABELS = [
    'dog', 'rooster', 'pig', 'cow', 'frog',
    'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
    'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
    'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
    'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
    'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
    'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]


def load_model(model_path):
    """Load pretrained ACDNet model"""
    print(f"Loading model from: {model_path}")

    state = torch.load(model_path, map_location=DEVICE)
    config = state['config']  # Channel configuration
    weight = state['weight']  # Model weights

    # Create model
    model = GetACDNetModel(INPUT_LENGTH, NUM_CLASSES, SAMPLE_RATE, config).to(DEVICE)
    model.load_state_dict(weight)
    model.eval()

    return model


def preprocess_audio(audio_path):
    """
    Load and preprocess audio file for ACDNet
    Returns tensor of shape (1, 1, 1, INPUT_LENGTH)
    """
    # Load audio at 20kHz
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Handle length (crop or pad to INPUT_LENGTH)
    if len(audio) > INPUT_LENGTH:
        # Center crop
        start = (len(audio) - INPUT_LENGTH) // 2
        audio = audio[start:start + INPUT_LENGTH]
    elif len(audio) < INPUT_LENGTH:
        # Zero pad
        pad_len = INPUT_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')

    # Convert to tensor: (batch=1, channels=1, height=1, width=INPUT_LENGTH)
    audio_tensor = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return audio_tensor.to(DEVICE)


def predict_single_file(model, audio_path):
    """Get prediction for a single audio file"""
    audio_tensor = preprocess_audio(audio_path)

    with torch.no_grad():
        output = model(audio_tensor)  # Shape: (1, 50)
        probs = output.squeeze(0).cpu().numpy()  # Get probabilities
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

    return pred_class, confidence, probs


def predict_esc50_dataset(model, esc50_root, output_csv='esc50_predictions.csv'):
    """
    Run inference on all ESC-50 files

    Args:
        model: Loaded ACDNet model
        esc50_root: Path to ESC-50 dataset root (contains audio/ and meta/)
        output_csv: Output CSV file path
    """
    # Load metadata
    meta_path = os.path.join(esc50_root, 'meta', 'esc50.csv')
    df = pd.read_csv(meta_path)

    results = []

    print(f"Processing {len(df)} files...")
    for idx, row in df.iterrows():
        filename = row['filename']
        fold = row['fold']
        true_class = row['target']

        # Construct file path
        audio_path = os.path.join(esc50_root, 'audio', filename)

        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            continue

        # Get prediction
        pred_class, confidence, probs = predict_single_file(model, audio_path)
        pred_label = ESC50_LABELS[pred_class]
        true_label = ESC50_LABELS[true_class]

        # Store result
        results.append({
            'filename': filename,
            'fold': fold,
            'true_class': true_class,
            'true_label': true_label,
            'pred_class': pred_class,
            'pred_label': pred_label,
            'confidence': confidence,
            'correct': pred_class == true_class
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} files...")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    # Print summary
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"\nResults saved to: {output_csv}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Correct: {results_df['correct'].sum()}/{len(results_df)}")

    return results_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ACDNet inference on ESC-50')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to ACDNet checkpoint (.pt file)')
    parser.add_argument('--esc50_root', type=str, required=True,
                        help='Path to ESC-50 dataset root directory')
    parser.add_argument('--output_csv', type=str, default='esc50_predictions.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)

    # Run inference
    results = predict_esc50_dataset(model, args.esc50_root, args.output_csv)