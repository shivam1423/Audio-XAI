#!/usr/bin/env python3
"""
Inference script for Wav2Vec2 ESC-50 model
"""
import argparse
import os
import sys
import torch
import torchaudio
import numpy as np
from typing import List, Optional, Tuple
import warnings
import csv
import pandas as pd
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
    
    # Create model
    model = Wav2Vec2Classifier(
        model_name=checkpoint['config'].MODEL_NAME,
        num_classes=checkpoint['config'].NUM_CLASSES,
        dropout_rate=checkpoint['config'].DROPOUT_RATE
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_audio(file_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Preprocess audio file for inference
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate
        
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
    
    return waveform


def predict_single_file(
    model: Wav2Vec2Classifier,
    file_path: str,
    class_names: List[str],
    device: torch.device,
    target_sample_rate: int = 16000
) -> Tuple[str, float, List[float]]:
    """
    Predict class for a single audio file
    
    Args:
        model: Trained model
        file_path: Path to audio file
        class_names: List of class names
        device: Device to run inference on
        target_sample_rate: Target sample rate
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Preprocess audio
    audio = preprocess_audio(file_path, target_sample_rate)
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
    target_sample_rate: int = 16000
) -> List[Tuple[str, str, float, List[float]]]:
    """
    Predict classes for multiple audio files
    
    Args:
        model: Trained model
        file_paths: List of audio file paths
        class_names: List of class names
        device: Device to run inference on
        target_sample_rate: Target sample rate
        
    Returns:
        List of tuples (file_path, predicted_class, confidence, all_probabilities)
    """
    results = []
    
    for file_path in file_paths:
        try:
            predicted_class, confidence, all_probabilities = predict_single_file(
                model, file_path, class_names, device, target_sample_rate
            )
            results.append((file_path, predicted_class, confidence, all_probabilities))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append((file_path, "ERROR", 0.0, []))
    
    return results

def run_esc50_evaluation(
    model: Wav2Vec2Classifier,
    esc50_root: str,
    class_names: List[str],
    device: torch.device,
    target_sample_rate: int = 16000,
    output_csv: Optional[str] = None,
) -> List[dict]:
    """
    Run inference on all ESC-50 files using meta/esc50.csv.
    Returns list of dicts with filename, fold, true_class, true_label,
    pred_class, pred_label, confidence, correct.
    """
    meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")

    df = pd.read_csv(meta_path)
    required = ["filename", "fold", "target", "category"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"esc50.csv missing column: {col}")

    rows = []
    for idx, row in df.iterrows():
        filename = row["filename"]
        fold = int(row["fold"])
        true_class = int(row["target"])
        true_label = str(row["category"]).strip()

        audio_path = os.path.join(esc50_root, "audio", filename)
        if not os.path.isfile(audio_path):
            print(f"Warning: missing {audio_path}")
            rows.append({
                "filename": filename,
                "fold": fold,
                "true_class": true_class,
                "true_label": true_label,
                "pred_class": -1,
                "pred_label": "",
                "confidence": 0.0,
                "correct": False,
            })
            continue
        target_to_label = (
            df.drop_duplicates("target")
            .set_index("target")["category"]
            .str.strip()
            .to_dict()
        )
        try:
            _, confidence, all_probs = predict_single_file(
                model, audio_path, class_names, device, target_sample_rate
            )
            pred_class = int(np.argmax(all_probs))
            pred_label = target_to_label.get(pred_class, "")  # ESC-50 category for this index
            correct = pred_class == true_class
            rows.append({
                "filename": filename,
                "fold": fold,
                "true_class": true_class,
                "true_label": true_label,
                "pred_class": pred_class,
                "pred_label": pred_label,
                "confidence": float(confidence),
                "correct": correct,
            })
        except Exception as e:
            print(f"Error {audio_path}: {e}")
            rows.append({
                "filename": filename,
                "fold": fold,
                "true_class": true_class,
                "true_label": true_label,
                "pred_class": -1,
                "pred_label": "",
                "confidence": 0.0,
                "correct": False,
            })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} files...")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["filename", "fold", "true_class", "true_label", "pred_class", "pred_label", "confidence", "correct"],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {output_csv}")

    n_correct = sum(1 for r in rows if r["correct"])
    print(f"Accuracy: {n_correct}/{len(rows)} = {100.0 * n_correct / len(rows):.2f}%")
    return rows

def print_top_predictions(probabilities: List[float], class_names: List[str], top_k: int = 5):
    """
    Print top-k predictions with probabilities
    
    Args:
        probabilities: List of probabilities for all classes
        class_names: List of class names
        top_k: Number of top predictions to show
    """
    # Get top-k indices
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    print(f"\nTop {top_k} predictions:")
    print("-" * 40)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {probabilities[idx]:.4f}")
# def run_esc50_evaluation(
#     model: Wav2Vec2Classifier,
#     esc50_root: str,
#     class_names: List[str],
#     device: torch.device,
#     target_sample_rate: int = 16000,
#     output_csv: Optional[str] = None,
# ) -> List[dict]:
#     """
#     Run inference on all ESC-50 files using meta/esc50.csv.
#     Returns list of dicts with filename, fold, true_class, true_label,
#     pred_class, pred_label, confidence, correct.
#     """
#     meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
#     if not os.path.isfile(meta_path):
#         raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")
#
#     df = pd.read_csv(meta_path)
#     required = ["filename", "fold", "target", "category"]
#     for col in required:
#         if col not in df.columns:
#             raise ValueError(f"esc50.csv missing column: {col}")
#
#     rows = []
#     for idx, row in df.iterrows():
#         filename = row["filename"]
#         fold = int(row["fold"])
#         true_class = int(row["target"])
#         true_label = str(row["category"]).strip()
#
#         audio_path = os.path.join(esc50_root, "audio", filename)
#         if not os.path.isfile(audio_path):
#             print(f"Warning: missing {audio_path}")
#             rows.append({
#                 "filename": filename,
#                 "fold": fold,
#                 "true_class": true_class,
#                 "true_label": true_label,
#                 "pred_class": -1,
#                 "pred_label": "",
#                 "confidence": 0.0,
#                 "correct": False,
#             })
#             continue
#
#         try:
#             pred_label, confidence, all_probs = predict_single_file(
#                 model, audio_path, class_names, device, target_sample_rate
#             )
#             pred_class = int(np.argmax(all_probs))
#             correct = pred_class == true_class
#             rows.append({
#                 "filename": filename,
#                 "fold": fold,
#                 "true_class": true_class,
#                 "true_label": true_label,
#                 "pred_class": pred_class,
#                 "pred_label": pred_label,
#                 "confidence": float(confidence),
#                 "correct": correct,
#             })
#         except Exception as e:
#             print(f"Error {audio_path}: {e}")
#             rows.append({
#                 "filename": filename,
#                 "fold": fold,
#                 "true_class": true_class,
#                 "true_label": true_label,
#                 "pred_class": -1,
#                 "pred_label": "",
#                 "confidence": 0.0,
#                 "correct": False,
#             })
#
#         if (idx + 1) % 100 == 0:
#             print(f"Processed {idx + 1}/{len(df)} files...")
#
#     if output_csv:
#         os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
#         with open(output_csv, "w", newline="") as f:
#             w = csv.DictWriter(
#                 f,
#                 fieldnames=["filename", "fold", "true_class", "true_label", "pred_class", "pred_label", "confidence", "correct"],
#             )
#             w.writeheader()
#             w.writerows(rows)
#         print(f"Saved {output_csv}")
#
#     n_correct = sum(1 for r in rows if r["correct"])
#     print(f"Accuracy: {n_correct}/{len(rows)} = {100.0 * n_correct / len(rows):.2f}%")
#     return rows

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run inference with trained Wav2Vec2 model")
    parser.add_argument("--esc50_root", type=str,
                        help="ESC-50 dataset root (audio/ and meta/esc50.csv). Enables per-file CSV with filename,fold,true_class,true_label,pred_class,pred_label,confidence,correct")
    parser.add_argument("--output_csv", type=str,
                        help="Output CSV path (required if --esc50_root is set)")
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
    if args.esc50_root:
        if not args.output_csv:
            print("Error: --output_csv is required when using --esc50_root")
            sys.exit(1)
        run_esc50_evaluation(
            model=model,
            esc50_root=args.esc50_root,
            class_names=class_names,
            device=device,
            target_sample_rate=config.SAMPLE_RATE,
            output_csv=args.output_csv,
        )
        return
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
        target_sample_rate=config.SAMPLE_RATE
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

