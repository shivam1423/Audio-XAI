#!/usr/bin/env python
"""
Per-file ESC-50 prediction for ResNet50 cross-validation folds.

For each checkpoint resnet50_esc50_fold{N}.pt (N=1..5), this script:
- Loads the model
- Runs inference on ALL ESC-50 audio files (using the same mel-spectrogram
  preprocessing as training, via ESC50SpectrogramDataset)
- Saves a CSV with columns:
    filename, fold, true_class, true_label,
    pred_class, pred_label, confidence, correct

Example
-------
python resnet50_esc50_per_file_inference.py \
    --esc50_root /path/to/ESC-50 \
    --checkpoints_dir /path/to/ResNet50/checkpoints \
    --output_dir /path/to/output_csvs \
    --device cuda \
    --batch_size 32
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader

# Local imports (works both as package and as script)
try:
    from .utils import ESC50SpectrogramDataset, preprocess as default_preprocess
except ImportError:
    from utils import ESC50SpectrogramDataset, preprocess as default_preprocess  # type: ignore


def load_resnet50(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a ResNet-50 ESC-50 model from checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 50)

    state = torch.load(checkpoint_path, map_location=device)
    # Handle DataParallel checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    backbone.load_state_dict(state)
    model = backbone.to(device)
    model.eval()
    return model


def run_per_file_inference_for_fold(
    fold_id: int,
    checkpoint_path: str,
    esc50_root: str,
    df: pd.DataFrame,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> None:
    """
    Run inference for one fold model on ALL ESC-50 files and save a CSV.

    - fold_id: 1..5 (used only in filenames / logging; per-row fold comes from df["fold"])
    - df: ESC-50 metadata DataFrame (meta/esc50.csv)
    - loader: DataLoader over ESC50SpectrogramDataset with shuffle=False
    """
    print(f"\n=== Fold {fold_id}: loading {checkpoint_path} ===")
    model = load_resnet50(checkpoint_path, device)

    all_preds: List[int] = []
    all_confs: List[float] = []

    with torch.no_grad():
        for specs, _labels in tqdm(loader, desc=f"Inference (fold {fold_id})"):
            specs = specs.to(device)
            logits = model(specs)                 # (B, 50)
            probs = torch.softmax(logits, dim=1)  # (B, 50)
            pred_idx = probs.argmax(dim=1)        # (B,)
            conf = probs[torch.arange(probs.size(0)), pred_idx]  # (B,)

            all_preds.extend(pred_idx.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())

    if len(all_preds) != len(df):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(all_preds)} preds, "
            f"but df has {len(df)} rows."
        )

    # Build mapping target -> category from ESC-50 metadata
    target_to_label = (
        df.drop_duplicates("target")
          .set_index("target")["category"]
          .str.strip()
          .to_dict()
    )

    records = []
    for i in range(len(df)):
        filename = df.loc[i, "filename"]
        fold_orig = int(df.loc[i, "fold"])      # official ESC-50 fold (1..5)
        true_class = int(df.loc[i, "target"])
        true_label = str(df.loc[i, "category"]).strip()

        pred_class = int(all_preds[i])
        confidence = float(all_confs[i])
        pred_label = target_to_label.get(pred_class, "")

        correct = pred_class == true_class

        records.append({
            "filename": filename,
            "fold": fold_orig,
            "true_class": true_class,
            "true_label": true_label,
            "pred_class": pred_class,
            "pred_label": pred_label,
            "confidence": confidence,
            "correct": correct,
        })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"esc50_predictions_resnet50_fold{fold_id}.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved per-file predictions for fold {fold_id} to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-file ESC-50 predictions for ResNet50 cross-validation folds"
    )
    p.add_argument(
        "--esc50_root",
        type=str,
        required=True,
        help="Path to ESC-50 root (contains 'audio/' and 'meta/esc50.csv')",
    )
    p.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing resnet50_esc50_fold{N}.pt checkpoints",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save per-fold prediction CSVs",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g. 'cuda', 'cpu')",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    return p.parse_args()


def main():
    args = parse_args()

    esc50_root = Path(args.esc50_root)
    meta_path = esc50_root / "meta" / "esc50.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")

    # Load metadata once
    df = pd.read_csv(meta_path)

    # Dataset & DataLoader over ALL ESC-50 files (same order as df)
    dataset = ESC50SpectrogramDataset(
        esc50_root=str(esc50_root),
        csv_name="meta/esc50.csv",
        transform=default_preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Run for folds 1..5
    for fold_id in range(1, 6):
        ckpt = Path(args.checkpoints_dir) / f"resnet50_esc50_fold{fold_id}.pt"
        run_per_file_inference_for_fold(
            fold_id=fold_id,
            checkpoint_path=str(ckpt),
            esc50_root=str(esc50_root),
            df=df,
            loader=loader,
            device=device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()