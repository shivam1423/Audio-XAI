#!/usr/bin/env python
# coding: utf-8

"""Randomized Input Sampling for Explanations (RISE) – Audio version.

This script mirrors `RISE_original/Saliency.py` but works on the ESC-50 audio
classification dataset.  The high-level flow is identical:

1. Load data (now: spectrogram "images" generated on-the-fly).
2. Load a pretrained ResNet-50 model (no weights changed; we simply replicate
   the single-channel spectrograms to 3 channels).
3. Instantiate a RISE explainer.
4. Generate masks or load cached ones.
5. Produce saliency maps for a single example and / or for a batch.
"""

import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.models as models

from utils import (
        Dummy,
        ESC50SpectrogramDataset,
        preprocess,
        tensor_imshow,
        get_class_name,
        read_tensor,
        RangeSampler,
    )
from explanations import RISE

cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Args (kept in a Dummy container to preserve original coding style)
# -----------------------------------------------------------------------------

args = Dummy(
    workers=4,  # ESC-50 is small; no need for many workers
    esc50_root="ESC50",  # path **relative to project root**
    range=range(0, 50),  # subset for quick demo – change as you wish
    input_size=(224, 224),
    gpu_batch=250,
)

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

full_dataset = ESC50SpectrogramDataset(args.esc50_root, transform=preprocess)

# --------------------------------------------------------------------------
# Select 2 random samples per class  →  50 classes × 2 = 100 items
# --------------------------------------------------------------------------
import random
random.seed(42)  # reproducibility
indices = []
_targets = np.array(full_dataset.targets)
for _cls in range(len(full_dataset.classes)):
    cls_pool = np.where(_targets == _cls)[0]
    indices.extend(random.sample(cls_pool.tolist(), 2))
random.shuffle(indices)

# DataLoader (batch size 1, same as original)
data_loader = torch.utils.data.DataLoader(
    full_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    sampler=RangeSampler(indices),
)

print(
    f"Drawn {len(indices)} samples (2 per class) from the dataset of {len(full_dataset)} files.",
)
print(f"      {len(data_loader) * data_loader.batch_size:5d} items will be explained.")

# -----------------------------------------------------------------------------
# Generate / load RISE masks ONCE
# -----------------------------------------------------------------------------
maskspath = os.path.join(os.path.dirname(__file__), "masks.npy")
if not os.path.isfile(maskspath):
    _tmp_explainer = RISE(nn.Identity(), args.input_size, args.gpu_batch)
    _tmp_explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    print("Masks loaded from disk.")

# -----------------------------------------------------------------------------
# Iterate over the 5 fine-tuned folds
# -----------------------------------------------------------------------------
from pathlib import Path

for fold_id in range(1, 6):
    ckpt_path = f"resnet50_esc50_fold{fold_id}.pt"
    print(f"\n=== Processing fold {fold_id}: loading {ckpt_path} ===")

    # 1) Build architecture & load weights
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 50)
    state = torch.load(ckpt_path, map_location="cuda")
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    backbone.load_state_dict(state)

    model = nn.Sequential(backbone, nn.Softmax(dim=1)).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = nn.DataParallel(model)

    # 2) Explainer that re-uses the pre-generated masks
    explainer = RISE(model, args.input_size, args.gpu_batch)
    explainer.load_masks(maskspath)
    explainer.p1 = float(explainer.masks.mean())

    # 3) Predict labels for the subset (needed to pick class-specific map)
    subset_targets = np.empty(len(data_loader), np.int64)
    for i, (spec, _) in enumerate(tqdm(data_loader, desc=f"Fold {fold_id} – predict")):
        _, c = torch.max(model(spec.cuda()), dim=1)
        subset_targets[i] = c.item()

    # 4) Compute saliency maps
    explanations = np.empty((len(data_loader), *args.input_size))
    for i, (spec, _) in enumerate(tqdm(data_loader, desc=f"Fold {fold_id} – explain")):
        sal_maps = explainer(spec.cuda())
        explanations[i] = sal_maps[subset_targets[i]].cpu().numpy()

    # 5) Save overlay images
    out_dir = Path("saliency_outputs") / f"fold_{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, (spec, _) in enumerate(data_loader):
        orig_idx = indices[local_idx]
        base_name = Path(full_dataset.fnames[orig_idx]).stem
        prob, cls_id = torch.max(model(spec.cuda()), dim=1)
        prob, cls_id = prob.item(), cls_id.item()

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.axis("off")
        plt.title(f"{100 * prob:.2f}% {get_class_name(cls_id)}")
        tensor_imshow(spec[0])

        plt.subplot(122)
        plt.axis("off")
        plt.title(get_class_name(cls_id))
        tensor_imshow(spec[0])
        plt.imshow(explanations[local_idx], cmap="jet", alpha=0.5)

        out_file = out_dir / f"{base_name}_saliency.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

    print(f"Saved saliency overlays for fold {fold_id} to {out_dir}")

# -----------------------------------------------------------------------------
# End of script (interactive demo removed; everything is saved to disk)
# ----------------------------------------------------------------------------- 