#!/usr/bin/env python
# coding: utf-8
"""
Simple LIME version of Saliency.py
---------------------------------
For every image in a target folder this script
1. Predicts the top‐1 class with a pre-trained ResNet-50
2. Uses LIME to explain that prediction
3. Converts the LIME explanation to a saliency map where each super-pixel is
   filled with a solid colour whose intensity is proportional to the weight
   assigned by LIME (positive -> red / high value, negative -> blue / low value).
4. Saves the raw saliency map (.npy), a visualisation (.png) **and** collects
   all maps in a big array for later quantitative evaluation – mirroring the
   behaviour of Saliency.py and Saliency_Gradcam.py.

Important implementation choices
--------------------------------
• The model is identical to Saliency.py → ``ResNet-50 → Softmax``.  Gradients
  are **not** required by LIME, therefore all parameters are frozen and the
  model is wrapped in ``nn.DataParallel`` to utilise multiple GPUs if
  available.
• ``lime_image.LimeImageExplainer`` is used with a Quick-Shift segmentation.
  The ``batch_predict`` helper converts NumPy images (float32 0-255) coming
  from LIME to normalised Torch tensors expected by the model.
• The explanation returned by LIME is a list ``[(segment_id, weight), …]`` –
  we create an empty map ``(H, W)`` and fill every super-pixel region with the
  corresponding weight.  That produces a piece-wise constant saliency map in
  which *all* pixels of the same region have the *exact same* value, as
  requested.
• Finally the map is min-max normalised to ``[0, 1]`` for convenient display
  and saving.

Usage
-----
Adjust ``args.datadir`` to point to the image directory.  Then simply run

    python Saliency_Lime.py

The results will be stored in ``saliency_lime/`` next to the script.
"""

import os, glob, warnings
from typing import Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets.folder as folder

from lime import lime_image
from skimage.segmentation import quickshift

from utils import *  # provides preprocess, tensor_imshow, get_class_name, Dummy

# ---------------------------------------------------------------------------
cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)  # suppress skimage warnings

# Accept *.JPEG etc. ---------------------------------------------------------
folder.IMG_EXTENSIONS += tuple(e.upper() for e in folder.IMG_EXTENSIONS)

# ---------------------------------------------------------------------------
# Args (keep Dummy pattern used in the other scripts) ------------------------
args = Dummy()
args.workers      = 2
args.datadir      = '/beegfs/data/shared/imagenet/imagenet100/val/'  # change as needed
args.range        = None                # process all images
args.input_size   = (224, 224)
args.gpu_batch    = 200                 # only used inside batch_predict
out_dir           = 'saliency_lime'     # output folder

# ---------------------------------------------------------------------------
# Dataset identical to Saliency.py ------------------------------------------
class CustomImageDataset(torch.utils.data.Dataset):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.files = []
        for e in self.exts:
            self.files.extend(glob.glob(os.path.join(root_dir, '**', e), recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f'No images found in {root_dir}')
        print(f'Found {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        img   = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0, path

preprocess_tf = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset     = CustomImageDataset(args.datadir, preprocess_tf)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)

# ---------------------------------------------------------------------------
# Model (same as Saliency.py) ------------------------------------------------
backbone = models.resnet50(pretrained=True)
model = nn.Sequential(backbone, nn.Softmax(dim=1)).cuda().eval()
for p in model.parameters():
    p.requires_grad = False
model = nn.DataParallel(model)

# ---------------------------------------------------------------------------
# Helper for LIME ------------------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def normalize(t: torch.Tensor) -> torch.Tensor:
    """Denormalise a Tensor to uint8 image (H, W, C)."""
    np_img = t.cpu().numpy().transpose(1, 2, 0)
    np_img = std * np_img + mean
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return np_img

def batch_predict(images: np.ndarray) -> np.ndarray:
    """LIME callback: predict probabilities for a list/array of images.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W, 3), uint8 0-255.
    Returns
    -------
    np.ndarray
        Predicted probabilities (N, 1000).
    """
    model.eval()
    batch = []
    for img in images:
        img = Image.fromarray(img)
        batch.append(preprocess_tf(img))
    batch = torch.stack(batch, dim=0).cuda(non_blocking=True)
    with torch.no_grad():
        preds = model(batch)
    return preds.cpu().numpy()

# LIME explainer – will keep segmentation constant for reproducibility -------
explainer = lime_image.LimeImageExplainer(verbose=False)

segmentation_fn = lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)

# ---------------------------------------------------------------------------
# Explanation loop -----------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
explanations = np.empty((len(data_loader), *args.input_size), dtype=np.float32)

for i, (img_cpu, _, path_tensor) in enumerate(tqdm(data_loader, desc='Explaining images')):
    img_gpu = img_cpu.cuda(non_blocking=True)

    # -----------------------------------------------------------------------
    # 1. Predict label -------------------------------------------------------
    with torch.no_grad():
        preds = model(img_gpu)
        c = preds.argmax(dim=1).item()  # predicted class id

    # -----------------------------------------------------------------------
    # 2. Prepare image for LIME (uint8 numpy) -------------------------------
    orig_img_uint8 = normalize(img_cpu[0])  # (H, W, C) uint8

    # -----------------------------------------------------------------------
    # 3. Run LIME -----------------------------------------------------------
    explanation = explainer.explain_instance(
        orig_img_uint8,
        batch_predict,
        labels=(c,),
        top_labels=None,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmentation_fn,
    )

    # LIME stores weights per super-pixel -----------------------------------
    superpixel_weights = dict(explanation.local_exp[c])  # {seg_id: weight}
    mask = explanation.segments  # (H, W) segment id per pixel

    saliency = np.zeros(mask.shape, dtype=np.float32)
    for seg_id, weight in superpixel_weights.items():
        saliency[mask == seg_id] = weight

    # Normalise to [0, 1] ----------------------------------------------------
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    # max_abs = np.abs(saliency).max()
    # if max_abs > 0:
    #     saliency /= max_abs

    explanations[i] = saliency

    name = os.path.splitext(os.path.basename(path_tensor[0]))[0]

    # 4a. Save raw saliency map --------------------------------------------
    np.save(os.path.join(out_dir, f'{name}_lime.npy'), saliency)

    # 4b. Composite overlay PNG --------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.axis('off')
    tensor_imshow(img_cpu[0]); plt.title(get_class_name(c))
    plt.subplot(122); plt.axis('off')
    tensor_imshow(img_cpu[0])
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.savefig(os.path.join(out_dir, f'{name}_lime.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Save big array for later evaluation (optional) -----------------------------
np.save(os.path.join(out_dir, 'all_lime_maps.npy'), explanations)
print(f'Done! Saved LIME maps to {out_dir}/')
