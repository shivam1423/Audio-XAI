#!/usr/bin/env python
# coding: utf-8
#  Simple Grad-CAM version of Saliency.py
#  ─────────────────────────────────────

import os, glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.datasets.folder as folder
from utils import *
# --- pytorch-grad-cam ------------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# ──────────────────────────────────────
cudnn.benchmark = True

# Accept *.JPEG etc.
folder.IMG_EXTENSIONS += tuple(e.upper() for e in folder.IMG_EXTENSIONS)

# ──────────────────────────────────────
# Args (keep Dummy pattern)
args = Dummy()
args.workers      = 2
args.datadir      = '/beegfs/data/shared/imagenet/imagenet100/val/'      # change as needed
args.range        = None                # process all images
args.input_size   = (224, 224)
args.gpu_batch    = 250
out_dir           = 'saliency_gradcam'  # output folder

# ──────────────────────────────────────
# Dataset identical to Saliency.py
class CustomImageDataset(torch.utils.data.Dataset):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = []
        for e in self.exts:
            self.files.extend(glob.glob(os.path.join(root_dir, '**', e), recursive=True))
        print(f'Found {len(self.files)} images')

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path  = self.files[idx]
        img   = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, 0, path

preprocess = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])
dataset     = CustomImageDataset(args.datadir, preprocess)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)

# ──────────────────────────────────────
# Model (keep single-GPU object for grad-cam)
backbone = models.resnet50(True).cuda().eval()
# We do NOT append Softmax and we do NOT freeze params – gradients must flow.
target_layers = [backbone.layer4[-1]]
gc = GradCAM(model=backbone, target_layers=target_layers, use_cuda=True)

# ──────────────────────────────────────
# Explanation loop (mirrors Saliency.py)
os.makedirs(out_dir, exist_ok=True)
explanations = np.empty((len(data_loader), *args.input_size))

for i, (img_cpu, _, path) in enumerate(tqdm(data_loader, desc='Explaining images')):
    img_gpu = img_cpu.cuda().requires_grad_(True)

    # get predicted class id
    with torch.no_grad():
        preds = backbone(img_gpu)
        c = preds.argmax(dim=1).item()

    grayscale_cam = gc(input_tensor=img_gpu,
                       targets=[ClassifierOutputTarget(c)])[0]   # (224,224)
    cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max()+1e-8)

    explanations[i] = cam
    name = os.path.splitext(os.path.basename(path[0]))[0]

    # save raw CAM
    np.save(os.path.join(out_dir, f'{name}_gradcam.npy'), cam)

    # Composite overlay PNG
    plt.figure(figsize=(10,5))
    plt.subplot(121); plt.axis('off')
    tensor_imshow(img_cpu[0]); plt.title(get_class_name(c))
    plt.subplot(122); plt.axis('off')
    tensor_imshow(img_cpu[0])
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.savefig(os.path.join(out_dir, f'{name}_gradcam.png'),
                dpi=150,bbox_inches='tight')
    plt.close()

# Save big array for later evaluation (optional)
np.save(os.path.join(out_dir,'all_gradcams.npy'), explanations)
print(f'Done! Saved Grad-CAM maps to {out_dir}/') 