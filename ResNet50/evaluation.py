from torch import nn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms  # Add this import
try:
    from .utils import *  # noqa: F401,F403
except ImportError:
    # Fallback if the module is executed as a standalone script
    from utils import *  # type: ignore  # noqa: F401,F403
from explanations import RISE

# -----------------------------------------------------------------------------
#  This is a near-verbatim copy of `RISE_original/evaluation.py` with only the
#  constants updated so that it works with ESC-50 spectrogram "images".
# -----------------------------------------------------------------------------

HW = 224 * 224  # spectrogram area after resize (same as original)
N_CLASSES = 50  # ESC-50 has 50 categories


def gkern(klen, nsig):
    """Returns a (3×3×k×k) Gaussian kernel for image blurring."""
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype("float32"))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric:
    def __init__(self, model, mode, step, substrate_fn):
        assert mode in ["del", "ins"]
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        pred = self.model(img_tensor.cuda())
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == "del":
            title = "Deletion game"
            ylabel = "Pixels deleted"
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == "ins":
            title = "Insertion game"
            ylabel = "Pixels inserted"
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps + 1):
            pred = self.model(start.cuda())
            pr, cl = torch.topk(pred, 1)
            if verbose == 2:
                print(f"{get_class_name(cl[0][0])}: {float(pr[0][0]):.3f}")
                print(f"{get_class_name(cl[0][1])}: {float(pr[0][1]):.3f}")
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title(f"{ylabel} {100 * i / n_steps:.1f}%, P={scores[i]:.4f}")
                plt.axis("off")
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[: i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[: i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + f"/{i:03d}.png")
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i : self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = (
                    finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
                )
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, N_CLASSES)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc="Predicting labels"):
            preds = self.model(img_batch[i * batch_size : (i + 1) * batch_size].cuda()).cpu()
            predictions[i * batch_size : (i + 1) * batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc="Substrate"):
            substrate[j * batch_size : (j + 1) * batch_size] = self.substrate_fn(
                img_batch[j * batch_size : (j + 1) * batch_size]
            )

        if self.mode == "del":
            caption = "Deleting  "
            start = img_batch.clone()
            finish = substrate
        elif self.mode == "ins":
            caption = "Inserting "
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps + 1), desc=caption + "pixels"):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j * batch_size : (j + 1) * batch_size].cuda())
                preds = preds.cpu().numpy()[
                    range(batch_size),
                    top[j * batch_size : (j + 1) * batch_size],
                ]
                scores[i, j * batch_size : (j + 1) * batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i : self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = (
                finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
            )
        print(f"AUC: {auc(scores.mean(1))}")
        return scores 

# -----------------------------------------------------------------------------
# Custom dataset for loading spectrogram images directly
# -----------------------------------------------------------------------------

class SpectrogramImageDataset(torch.utils.data.Dataset):
    """Dataset that loads spectrogram images directly from PNG files."""
    
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform or self._default_transform()
        
    def _default_transform(self):
        """Default transform for spectrogram images."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = image.resize((224, 224), Image.BILINEAR)  # Resize to 224x224
        
        if self.transform:
            tensor = self.transform(image)
        
        # Return dummy label (0) since we don't have class labels from filenames
        # The model will predict the actual class
        return tensor, 0

# -----------------------------------------------------------------------------
# If executed as a script, run evaluation on specific files from ESC50_spectrograms_images
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import random, os
    from pathlib import Path
    import torchvision.models as models
    import pandas as pd
    from PIL import Image
    import glob

    # ----------------------------------------------------------------------
    # Load spectrogram images directly from ESC50_spectrograms_images
    # ----------------------------------------------------------------------
    spectrograms_dir = "ESC50_spectrograms_images"
    image_files = glob.glob(os.path.join(spectrograms_dir, "*.png"))
    
    print(f"Found {len(image_files)} spectrogram images")
    
    # Create dataset from image files directly
    image_dataset = SpectrogramImageDataset(image_files)

    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    INPUT_SIZE = (224, 224)
    GPU_BATCH = 100

    # Ensure masks exist once
    masks_path = os.path.join(os.path.dirname(__file__), "masks.npy")
    if not os.path.isfile(masks_path):
        _tmp_exp = RISE(nn.Identity(), INPUT_SIZE, GPU_BATCH)
        _tmp_exp.generate_masks(N=6000, s=8, p1=0.1, savepath=masks_path)

    # Prepare substrate fns
    klen, ksig = 11, 5
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern.to(x.device), padding=klen // 2)

    for fold in range(1, 6):
        ckpt = f"resnet50_esc50_fold{fold}.pt"
        print(f"\n=== Evaluating fold {fold} ({ckpt}) ===")

        # 1️⃣  Load model
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 50)
        state = torch.load(ckpt, map_location="cuda")
        if next(iter(state)).startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state)
        model = nn.Sequential(net, nn.Softmax(dim=1)).cuda().eval()
        model = nn.DataParallel(model)

        # 2️⃣  Explainer + causal metrics
        expl = RISE(model, INPUT_SIZE, GPU_BATCH)
        expl.load_masks(masks_path)
        ins_metric = CausalMetric(model, "ins", 224, substrate_fn=blur)
        del_metric = CausalMetric(model, "del", 224, substrate_fn=torch.zeros_like)

        # 3️⃣  Evaluate subset
        out_dir = Path("evaluation_graphs") / f"fold_{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_records = []

        for local_i, (spec, _) in enumerate(tqdm(loader, desc=f"Fold {fold}")):
            with torch.no_grad():
                sal = expl(spec.cuda())
                prob, cls_id = torch.max(model(spec.cuda()), dim=1)
            cls_id = cls_id.item()
            sal_map = sal[cls_id].cpu().numpy()

            # deletion / insertion curves & AUC
            with torch.no_grad():
                del_curve = del_metric.single_run(spec, sal_map, verbose=0)
                ins_curve = ins_metric.single_run(spec, sal_map, verbose=0)
            auc_del = auc(del_curve)
            auc_ins = auc(ins_curve)

            # Get filename for this sample
            img_file = image_files[local_i]
            base_name = Path(img_file).stem
            csv_records.append({
                "sample": base_name,
                "class": get_class_name(cls_id),
                "auc_deletion": auc_del,
                "auc_insertion": auc_ins,
            })

            # Save comprehensive visualization for each sample
            plt.figure(figsize=(20, 5))
            
            # Original spectrogram
            plt.subplot(141)
            plt.axis('off')
            plt.title(f'Sample {local_i}: {get_class_name(cls_id)} ({100*prob.item():.1f}%)', fontsize=12)
            tensor_imshow(spec[0])
            
            # Saliency overlay
            plt.subplot(142)
            plt.axis('off')
            plt.title(f'Saliency Overlay', fontsize=12)
            tensor_imshow(spec[0])
            plt.imshow(sal_map, alpha=0.5, cmap='jet')
            
            # Deletion curve
            plt.subplot(143)
            steps = np.arange(len(del_curve)) / (len(del_curve) - 1)
            plt.plot(steps, del_curve, 'r-', linewidth=2, label='Deletion')
            plt.fill_between(steps, 0, del_curve, alpha=0.3, color='red')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Fraction of pixels deleted', fontsize=10)
            plt.ylabel('Model confidence', fontsize=10)
            plt.title(f'Deletion AUC: {auc_del:.3f}', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Insertion curve
            plt.subplot(144)
            steps = np.arange(len(ins_curve)) / (len(ins_curve) - 1)
            plt.plot(steps, ins_curve, 'g-', linewidth=2, label='Insertion')
            plt.fill_between(steps, 0, ins_curve, alpha=0.3, color='green')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Fraction of pixels inserted', fontsize=10)
            plt.ylabel('Model confidence', fontsize=10)
            plt.title(f'Insertion AUC: {auc_ins:.3f}', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{local_i:03d}_{base_name}_detailed.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Save CSV summary per fold
        pd.DataFrame(csv_records).to_csv(out_dir / "detailed_results.csv", index=False)
        
        # Create fold-level summary visualization
        del_aucs = [r["auc_deletion"] for r in csv_records]
        ins_aucs = [r["auc_insertion"] for r in csv_records]
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: AUC distribution
        plt.subplot(2, 3, 1)
        plt.hist(del_aucs, bins=10, alpha=0.7, label='Deletion AUC', color='red')
        plt.hist(ins_aucs, bins=10, alpha=0.7, label='Insertion AUC', color='green')
        plt.xlabel('AUC Score')
        plt.ylabel('Frequency')
        plt.title('AUC Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        plt.subplot(2, 3, 2)
        plt.scatter(del_aucs, ins_aucs, alpha=0.7, c='blue')
        plt.xlabel('Deletion AUC')
        plt.ylabel('Insertion AUC')
        plt.title('Deletion vs Insertion AUC')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Sample scores over time
        plt.subplot(2, 3, 3)
        sample_indices = list(range(len(del_aucs)))
        plt.plot(sample_indices, del_aucs, 'r-', marker='o', markersize=3, label='Deletion')
        plt.plot(sample_indices, ins_aucs, 'g-', marker='s', markersize=3, label='Insertion')
        plt.xlabel('Sample Index')
        plt.ylabel('AUC Score')
        plt.title('AUC Scores by Sample')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Box plot
        plt.subplot(2, 3, 4)
        plt.boxplot([del_aucs, ins_aucs], labels=['Deletion', 'Insertion'])
        plt.ylabel('AUC Score')
        plt.title('AUC Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Class distribution (if multiple classes)
        plt.subplot(2, 3, 5)
        classes = [r["class"] for r in csv_records]
        unique_classes = list(set(classes))
        if len(unique_classes) > 1:
            class_counts = {c: classes.count(c) for c in unique_classes}
            plt.bar(range(len(unique_classes)), class_counts.values())
            plt.xticks(range(len(unique_classes)), [c[:10] for c in unique_classes], rotation=45)
            plt.ylabel('Count')
            plt.title('Class Distribution')
        else:
            plt.text(0.5, 0.5, f'Single class: {unique_classes[0]}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Single Class Dataset')
        
        # Plot 6: Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = f"""
            EVALUATION SUMMARY - FOLD {fold}
            
            Samples: {len(csv_records)}
            Model: ResNet-50 (ESC-50)
            
            DELETION AUC:
            Mean: {np.mean(del_aucs):.3f}
            Std:  {np.std(del_aucs):.3f}
            Min:  {np.min(del_aucs):.3f}
            Max:  {np.max(del_aucs):.3f}
            
            INSERTION AUC:
            Mean: {np.mean(ins_aucs):.3f}
            Std:  {np.std(ins_aucs):.3f}
            Min:  {np.min(ins_aucs):.3f}
            Max:  {np.max(ins_aucs):.3f}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(out_dir / "fold_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved evaluation curves & CSV for fold {fold} in {out_dir}") 