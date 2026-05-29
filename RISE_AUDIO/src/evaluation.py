import numpy as np
import torch
import torch.nn as nn
import os
import glob
from tqdm import tqdm
from typing import List
from scipy.ndimage import gaussian_filter
from src.audio_io import AudioProcessor
from src.feature_extractor import process_masked_audio_list
from src.models.resnet50 import ResNetModel
from src.models.base import get_model_type, ModelInputType

HW = 224 * 224  # image area
n_classes = 1000

def gkern(klen, nsig, channels):
    """Returns a Gaussian kernel array for blurring."""
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig).astype(np.float32)
    # kern = np.zeros((3, 3, klen, klen))
    kern = np.zeros((channels, channels, klen, klen), dtype=np.float32)
    for i in range(channels):
        kern[i, i] = k
    return torch.from_numpy(kern)

def blur_spec(spec_2ft: torch.Tensor, klen: int = 11, nsig: float = 5.0) -> torch.Tensor:
    # spec_2ft: (2, F, T) or (C, F, T)
    assert spec_2ft.dim() == 3, "Expected (C, F, T)"
    C = spec_2ft.shape[0]
    x = spec_2ft.unsqueeze(0)  # (1, C, F, T)
    kernel = gkern(klen, nsig, C)
    # Match device and dtype of input
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    pad = klen // 2
    y = torch.nn.functional.conv2d(x, kernel, padding=pad)  # (1, C, F, T)
    return y.squeeze(0)

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric:
    def __init__(self, model, mode, step, substrate_fn):
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, orig_mel_img, real_imag_spec, explanation, verbose=0, audio_processor=None, n_samples=None, steps=224, save_dir=None):
        """Run metric on one audio-saliency pair using STFT-domain editing with model-aware scoring."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from src.audio_io import AudioProcessor
        from src.feature_extractor import process_masked_audio_list
        import torchvision.transforms as transforms

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # Ensure tensors are on the right device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        real_imag_spec = real_imag_spec.to(device)

        # Build transform to convert PIL -> model-ready tensor (for spectrogram models)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ===== DETECT MODEL TYPE AND GET TARGET CLASS =====
        model_type = get_model_type(self.model)
        ap = audio_processor or AudioProcessor()
        
        if model_type == ModelInputType.RAW_AUDIO:
            # For raw audio models (wav2vec2), get prediction from original audio
            print("Detected RAW_AUDIO model (wav2vec2)")
            import torchaudio.transforms as T
            
            # Reconstruct audio from spectrogram
            complex_spec = ap.real_imag_to_complex(real_imag_spec.cpu())
            n_samples_est = (real_imag_spec.shape[2] - 1) * ap.hop_length
            audio = ap.spectrogram_to_audio(complex_spec, int(n_samples_est))
            
            # Resample to model's sample rate
            model_sr = getattr(self.model, 'sample_rate', 16000)
            if ap.target_sr != model_sr:
                resampler = T.Resample(orig_freq=ap.target_sr, new_freq=model_sr)
                audio = resampler(audio)
            
            # Pad or truncate to model's expected length (for ACDNet)
            model_input_length = getattr(self.model, 'input_length', None)
            if model_input_length is not None:
                from src.saliency import pad_or_truncate_waveform
                audio = pad_or_truncate_waveform(audio, model_input_length)
            
            audio = audio.to(device)
            with torch.no_grad():
                logits, probs = self.model.predict(audio)
            c = torch.argmax(probs, dim=1).item()
        else:
            # For spectrogram models (ResNet, AST), use provided mel image
            print("Detected SPECTROGRAM model (ResNet/AST)")
            if orig_mel_img is None:
                raise ValueError("orig_mel_img must be provided for spectrogram models.")
            original_tensor = img_transform(orig_mel_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, probs = self.model.predict(original_tensor)
            c = torch.argmax(probs, dim=1).item()
        
        print(f"\nTarget class: {c}, Probability: {probs[0,c]:.4f}\n")
        # Setup sizes
        F_bins, T_frames = real_imag_spec.shape[1], real_imag_spec.shape[2]
        # print("\n\nshape of F_bin and t_frames are ",F_bins,T_frames)
        # Resize explanation to (F, T) if needed
        if isinstance(explanation, torch.Tensor):
            exp = explanation.detach().cpu().numpy()
        else:
            exp = np.asarray(explanation)
        # print("exp shape is ",exp.shape)
        if exp.shape != (F_bins, T_frames):
            exp_t = torch.from_numpy(exp).float().unsqueeze(0).unsqueeze(0)
            exp_t = F.interpolate(exp_t, size=(F_bins, T_frames), mode='bilinear', align_corners=False)
            exp = exp_t.squeeze().cpu().numpy()

        # # Saliency order (descending importance) - make a copy to avoid negative strides
        flat = exp.reshape(-1).copy()  # Make a copy to avoid negative strides
        order = np.argsort(flat)[::-1]
        # order = np.flip(np.argsort(exp.reshape(-1, HW), axis=1), axis=-1)
        # print("order is ",order)

        total_bins = F_bins * T_frames
        steps = int(steps)
        scores = np.empty(steps + 1, dtype=np.float32)

        # Compute n_samples if not provided
        if n_samples is None:
            est = (T_frames - 1) * ap.hop_length
            n_samples = int(est)

        # Prepare originals
        orig_spec = real_imag_spec.clone()
        zeros_spec = torch.zeros_like(orig_spec)

        # ===== MODEL-TYPE-AWARE SCORING FUNCTION =====
        def score_spec(spec_2ft: torch.Tensor, step: int = None) -> float:
            """Score a spectrogram by reconstructing audio and feeding to model."""
            with torch.no_grad():
                # Move to CPU for audio reconstruction
                spec_2ft_cpu = spec_2ft.cpu()

                # Ensure correct shape (2, F, T)
                if spec_2ft_cpu.dim() == 3 and spec_2ft_cpu.shape[0] == 2:
                    complex_spec = ap.real_imag_to_complex(spec_2ft_cpu)
                else:
                    spec_2ft_cpu = spec_2ft_cpu.view(2, F_bins, T_frames)
                    complex_spec = ap.real_imag_to_complex(spec_2ft_cpu)

                if complex_spec.numel() == 0:
                    print("Warning: Empty complex spectrogram, returning zero score")
                    return 0.0
                    
                # Reconstruct audio
                audio = ap.spectrogram_to_audio(complex_spec, n_samples)
                
                # ===== ROUTE BASED ON MODEL TYPE =====
                if model_type == ModelInputType.RAW_AUDIO:
                    # RAW AUDIO PIPELINE (wav2vec2, ACDNet)
                    import torchaudio.transforms as T
                    model_sr = getattr(self.model, 'sample_rate', 16000)
                    
                    # Resample if needed
                    if ap.target_sr != model_sr:
                        resampler = T.Resample(orig_freq=ap.target_sr, new_freq=model_sr)
                        audio_resampled = resampler(audio)
                    else:
                        audio_resampled = audio
                    
                    # Pad or truncate to model's expected length (for ACDNet)
                    model_input_length = getattr(self.model, 'input_length', None)
                    if model_input_length is not None:
                        from src.saliency import pad_or_truncate_waveform
                        audio_resampled = pad_or_truncate_waveform(audio_resampled, model_input_length)
                    
                    # Move to device and get prediction
                    audio_tensor = audio_resampled.to(device)
                    
                    if hasattr(self.model, 'predict'):
                        logits, probs = self.model.predict(audio_tensor)
                    else:
                        logits = self.model(audio_tensor)
                        probs = torch.softmax(logits, dim=1)
                    
                    return probs[0, c].item()
                
                else:
                    # SPECTROGRAM PIPELINE (ResNet, AST)
                    model_sr = getattr(self.model, 'sample_rate', 22050)
                    
                    # Convert to mel spectrogram
                    mel_img = process_masked_audio_list(
                        [audio],
                        sr=model_sr,
                        n_fft=1024,
                        hop_length=512,
                        n_mels=128,
                        orig_sr=ap.target_sr
                    )[0]
                    
                    if save_dir is not None and step is not None:
                        mel_img.save(os.path.join(save_dir, f"{self.mode}_step_{step:03d}.png"))

                    # Convert to tensor and move to device
                    img_tensor = img_transform(mel_img).unsqueeze(0).to(device)
                    
                    if hasattr(self.model, 'predict'):
                        logits, probs = self.model.predict(img_tensor)
                    else:
                        logits = self.model(img_tensor)
                        probs = torch.softmax(logits, dim=1)
                    
                    return probs[0, c].item()

        # Build substrate (baseline) spectrogram
        if callable(self.substrate_fn):
            sub = self.substrate_fn(orig_spec)
            if isinstance(sub, np.ndarray):
                substrate_spec = torch.from_numpy(sub).to(orig_spec.device).type_as(orig_spec)
            else:
                substrate_spec = sub.to(orig_spec.device).type_as(orig_spec)
        else:
            substrate_spec = zeros_spec.clone()

        # Start state per mode
        if self.mode == 'del':
            current_spec = orig_spec.clone()
        else:  # 'ins'
            current_spec = substrate_spec.clone()
        scores[0] = score_spec(current_spec, step=0)

        # Progressive modification
        for i in range(1, steps + 1):
            k = min((i * total_bins) // steps, total_bins)
            idx = order[:k]

            # Build a binary mask of selected bins with shape (F,T)
            bin_mask = torch.zeros((F_bins, T_frames), device=device, dtype=torch.bool)
            # Make a copy of idx to avoid negative strides issue
            idx_copy = idx.copy()
            bin_mask.view(-1)[torch.from_numpy(idx_copy).to(device)] = True

            if self.mode == 'del':
                # Replace selected bins with substrate (not zeros)
                current_spec[:, bin_mask] = substrate_spec[:, bin_mask]
            else:
                # Copy selected bins from original into substrate baseline
                current_spec[:, bin_mask] = orig_spec[:, bin_mask]

            scores[i] = score_spec(current_spec, step=i)

            if verbose:
                print(f"{self.mode} step {i}/{steps}: score={scores[i]:.4f}")

        return scores

def find_audio_files(audio_dir: str) -> List[str]:
    """Find all audio files in directory."""
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return audio_files

def evaluate_audio_saliency_maps(audio_dir, saliency_dir, output_dir, model, steps=224, verbose=0, ins_dir=None, del_dir=None):
    """Evaluate audio saliency maps using insertion/deletion metrics.
    
    Args:
        audio_dir: Directory containing audio files
        saliency_dir: Directory containing saliency maps
        output_dir: Output directory for results
        model: Model instance for evaluation
        steps: Number of evaluation steps
        verbose: Verbosity level
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio files
    audio_files = find_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Results storage
    insertion_scores = []
    deletion_scores = []
    audio_names = []
    
    print("Starting evaluation...")
    for audio_path in tqdm(audio_files, desc="Evaluating audio files"):
        try:
            # Get audio filename without extension
            name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]

            # Load audio and create original mel image
            orig_audio, orig_sr = audio_processor.load_audio(audio_path)
            print(orig_sr)
            orig_mel_img = process_masked_audio_list(
                [orig_audio],
                sr=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                orig_sr=orig_sr
            )[0]

            # Find corresponding saliency map
            saliency_candidates = [
                os.path.join(saliency_dir, f"{os.path.basename(audio_path)}_saliency.npy"),
                os.path.join(saliency_dir, f"{os.path.basename(audio_path)}.npz"),
                os.path.join(saliency_dir, f"{name_no_ext}_saliency.npy"),
            ]

            saliency_path = None
            for candidate in saliency_candidates:
                if os.path.exists(candidate):
                    saliency_path = candidate
                    break

            if saliency_path is None:
                print(f"Warning: No saliency map found for {audio_path}")
                continue

            # Load saliency map
            if saliency_path.endswith('.npz'):
                data = np.load(saliency_path)
                saliency_map = data['saliency_map']
            else:
                saliency_map = np.load(saliency_path)

            # Get spectrogram for evaluation
            spec = audio_processor.audio_to_spectrogram(orig_audio)
            real_imag_spec = audio_processor.complex_to_real_imag(spec)


            # Initialize metrics
            insertion_metric = CausalMetric(model, 'ins', 224, substrate_fn=blur_spec)
            deletion_metric = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

            # Run evaluation
            ins_score = insertion_metric.single_run(
                orig_mel_img, real_imag_spec, saliency_map,
                verbose=verbose, audio_processor=audio_processor,
                n_samples=orig_audio.shape[1], steps=steps, save_dir=ins_dir
            )
            del_score = deletion_metric.single_run(
                orig_mel_img, real_imag_spec, saliency_map,
                verbose=verbose, audio_processor=audio_processor,
                n_samples=orig_audio.shape[1], steps=steps, save_dir=del_dir
            )

            # Calculate AUC
            insertion_auc = auc(ins_score)
            deletion_auc = auc(del_score)

            # Store results
            insertion_scores.append(insertion_auc)
            deletion_scores.append(deletion_auc)
            audio_names.append(name_no_ext)

            print(f"{name_no_ext}: Insertion AUC = {insertion_auc:.4f}, Deletion AUC = {deletion_auc:.4f}")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Calculate and save results
    if insertion_scores and deletion_scores:
        mean_insertion_auc = np.mean(insertion_scores)
        mean_deletion_auc = np.mean(deletion_scores)
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Mean Insertion AUC: {mean_insertion_auc:.4f}")
        print(f"Mean Deletion AUC: {mean_deletion_auc:.4f}")
        print(f"Number of audio files evaluated: {len(insertion_scores)}")
        
        # Save results
        results = {
            'audio_names': audio_names,
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'mean_insertion_auc': mean_insertion_auc,
            'mean_deletion_auc': mean_deletion_auc,
            'total_audio_files': len(insertion_scores)
        }
        
        np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
        
        # Save as text file
        with open(os.path.join(output_dir, 'evaluation_summary_rise.txt'), 'w') as f:
            f.write("=== AUDIO SALIENCY MAP EVALUATION ===\n\n")
            f.write(f"Mean Insertion AUC: {mean_insertion_auc:.4f}\n")
            f.write(f"Mean Deletion AUC: {mean_deletion_auc:.4f}\n")
            f.write(f"Number of audio files evaluated: {len(insertion_scores)}\n\n")
            f.write("Detailed Results:\n")
            f.write("Audio Name\tInsertion AUC\tDeletion AUC\n")
            f.write("-" * 50 + "\n")
            for name, ins_auc, del_auc in zip(audio_names, insertion_scores, deletion_scores):
                f.write(f"{name}\t{ins_auc:.4f}\t{del_auc:.4f}\n")
        
        print(f"\nResults saved to {output_dir}/")
        
    else:
        print("No valid audio files found for evaluation.")