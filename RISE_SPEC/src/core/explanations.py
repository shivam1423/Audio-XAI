import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal


# TF-structured mask generator
class TFStructuredRISE(RISE):
    """RISE-style explainer but with time–frequency structured masks.

    Generates a bank of binary/soft masks reflecting typical audio invariances:
    - Time stripes (vertical bands)
    - Frequency bands (horizontal bands)
    - Axis-aligned rectangular patches
    - Mel-band masks (bands aligned to mel-scaled frequency bins)
    """

    def generate_tf_masks(
        self,
        N: int,
        time_stripe_frac: float = 0.25,
        freq_band_frac: float = 0.25,
        rect_patch_frac: float = 0.25,
        mel_band_frac: float = 0.25,
        time_stripe_width_px: tuple = (4, 24),
        freq_band_height_px: tuple = (4, 24),
        rect_size_px: tuple = (8, 48),
        rect_count_range: tuple = (1, 6),
        stripe_count_range: tuple = (1, 12),
        mel_bands: int = 64,
        band_keep_prob: float = 0.3,
        soften_edges: bool = True,
        edge_sigma_px: float = 1.0,
        savepath: str = 'masks_tf.npy',
    ) -> None:
        H, W = self.input_size
        # Determine counts for each strategy
        n_time = int(N * time_stripe_frac)
        n_freq = int(N * freq_band_frac)
        n_rect = int(N * rect_patch_frac)
        n_mel = N - n_time - n_freq - n_rect

        def rand_int(a: int, b: int) -> int:
            return int(np.random.randint(a, b + 1))

        masks = []

        # Time stripes: choose several vertical stripes
        for _ in range(n_time):
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*stripe_count_range)
            for _ in range(k):
                width = rand_int(*time_stripe_width_px)
                x0 = rand_int(0, W - 1)
                x1 = min(W, x0 + width)
                m[:, x0:x1] = 1.0
            if soften_edges:
                m = self._soften_mask(m, edge_sigma_px)
            masks.append(m)

        # Frequency bands: choose several horizontal bands
        for _ in range(n_freq):
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*stripe_count_range)
            for _ in range(k):
                height = rand_int(*freq_band_height_px)
                y0 = rand_int(0, H - 1)
                y1 = min(H, y0 + height)
                m[y0:y1, :] = 1.0
            if soften_edges:
                m = self._soften_mask(m, edge_sigma_px)
            masks.append(m)

        # Rectangular TF patches: several rectangles aligned to axes
        for _ in range(n_rect):
            m = np.zeros((H, W), dtype=np.float32)
            k = rand_int(*rect_count_range)
            for _ in range(k):
                rh = rand_int(*rect_size_px)
                rw = rand_int(*rect_size_px)
                y0 = rand_int(0, max(0, H - rh))
                x0 = rand_int(0, max(0, W - rw))
                m[y0:y0 + rh, x0:x0 + rw] = 1.0
            if soften_edges:
                m = self._soften_mask(m, edge_sigma_px)
            masks.append(m)

        # Mel-band masks: select mel bands to keep
        # Build mel edges along frequency axis (H direction)
        mel_edges = self._mel_edges(num_mel=mel_bands, fmin=0.0, fmax=8000.0)  # synthetic range
        # Map mel edges into image rows [0, H)
        mel_rows = np.unique(np.clip(np.round(mel_edges / mel_edges.max() * (H - 1)).astype(int), 0, H - 1))
        # Ensure bins
        if len(mel_rows) < 2:
            mel_rows = np.array([0, H - 1])
        for _ in range(n_mel):
            m = np.zeros((H, W), dtype=np.float32)
            # iterate bands and randomly keep some with band_keep_prob
            for i in range(len(mel_rows) - 1):
                y0 = int(mel_rows[i])
                y1 = int(mel_rows[i + 1]) + 1
                if np.random.rand() < band_keep_prob:
                    m[y0:y1, :] = 1.0
            if soften_edges:
                m = self._soften_mask(m, edge_sigma_px)
            masks.append(m)

        masks = np.stack(masks, axis=0)  # [N, H, W]
        masks = masks.reshape(-1, 1, H, W)
        np.save(savepath, masks)
        self.masks = torch.from_numpy(masks).float().cuda()
        self.N = self.masks.shape[0]
        # Use average keep probability for normalization like RISE
        self.p1 = float(self.masks.mean().item())

    @staticmethod
    def _soften_mask(mask2d: np.ndarray, sigma_px: float) -> np.ndarray:
        # Light gaussian blur on 0/1 boundaries to reduce artifacts
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(mask2d.astype(np.float32), sigma=sigma_px)
        except Exception:
            return mask2d.astype(np.float32)

    @staticmethod
    def _mel_edges(num_mel: int, fmin: float, fmax: float) -> np.ndarray:
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        mmin = hz_to_mel(fmin)
        mmax = hz_to_mel(fmax)
        mel_points = np.linspace(mmin, mmax, num_mel + 1)
        return mel_to_hz(mel_points)

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
