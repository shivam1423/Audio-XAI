from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from rise_utils import *

HW = 224 * 224 # image area
n_classes = 1000

def gkern(klen, nsig, num_channels=3):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring.
    
    Args:
        klen: Kernel size
        nsig: Gaussian sigma
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((num_channels, num_channels, klen, klen))
    for i in range(num_channels):
        kern[i, i] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, input_size=None):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            input_size (tuple): Input dimensions (H, W) for dynamic size support.
                                If None, defaults to (224, 224) for backward compatibility.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        
        # Dynamic input size support
        if input_size is None:
            self.input_size = (224, 224)  # Default for backward compatibility
        else:
            self.input_size = input_size
        self.HW = self.input_size[0] * self.input_size[1]  # Total pixels

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.cuda())
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        initial_confidence = float(pred[0, c].item())
        n_steps = (self.HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        explanation_flat = explanation.reshape(-1, self.HW)
        salient_order = np.flip(np.argsort(explanation_flat, axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start.cuda())
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                # Get number of channels dynamically from tensor
                n_channels = start.shape[1]
                start_reshaped = start.cpu().numpy().reshape(1, n_channels, self.HW)
                finish_reshaped = finish.cpu().numpy().reshape(1, n_channels, self.HW)
                start_reshaped[0, :, coords[0]] = finish_reshaped[0, :, coords[0]]
                start = torch.from_numpy(start_reshaped.reshape(start.shape)).to(start.device)
        
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        assert n_samples % batch_size == 0
        # Get number of classes dynamically from model
        with torch.no_grad():
            first_batch_preds = self.model(img_batch[:batch_size].cuda()).cpu()
            num_classes = first_batch_preds.shape[1]

        predictions = torch.FloatTensor(n_samples, num_classes)
        predictions[:batch_size] = first_batch_preds  # Store first batch predictions

        for i in tqdm(range(1, n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i * batch_size:(i + 1) * batch_size].cuda()).cpu()
            predictions[i * batch_size:(i + 1) * batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (self.HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, self.HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        # Use proper substrate function instead of zeros
        substrate = self.substrate_fn(img_batch)

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda())
                preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            n_channels = start.shape[1]
            start_reshaped = start.cpu().numpy().reshape(n_samples, n_channels, self.HW)
            finish_reshaped = finish.cpu().numpy().reshape(n_samples, n_channels, self.HW)
            start_reshaped[r, :, coords] = finish_reshaped[r, :, coords]
            start = torch.from_numpy(start_reshaped.reshape(start.shape)).to(start.device)
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores
