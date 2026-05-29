"""
BC (Between-Class) Learning Data Generator for ACDNet
Mixes two samples from different classes to create augmented training data

This version loads from preprocessed NPZ files for instant initialization
"""

import numpy as np
import random
import torch
from torch.utils.data import Dataset
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import helpers as U


class BCLearningGenerator(Dataset):
    """
    BC Learning data generator for training ACDNet
    
    Between-Class (BC) Learning: Mix two audio samples from different classes
    to create augmented training data with soft labels
    
    Args:
        npz_path: Path to preprocessed NPZ file (urbansound8k_20k.npz)
        config: ACDNetConfig object with training parameters
        train_folds: List of fold numbers to use for training (e.g., [1,2,3,4,5,6,7,8])
    """
    
    def __init__(self, npz_path, config, train_folds):
        self.config = config
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.sr = config.sr
        self.input_length = config.input_length
        
        # Load preprocessed NPZ file
        print(f"Loading preprocessed dataset from: {npz_path}")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"NPZ file not found: {npz_path}\n"
                f"Please run: python scripts/prepare_urbansound8k.py --data_dir <path> --output_dir ./data"
            )
        
        dataset = np.load(npz_path, allow_pickle=True)
        
        # Load data from specified folds
        self.data = []
        total_samples = 0
        for fold in train_folds:
            fold_key = f'fold{fold}'
            if fold_key not in dataset:
                raise KeyError(f"Fold {fold} not found in NPZ file. Available: {list(dataset.keys())}")
            
            fold_data = dataset[fold_key].item()
            sounds = fold_data['sounds']
            labels = fold_data['labels']
            
            for sound, label in zip(sounds, labels):
                self.data.append((sound, label))
            
            total_samples += len(sounds)
        
        print(f"BC Learning Generator initialized:")
        print(f"  - Loaded folds: {train_folds}")
        print(f"  - Total samples: {len(self.data)}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Batches per epoch: {len(self)}")
        
        # Setup preprocessing functions
        self.preprocess_funcs = self.preprocess_setup()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """
        Generate one batch of BC-mixed data
        
        Returns:
            batchX: NumPy array of shape (batch_size, 1, input_length, 1)
            batchY: NumPy array of soft labels (batch_size, num_classes)
        
        Note: Trainer will apply moveaxis to convert to (batch, 1, 1, length) for model
        """
        batchX, batchY = self.generate_batch(batch_idx)
        
        # Reshape following original ACDNet pattern: (batch, 1, length, 1)
        # Trainer will use moveaxis(x, 3, 1) to get final shape (batch, 1, 1, length)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        
        return batchX, batchY
    
    def generate_batch(self, batch_idx):
        """
        Generate a batch of BC-mixed samples
        
        BC Learning process:
        1. Select two samples from different classes
        2. Preprocess each sample (augmentation)
        3. Mix with random ratio r
        4. Create soft label: label = r * label1 + (1-r) * label2
        5. Apply random gain augmentation
        """
        sounds = []
        labels = []
        
        for i in range(self.batch_size):
            # Select two training examples from different classes
            while True:
                idx1 = random.randint(0, len(self.data) - 1)
                idx2 = random.randint(0, len(self.data) - 1)
                sound1, label1 = self.data[idx1]
                sound2, label2 = self.data[idx2]
                
                if label1 != label2:
                    break
            
            # Preprocess both sounds
            sound1 = self.preprocess(sound1.copy())
            sound2 = self.preprocess(sound2.copy())
            
            # Mix two examples with random ratio
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.sr).astype(np.float32)
            
            # Create soft label (one-hot encoded, then mixed)
            eye = np.eye(self.num_classes)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)
            
            # Apply random gain augmentation
            if self.config.strong_augment:
                sound = U.random_gain(6)(sound).astype(np.float32)
            
            sounds.append(sound)
            labels.append(label)
        
        sounds = np.asarray(sounds)
        labels = np.asarray(labels)
        
        return sounds, labels
    
    def preprocess_setup(self):
        """Setup preprocessing functions"""
        funcs = []
        
        # Random scale augmentation if strong augmentation enabled
        if self.config.strong_augment:
            funcs += [U.random_scale(1.25)]
        
        # Standard preprocessing: padding, random crop, normalize
        funcs += [
            U.padding(self.input_length // 2),
            U.random_crop(self.input_length),
            U.normalize(32768.0)
        ]
        
        return funcs
    
    def preprocess(self, sound):
        """Apply preprocessing functions to audio"""
        for f in self.preprocess_funcs:
            sound = f(sound)
        return sound


def create_bc_generator(npz_path, config, train_folds):
    """
    Create BC Learning generator from preprocessed NPZ file
    
    Args:
        npz_path: Path to preprocessed NPZ file (urbansound8k_20k.npz)
        config: ACDNetConfig object
        train_folds: List of fold numbers for training
    
    Returns:
        BCLearningGenerator instance
    """
    return BCLearningGenerator(npz_path, config, train_folds)



def setup_generator(npz_path, config, train_folds):
    """
    Setup BC Learning generator from NPZ file (follows original ACDNet pattern)
    
    Args:
        npz_path: Path to preprocessed NPZ file
        config: ACDNetConfig object
        train_folds: List of training fold numbers
    
    Returns:
        BCLearningGenerator instance ready for training
    """
    return BCLearningGenerator(npz_path, config, train_folds)
