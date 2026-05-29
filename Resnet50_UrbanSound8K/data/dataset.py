"""
UrbanSound8K Dataset Loader for ResNet50
Handles fold-based splitting and returns mel‑spectrogram images/tensors.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessor import AudioPreprocessor


class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K Dataset for ResNet50 training and evaluation.

    Returns mel‑spectrograms suitable for ResNet50 input, using a preprocessing
    pipeline that mirrors the ESC-50 ResNet50 setup:

    - Audio → mel spectrogram (librosa)
    - Mel spectrogram → grayscale 224×224 PIL image
    - Transform (if provided) converts to 3×224×224 tensor for ResNet50.

    UrbanSound8K structure:
    - 10 classes of urban sounds
    - 10 folds for cross-validation
    - Audio files organized in fold1/, fold2/, ..., fold10/ subdirectories
    - Metadata in UrbanSound8K.csv
    """

    def __init__(self, data_dir, folds, config, transform=None):
        """
        Initialize UrbanSound8K dataset.

        Args:
            data_dir: Root directory containing UrbanSound8K dataset
            folds: List of fold numbers to include (e.g., [1,2,3,4,5,6,7,8] for training)
            config: Configuration object with audio parameters
            transform: Optional transform to apply to spectrogram images
        """
        self.data_dir = data_dir
        self.folds = folds
        self.config = config
        self.transform = transform

        # Initialize preprocessor (audio → 224×224 grayscale image)
        self.preprocessor = AudioPreprocessor(
            sr=config.sr,
            duration=config.duration,
            n_mels=config.n_mels,
            fmax=config.fmax,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            target_width=config.spec_width,
            image_size=(config.spec_height, config.spec_width),
        )

        # Find and load metadata
        self.metadata_path = self._find_metadata("UrbanSound8K.csv")
        self.metadata = pd.read_csv(self.metadata_path)

        # Filter by folds
        if "fold" not in self.metadata.columns:
            raise ValueError("Metadata file must contain 'fold' column")

        self.metadata = self.metadata[self.metadata["fold"].isin(folds)]
        self.metadata = self.metadata.reset_index(drop=True)

        # Build file paths
        self._build_file_paths()

        # Extract class information
        self.num_classes = len(self.metadata["classID"].unique())

        print(f"Loaded UrbanSound8K dataset:")
        print(f"  - Folds: {sorted(self.folds)}")
        print(f"  - Total samples: {len(self.metadata)}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Sample rate: {config.sr} Hz")
        print(f"  - Audio duration: {config.duration}s")
        print(f"  - Spectrogram size (input to model): {config.spec_height}x{config.spec_width}")
    
    def _find_metadata(self, metadata_file):
        """Find metadata CSV file in common locations"""
        possible_paths = [
            os.path.join(self.data_dir, metadata_file),
            os.path.join(self.data_dir, 'metadata', metadata_file),
            os.path.join(self.data_dir, '..', metadata_file),
            os.path.join(os.path.dirname(self.data_dir), metadata_file),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Metadata file '{metadata_file}' not found. Searched in:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def _build_file_paths(self):
        """Build full file paths for audio files."""
        # Try different possible structures for UrbanSound8K

        # Structure 1: data_dir/fold1/, fold2/, etc.
        self.metadata["audio_path"] = self.metadata.apply(
            lambda row: os.path.join(
                self.data_dir, 
                f"fold{row['fold']}",
                row["slice_file_name"],
            ),
            axis=1,
        )

        # Verify that at least some files exist
        sample_paths = self.metadata["audio_path"].head(5)
        existing = sum(1 for p in sample_paths if os.path.exists(p))

        if existing == 0:
            # Structure 2: data_dir/audio/fold1/, fold2/, etc. (standard UrbanSound8K)
            self.metadata["audio_path"] = self.metadata.apply(
                lambda row: os.path.join(
                    self.data_dir,
                    "audio",
                    f"fold{row['fold']}",
                    row["slice_file_name"],
                ),
                axis=1,
            )
            sample_paths = self.metadata["audio_path"].head(5)
            existing = sum(1 for p in sample_paths if os.path.exists(p))

            if existing == 0:
                # Structure 3: audio files directly in data_dir
                self.metadata["audio_path"] = self.metadata["slice_file_name"].apply(
                    lambda x: os.path.join(self.data_dir, x)
                )
                sample_paths = self.metadata["audio_path"].head(5)
                existing = sum(1 for p in sample_paths if os.path.exists(p))

                if existing == 0:
                    raise FileNotFoundError(
                        f"Audio files not found. Please check data_dir: {self.data_dir}\n"
                        f"Expected structures:\n"
                        f"  - {self.data_dir}/fold1/, fold2/, etc.\n"
                        f"  - {self.data_dir}/audio/fold1/, fold2/, etc. (standard)\n"
                        f"  - {self.data_dir}/<audio_files>"
                    )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            x: Preprocessed input for ResNet50 (typically 3×224×224 tensor)
            label: Integer class label (0-9)
        """
        row = self.metadata.iloc[idx]

        # Get audio path and label
        audio_path = row["audio_path"]
        label = int(row["classID"])

        # Verify file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and preprocess audio to mel‑spectrogram image
        try:
            mel_img = self.preprocessor.preprocess(audio_path)
        except Exception as e:
            raise Exception(f"Error preprocessing {audio_path}: {e}")

        # Apply additional transforms (e.g., ToTensor + channel replication + normalisation)
        if self.transform:
            x = self.transform(mel_img)
        else:
            x = mel_img

        return x, label
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        return self.metadata['classID'].value_counts().sort_index()
    
    def get_samples_by_class(self, class_id):
        """Get all samples for a specific class"""
        class_metadata = self.metadata[self.metadata['classID'] == class_id]
        return class_metadata.index.tolist()
    
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.config.class_labels[class_id]


def create_data_loaders(config):
    """
    Create data loaders for train, validation, and test sets.

    Args:
        config: ResNet50Config object with dataset parameters

    Returns:
        train_loader, val_loader, test_loader
    """
    # Import here to avoid making torchvision a hard dependency for modules that
    # only need metadata helpers.
    import torchvision.transforms as transforms

    # Preprocessing pipeline matching ESC-50 ResNet50 setup:
    #   PIL grayscale image → tensor → replicate to 3 channels → normalise.
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    # Training dataset
    train_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=config.train_folds,
        config=config,
        transform=preprocess,
    )

    # Validation dataset
    val_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=[config.val_fold],
        config=config,
        transform=preprocess,
    )

    # Test dataset
    test_dataset = UrbanSound8KDataset(
        data_dir=config.data_dir,
        folds=[config.test_fold],
        config=config,
        transform=preprocess,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_data_info(config):
    """
    Get information about the dataset splits
    
    Args:
        config: ResNet50Config object
    
    Returns:
        Dictionary with dataset statistics
    """
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    info = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'num_classes': train_dataset.num_classes,
        'train_folds': config.train_folds,
        'val_fold': config.val_fold,
        'test_fold': config.test_fold,
        'sample_rate': config.sr,
        'audio_duration': config.duration,
        'spec_shape': (config.spec_height, config.spec_width)
    }
    
    return info
