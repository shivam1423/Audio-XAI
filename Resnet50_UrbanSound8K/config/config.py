"""
Configuration for ResNet50 UrbanSound8K Training
Audio classification using mel spectrograms and ResNet50 architecture
"""

import os


class ResNet50Config:
    """Configuration parameters for ResNet50 UrbanSound8K training and evaluation"""
    
    # Dataset configuration
    dataset = 'urbansound8k'
    data_dir = None  # Set via command line
    num_classes = 10
    
    # Audio parameters
    sr = 22050  # 22.05kHz sampling rate (matches ESC-50 / RISE_audio ResNet)
    duration = 4.0  # Audio duration in seconds (UrbanSound8K clips ~4s)
    input_length = int(sr * duration)  # 88200 samples for 4 seconds
    
    # Mel spectrogram parameters (aligned with ESC-50 ResNet50 preprocessing)
    n_mels = 128
    fmax = None       # Use full Nyquist (sr/2), like ESC-50 default
    hop_length = 512
    n_fft = 1024      # Match ESC-50 / RISE_audio ResNet pipeline
    
    # Expected spectrogram *image* dimensions for ResNet input
    spec_height = 224  # Height after resizing mel image
    spec_width = 224   # Width after resizing mel image
    
    # Training parameters
    batch_size = 32
    n_epochs = 100
    lr = 0.001  # Initial learning rate for Adam/SGD
    optimizer_type = 'sgd'  # 'sgd' or 'adam'
    momentum = 0.9  # For SGD
    weight_decay = 1e-4
    
    # Learning rate scheduler
    scheduler_type = 'plateau'  # 'plateau', 'step', or 'cosine'
    scheduler_patience = 10  # For ReduceLROnPlateau
    scheduler_factor = 0.1  # LR reduction factor
    scheduler_min_lr = 1e-6
    
    # Cross-validation setup (single fold strategy)
    # Standard UrbanSound8K has 10 folds
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_fold = 9
    test_fold = 10
    n_folds = 10
    
    # Model parameters
    model_name = 'resnet50'
    pretrained = True  # Use ImageNet pretrained weights
    
    # Data loading
    num_workers = 4
    pin_memory = True
    
    # Device configuration
    device = 'cuda'  # Set via command line or auto-detect
    
    # Paths (will be set relative to project root)
    output_dir = 'trained_models'
    results_dir = 'results'
    log_file = None  # Will be set during training
    
    # Checkpoint configuration
    save_interval = 10  # Save checkpoint every N epochs
    save_best_only = True  # Save best model based on validation accuracy
    max_checkpoints = 5  # Keep only last N epoch checkpoints
    
    # Class labels for UrbanSound8K
    class_labels = [
        'air_conditioner',  # 0
        'car_horn',          # 1
        'children_playing',  # 2
        'dog_bark',          # 3
        'drilling',          # 4
        'engine_idling',     # 5
        'gun_shot',          # 6
        'jackhammer',        # 7
        'siren',             # 8
        'street_music'       # 9
    ]
    
    # Reproducibility
    seed = 42
    
    def __init__(self):
        """Initialize configuration with default values"""
        pass
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'data_dir') and args.data_dir is not None:
            self.data_dir = args.data_dir
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            self.output_dir = args.output_dir
        if hasattr(args, 'device') and args.device is not None:
            self.device = args.device
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.n_epochs = args.epochs
        if hasattr(args, 'lr') and args.lr is not None:
            self.lr = args.lr
        if hasattr(args, 'optimizer') and args.optimizer is not None:
            self.optimizer_type = args.optimizer
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            self.num_workers = args.num_workers
    
    def validate(self):
        """Validate configuration parameters"""
        # Data directory is required
        assert self.data_dir is not None, "data_dir must be specified"
        assert os.path.exists(self.data_dir), f"Data directory does not exist: {self.data_dir}"
        
        # Check metadata file exists
        metadata_paths = [
            os.path.join(self.data_dir, 'UrbanSound8K.csv'),
            os.path.join(self.data_dir, 'metadata', 'UrbanSound8K.csv'),
        ]
        metadata_exists = any(os.path.exists(p) for p in metadata_paths)
        assert metadata_exists, f"UrbanSound8K.csv not found in {self.data_dir}"
        
        # Validate parameters
        assert self.num_classes == 10, "UrbanSound8K has 10 classes"
        assert len(self.train_folds) > 0, "train_folds must not be empty"
        assert self.val_fold not in self.train_folds, "val_fold must not be in train_folds"
        assert self.test_fold not in self.train_folds, "test_fold must not be in train_folds"
        assert self.sr > 0, "Sample rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.n_epochs > 0, "Number of epochs must be positive"
        assert self.optimizer_type in ['sgd', 'adam'], "Optimizer must be 'sgd' or 'adam'"
    
    def display(self):
        """Display configuration parameters"""
        print('+' + '-'*70 + '+')
        print('| ResNet50 UrbanSound8K Configuration')
        print('+' + '-'*70 + '+')
        print(f'| Dataset           : {self.dataset}')
        print(f'| Data Directory    : {self.data_dir}')
        print(f'| Output Directory  : {self.output_dir}')
        print(f'| Num Classes       : {self.num_classes}')
        print(f'| Sample Rate       : {self.sr} Hz')
        print(f'| Audio Duration    : {self.duration} seconds')
        print(f'| Input Length      : {self.input_length} samples')
        print(f'| Mel Spectrograms  : {self.n_mels} mels, fmax={self.fmax} Hz')
        print(f'| Spec Dimensions   : {self.spec_height}x{self.spec_width}')
        print(f'| Model             : {self.model_name} (pretrained={self.pretrained})')
        print(f'| Batch Size        : {self.batch_size}')
        print(f'| Epochs            : {self.n_epochs}')
        print(f'| Optimizer         : {self.optimizer_type.upper()}')
        print(f'| Learning Rate     : {self.lr}')
        if self.optimizer_type == 'sgd':
            print(f'| Momentum          : {self.momentum}')
        print(f'| Weight Decay      : {self.weight_decay}')
        print(f'| LR Scheduler      : {self.scheduler_type}')
        if self.scheduler_type == 'plateau':
            print(f'| Scheduler Patience: {self.scheduler_patience}')
        print(f'| Train Folds       : {self.train_folds}')
        print(f'| Val Fold          : {self.val_fold}')
        print(f'| Test Fold         : {self.test_fold}')
        print(f'| Device            : {self.device}')
        print(f'| Random Seed       : {self.seed}')
        print('+' + '-'*70 + '+')
    
    def get_save_path(self, filename):
        """Get full path for saving files in output directory"""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)
    
    def get_results_path(self, filename):
        """Get full path for saving results"""
        os.makedirs(self.results_dir, exist_ok=True)
        return os.path.join(self.results_dir, filename)


# Create a default config instance
config = ResNet50Config()
