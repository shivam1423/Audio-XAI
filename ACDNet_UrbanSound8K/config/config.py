"""
Configuration for ACDNet UrbanSound8K Training
Based on: "Environmental Sound Classification on the Edge" paper
arXiv: 2103.03483
"""

import os


class ACDNetConfig:
    """Configuration parameters for ACDNet UrbanSound8K training and evaluation"""
    
    # Dataset configuration
    dataset = 'urbansound8k'
    data_dir = None  # Set via command line (for preprocessing)
    npz_path = None  # Path to preprocessed NPZ file (set via command line or auto-detect)
    num_classes = 10
    
    # Audio parameters (from original ACDNet)
    sr = 20000  # 20kHz sampling rate
    input_length = 30225  # Matching original ESC-50 (ACDNet/common/opts.py:30)
    
    # Training parameters (based on original ACDNet - ACDNet/common/opts.py)
    batch_size = 64  # Original ACDNet uses 64
    n_epochs = 500  # Balanced: more than 120, less than ESC-50's 2000
    lr = 0.1  # Initial learning rate
    schedule = [0.3, 0.6, 0.9]  # LR decay at epochs 150, 300, 450 (matching original pattern)
    warmup = 10  # Warmup epochs like original ACDNet
    weight_decay = 5e-4
    momentum = 0.9
    
    # Cross-validation setup (single fold strategy)
    # Standard UrbanSound8K has 10 folds
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_fold = 9
    test_fold = 10
    n_folds = 10
    
    # BC (Between-Class) Learning configuration
    use_bc_learning = True
    strong_augment = True  # Random gain augmentation
    
    # Model parameters
    n_crops = 10  # For testing phase (10-crop evaluation)
    
    # Device configuration
    device = 'cuda'  # Set via command line or auto-detect
    
    # Paths (will be set relative to project root)
    trained_models_dir = 'trained_models'
    results_dir = 'results'
    
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
        if hasattr(args, 'npz_path') and args.npz_path is not None:
            self.npz_path = args.npz_path
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            self.trained_models_dir = args.output_dir
        if hasattr(args, 'device') and args.device is not None:
            self.device = args.device
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.n_epochs = args.epochs
        if hasattr(args, 'lr') and args.lr is not None:
            self.lr = args.lr
    
    def validate(self):
        """Validate configuration parameters"""
        # NPZ path is required for training
        assert self.npz_path is not None, (
            "npz_path must be specified. "
            "Please run: python scripts/prepare_urbansound8k.py --data_dir <path> --output_dir ./data"
        )
        assert os.path.exists(self.npz_path), f"NPZ file does not exist: {self.npz_path}"
        
        assert self.num_classes == 10, "UrbanSound8K has 10 classes"
        assert len(self.train_folds) > 0, "train_folds must not be empty"
        assert self.val_fold not in self.train_folds, "val_fold must not be in train_folds"
        assert self.test_fold not in self.train_folds, "test_fold must not be in train_folds"
        assert self.sr == 20000, "Sample rate must be 20kHz for ACDNet"
        assert self.input_length == 30225, "Input length must be 30225 samples (matching original ACDNet)"
    
    def display(self):
        """Display configuration parameters"""
        print('+' + '-'*60 + '+')
        print('| ACDNet UrbanSound8K Configuration')
        print('+' + '-'*60 + '+')
        print(f'| Dataset        : {self.dataset}')
        print(f'| Data Directory : {self.data_dir}')
        print(f'| Num Classes    : {self.num_classes}')
        print(f'| Sample Rate    : {self.sr} Hz')
        print(f'| Input Length   : {self.input_length} samples ({self.input_length/self.sr:.1f}s)')
        print(f'| Batch Size     : {self.batch_size}')
        print(f'| Epochs         : {self.n_epochs}')
        print(f'| Learning Rate  : {self.lr}')
        print(f'| LR Schedule    : {self.schedule}')
        print(f'| Weight Decay   : {self.weight_decay}')
        print(f'| Momentum       : {self.momentum}')
        print(f'| Train Folds    : {self.train_folds}')
        print(f'| Val Fold       : {self.val_fold}')
        print(f'| Test Fold      : {self.test_fold}')
        print(f'| BC Learning    : {self.use_bc_learning}')
        print(f'| Strong Augment : {self.strong_augment}')
        print(f'| Device         : {self.device}')
        print('+' + '-'*60 + '+')


# Create a default config instance
config = ACDNetConfig()
