"""
HTSAT Configuration for UrbanSound8K Evaluation
Based on: https://github.com/RetroCirce/HTS-Audio-Transformer
"""

class UrbanSound8KConfig:
    """Configuration for HTSAT UrbanSound8K Evaluation"""
    
    # Dataset Configuration
    dataset = "urbansound8k"
    data_dir = None  # Set via command line
    audio_dir = None  # Set via command line
    
    # UrbanSound8K has 10 folds (1-10)
    # Common splits:
    # - Test: fold 10
    # - Validation: fold 9
    # - Training: folds 1-8
    test_fold = 10
    val_fold = 9
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    num_classes = 10
    
    # Model Configuration (HTSAT)
    model_name = "htsat"
    pretrained_checkpoint = None  # Set via command line (AudioSet checkpoint)
    
    # Audio Processing Parameters
    sample_rate = 32000  # HTSAT uses 32kHz
    clip_samples = 320000  # 10 seconds at 32kHz
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    
    # Model Architecture Parameters (same as AudioSet/ESC-50)
    patch_size = 4
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [4, 8, 16, 32]
    window_size_spec = 8
    htsat_spec_size = 256
    htsat_stride = (4, 4)
    
    # Model features
    enable_tscam = True
    
    # Evaluation Parameters
    batch_size = 32
    num_workers = 4
    device = "cuda"  # or "cpu"
    
    # Class Labels for UrbanSound8K (10 classes)
    class_labels = [
        'air_conditioner',  # 0
        'car_horn',          # 1
        'children_playing',  # 2
        'dog_bark',          # 3
        'drilling',           # 4
        'engine_idling',     # 5
        'gun_shot',          # 6
        'jackhammer',        # 7
        'siren',             # 8
        'street_music'       # 9
    ]
    
    # Loss config (for reference, not used in evaluation)
    loss_type = "clip_ce"
    
    # Data augmentation flags (not used in evaluation)
    enable_token_label = False
    enable_time_shift = False
    enable_label_enhance = False
    enable_repeat_mode = False

config = UrbanSound8KConfig()

