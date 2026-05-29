"""
HTSAT Configuration for ESC-50 Evaluation
Based on: https://github.com/RetroCirce/HTS-Audio-Transformer
"""

class Config:
    """Configuration for HTSAT ESC-50 Evaluation"""
    
    # Dataset Configuration
    dataset = "esc50"
    data_dir = "/Users/shivampandey/SS 25/Thesis/RISE_dev/ESC50"
    audio_dir = "/Users/shivampandey/SS 25/Thesis/RISE_dev/ESC50/audio"
    
    # ESC-50 Fold Configuration
    # Validation fold: 2
    # Training folds: 1, 3, 4, 5
    val_fold = 2
    train_folds = [1, 3, 4, 5]
    num_classes = 50
    
    # Model Configuration (HTSAT)
    model_name = "htsat"
    pretrained_checkpoint = "/Users/shivampandey/SS 25/Thesis/RISE_dev/HTSAT/HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
    
    # Audio Processing Parameters
    sample_rate = 32000  # HTSAT uses 32kHz
    clip_samples = 320000  # 10 seconds at 32kHz
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    
    # Model Architecture Parameters
    patch_size = 4
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [4, 8, 16, 32]
    window_size_spec = 8
    
    # Evaluation Parameters
    batch_size = 32
    num_workers = 4
    device = "cuda"  # or "cpu"
    
    # Class Labels for ESC-50
    class_labels = [
        'dog', 'rooster', 'pig', 'cow', 'frog',  # 0-4
        'cat', 'hen', 'insects', 'sheep', 'crow',  # 5-9
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',  # 10-14
        'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',  # 15-19
        'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',  # 20-24
        'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',  # 25-29
        'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',  # 30-34
        'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',  # 35-39
        'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',  # 40-44
        'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'  # 45-49
    ]

config = Config()




