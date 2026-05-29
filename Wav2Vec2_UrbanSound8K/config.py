"""
Configuration file for Wav2Vec2 UrbanSound8K fine-tuning
"""
import os

class Config:
    # Model settings
    MODEL_NAME = "facebook/wav2vec2-base"
    NUM_CLASSES = 10  # UrbanSound8K has 10 classes
    DROPOUT_RATE = 0.1
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 30
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data settings
    SAMPLE_RATE = 16000
    MAX_DURATION = 4.0  # UrbanSound8K clips are ~4 seconds
    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]  # Training folds
    VAL_FOLD = 9  # Validation fold
    TEST_FOLD = 10  # Test fold
    
    # Paths (relative for cluster compatibility)
    DATA_DIR = "../UrbanSound8K"
    OUTPUT_DIR = "outputs"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Device settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    # UrbanSound8K class names (10 classes)
    CLASS_NAMES = [
        "air_conditioner",  # 0
        "car_horn",          # 1
        "children_playing",  # 2
        "dog_bark",          # 3
        "drilling",          # 4
        "engine_idling",     # 5
        "gun_shot",          # 6
        "jackhammer",        # 7
        "siren",             # 8
        "street_music"       # 9
    ]
