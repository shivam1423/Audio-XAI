"""
Configuration file for Wav2Vec2 ESC-50 fine-tuning
"""
import os

class Config:
    # Model settings
    MODEL_NAME = "facebook/wav2vec2-base"
    NUM_CLASSES = 50  # ESC-50 has 50 classes
    DROPOUT_RATE = 0.1
    
    # Training settings
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data settings
    SAMPLE_RATE = 16000
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Paths
    DATA_DIR = "data/ESC-50"
    OUTPUT_DIR = "outputs"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Device settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 2
    
    # ESC-50 class names
    CLASS_NAMES = [
        "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
        "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
        "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
        "crickets", "crow", "crying_baby", "dog", "door_wood_knock",
        "door_wood_creaks", "drinking_sipping", "engine", "fireworks", "footsteps",
        "frog", "glass_breaking", "hand_saw", "helicopter", "hen",
        "insects", "keyboard_typing", "laughing", "lawn_mower", "mouse_click",
        "pig", "pouring_water", "rain", "rooster", "sea_waves",
        "sheep", "siren", "snake_hissing", "sneezing", "thunderstorm",
        "toilet_flush", "train", "vacuum_cleaner", "washing_machine", "water_drops"
    ]

