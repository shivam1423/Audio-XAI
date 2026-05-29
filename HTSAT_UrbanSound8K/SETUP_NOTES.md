# Setup Notes for HTSAT_UrbanSound8K

## Folder Structure

```
HTSAT_UrbanSound8K/
├── evaluate_urbansound8k.py      # Main evaluation script
├── load_audioset_model.py        # AudioSet checkpoint loader
├── urbansound8k_dataset.py       # UrbanSound8K dataset loader
├── urbansound8k_config.py        # Configuration
├── requirements.txt               # Dependencies
├── run_evaluation.sh              # Shell script for SLURM
├── README.md                      # Full documentation
├── QUICK_START.md                # Quick start guide
├── .gitignore                    # Git ignore rules
└── SETUP_NOTES.md                # This file
```

## Dependencies

### Required Repository

The code expects the official HTSAT repository at:
- `../HTSAT/HTS-Audio-Transformer/` (relative to this folder)

If the repository is in a different location, you have two options:

1. **Create a symlink**:
   ```bash
   cd HTSAT_UrbanSound8K
   ln -s /path/to/HTS-Audio-Transformer HTS-Audio-Transformer
   ```

2. **Modify `load_audioset_model.py`**:
   Update the `OFFICIAL_REPO` path in the file.

## Key Differences from HTSAT Folder

1. **Dataset**: UrbanSound8K (10 classes) instead of ESC-50 (50 classes)
2. **Checkpoint**: AudioSet (527 classes) instead of ESC-50 fine-tuned
3. **Model Adaptation**: Transfers AudioSet weights but replaces final layer for 10 classes
4. **Metadata Format**: Different CSV structure (fold subdirectories)

## Usage Workflow

1. Download AudioSet checkpoint
2. Prepare UrbanSound8K dataset
3. Run evaluation script
4. View results in output directory

## Expected Performance

- **Zero-shot accuracy**: ~77% (using AudioSet features)
- This is competitive with other AudioSet pre-trained models
- For better results, consider fine-tuning on UrbanSound8K training folds

## Notes

- Original `HTSAT` folder remains untouched for XAI workflows
- This folder is self-contained and independent
- Can be safely deleted if not needed without affecting other code

