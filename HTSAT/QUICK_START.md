# HTSAT Evaluation - Quick Start

## TL;DR

Run these three commands:

```bash
cd "/Users/shivampandey/SS 25/Thesis/RISE_dev/HTSAT"
python test_setup.py
./run_evaluation.sh
```

Results will be in `./results_fold2/`

## What You Have

✅ **Checkpoint**: `HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt` (324 MB)  
✅ **Dataset**: ESC-50 audio files in `../ESC50/audio/`  
✅ **Configuration**:
- Validation fold: **2**
- Training folds: **1, 3, 4, 5**
- Sample rate: **32 kHz**
- Model: **HTSAT**

## Quick Commands

### 1. Test Setup (30 seconds)
```bash
python test_setup.py
```
Verifies everything is configured correctly.

### 2. Run Evaluation (2-10 minutes depending on GPU)
```bash
./run_evaluation.sh
```

Or with custom options:
```bash
python evaluate.py --val_fold 2 --device cuda --batch_size 32
```

### 3. View Results
```bash
# View accuracy
cat results_fold2/evaluation_summary_rise.txt

# View predictions
head results_fold2/predictions.csv

# View confusion matrix
open results_fold2/confusion_matrix.png  # macOS
```

## Expected Output

```
==========================================================
EVALUATION RESULTS
==========================================================
Total samples: ~400 (varies by fold)
Accuracy: ~98.5% (based on checkpoint name)
==========================================================
```

## Files Created

```
HTSAT/
├── evaluate.py                    # Main evaluation script ⭐
├── htsat_model.py                 # HTSAT model
├── esc50_dataset.py               # Dataset loader
├── htsat_config.py                # Configuration
├── test_setup.py                  # Setup verification
├── run_evaluation.sh              # Run script ⭐
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── USAGE_GUIDE.md                 # Detailed guide
└── QUICK_START.md                 # This file
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU / CUDA error | Use `--device cpu` |
| Out of memory | Reduce `--batch_size 16` |
| Import errors | Run `pip install -r requirements.txt` |
| Slow evaluation | Increase `--num_workers 8` |

## Next Steps

1. ✅ Run basic evaluation (above)
2. 📊 Analyze results in `results_fold2/`
3. 🔄 Evaluate other folds by changing `--val_fold`
4. 📖 Read `USAGE_GUIDE.md` for advanced options

## Alternative: Use Official Repository

For evaluation using the original HTSAT code:

```bash
# Setup (one-time)
./setup_official_htsat.sh

# Evaluate
python evaluate_with_official_repo.py
```

## Need Help?

1. Run `python test_setup.py` to diagnose issues
2. Check `USAGE_GUIDE.md` for detailed instructions
3. See `README.md` for complete documentation

---

**Ready to evaluate?** → `./run_evaluation.sh`




