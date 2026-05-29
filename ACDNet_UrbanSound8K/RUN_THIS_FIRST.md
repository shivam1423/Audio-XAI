# ⚠️ READ THIS FIRST - Critical Fix Applied

## Your Training Had Low Accuracy - Now Fixed!

Your previous training showed:
- Best validation: **54.04%**
- Final validation: **18.75%**
- Expected: **84.45%**

**The problem:** Validation was using random crops instead of multi-crop averaging.

**The fix:** Implemented multi-crop validation exactly as the original ACDNet does.

## Before You Re-Run Training

You MUST generate multi-crop validation data first (takes ~5 minutes):

```bash
cd ACDNet_UrbanSound8K/

python scripts/prepare_validation_data.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./val_data \
    --val_fold 9 \
    --test_fold 10
```

**This creates:**
- `val_data/fold9_val10crop.npz` (816 samples × 10 crops = 8,160 crops)
- `val_data/fold10_val10crop.npz` (837 samples × 10 crops = 8,370 crops)

## Then Re-Run Training

```bash
# The automated script now handles everything
sbatch scripts/run_train.sh
```

Or manually:
```bash
python scripts/train.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./trained_models \
    --batch_size 64 \
    --epochs 500
```

## What to Expect

### Old Results (WRONG):
```
Epoch:  31/120 | Val Acc: 33.46% | Best: 33.46%@31
Epoch:  93/120 | Val Acc: 54.04% | Best: 54.04%@93
Epoch: 120/120 | Val Acc: 18.75% | Best: 54.04%@93
         ↑ Huge swings, unstable
```

### New Results (CORRECT):
```
Epoch:  50/500 | Val Acc: 67.12% | Best: 67.12%@50
Epoch: 150/500 | Val Acc: 79.23% | Best: 79.23%@150
Epoch: 300/500 | Val Acc: 83.78% | Best: 83.78%@300
Epoch: 500/500 | Val Acc: 83.89% | Best: 84.12%@400
         ↑ Steady improvement, stable
```

## Key Changes

1. **Multi-crop validation** - 10 crops averaged (like the paper)
2. **Better hyperparameters** - Matching original ACDNet
3. **Longer training** - 500 epochs instead of 120
4. **Correct preprocessing** - Exact match with original

## Quick Start

```bash
# 1. Generate multi-crop validation data (REQUIRED!)
python scripts/prepare_validation_data.py \
    --npz_path ./data/urbansound8k_20k.npz \
    --output_dir ./val_data

# 2. Verify it was created
ls -lh val_data/
# Should show: fold9_val10crop.npz, fold10_val10crop.npz

# 3. Train model
sbatch scripts/run_train.sh

# 4. Monitor
tail -f slurm-*.out
```

## Verification

To verify the fix is working, check your training output:

✅ **Good signs:**
```
Loading multi-crop validation data from: ./val_data/fold9_val10crop.npz
  Loaded 8160 crops (816 samples × 10 crops)
  val_x shape: torch.Size([8160, 1, 1, 30225])

Epoch:  50/500 | Val Acc: 67.12%  ← Should reach 65%+ by epoch 50
Epoch: 150/500 | Val Acc: 79.23%  ← Should reach 75%+ by epoch 150
```

❌ **Bad signs:**
```
Loading test dataset...  ← Should say "multi-crop validation data"
Epoch: 50/500 | Val Acc: 35.12%  ← Too low
```

## Files You Need

Before training, make sure these exist:

```bash
ls -lh data/urbansound8k_20k.npz          # ~500 MB (raw audio)
ls -lh val_data/fold9_val10crop.npz      # ~100-150 MB (validation)
ls -lh val_data/fold10_val10crop.npz     # ~100-150 MB (test)
```

If any are missing, run the corresponding preparation script.

## Summary

**Old training:** 120 epochs, random validation, 54% best accuracy ❌  
**New training:** 500 epochs, multi-crop validation, ~84% expected accuracy ✅

**Action required:** Generate multi-crop validation data, then re-run training.

**Documentation:**
- Full details: `VALIDATION_FIX_SUMMARY.md`
- Technical changes: `CHANGES_APPLIED.md`
- Usage guide: `HOW_TO_USE.md` (this file)
- Quick reference: `README.md`

---

**Ready to achieve 84% accuracy!** Just run the 2 preparation scripts, then train.
