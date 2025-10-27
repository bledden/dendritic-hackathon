# Test 1b: Fixed-Epoch Compression - User Guide

## What's Different from Test 1?

**Test 1 (history-based):**
- Compression triggers after 3 epochs WITHOUT improvement
- Problem: If WER keeps improving → NO compression triggers
- Result: Baseline data on training with dendrites, but no actual compression

**Test 1b (fixed-epoch):**
- Compression triggers at FIXED epochs (8, 16, 24) regardless of WER
- Advantage: GUARANTEED compressions (3 cycles)
- Result: Actual parameter reduction + compression behavior data

## Quick Start

```bash
cd whisper-edge-optimization\dendritic
RUN_TEST_1B.bat
```

## Expected Behavior

### Epoch 8: First Compression
```
Epoch 8/25
Train loss: ~0.006
Validation WER: ~107-108%

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
Parameter change: 244M → ~230M (5-7% reduction)
```

**What happens:**
1. Loads best checkpoint (from epoch with lowest WER)
2. Creates candidates (244M → 442M temporarily)
3. Trains candidates for ~1 epoch
4. Selects best dendrites
5. Prunes: 442M → ~230M
6. Continues training

### Epoch 16: Second Compression
```
Epoch 16/25
Train loss: ~0.005
Validation WER: ~105-107%

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
Parameter change: 230M → ~220M (4-5% additional reduction)
```

**Diminishing returns:** Second compression typically gives less reduction than first.

### Epoch 24: Third Compression (if max_dendrites=3)
```
Epoch 24/25
Train loss: ~0.004
Validation WER: ~104-106%

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
Parameter change: 220M → ~215M (2-3% additional reduction)
```

**Very diminishing:** Third compression may give minimal benefit.

## Key Differences to Monitor

### Memory Pattern:
```
Epochs 1-7:   4.5GB (baseline + dendrites)
Epoch 8:      Peak ~10GB (candidates created)
Epochs 9-15:  4.0GB (compressed model, slightly smaller)
Epoch 16:     Peak ~10GB (candidates again)
Epochs 17-23: 3.8GB (more compressed)
Epoch 24:     Peak ~10GB (final compression)
Epoch 25:     3.5-3.8GB (final compressed model)
```

### Parameter Count Tracking:
```
Epoch 1-7:    244M params
Epoch 8:      Spike to 442M → compress to ~230M
Epoch 9-15:   230M params
Epoch 16:     Spike to ~420M → compress to ~220M
Epoch 17-23:  220M params
Epoch 24:     Spike to ~410M → compress to ~215M
Epoch 25:     Final: 215M params
```

## Success Criteria

### Must achieve:
1. ✅ **3 compressions occur** at epochs 8, 16, 24
2. ✅ **Parameter reduction >10%** (244M → <220M)
3. ✅ **No crashes** during restructuring
4. ✅ **WER stays reasonable** (<120%)

### Good result:
- 10-15% reduction (220-208M params)
- WER improves or stays stable through compressions
- Memory peaks stay <12GB

### Excellent result:
- >15% reduction (<208M params)
- WER improves after compressions
- Training accelerates (faster epochs after compression)

## What to Watch For

### During Compression (epochs 8, 16, 24):

**Expected messages:**
```
Updating PAI tracker...
Switching to dendrite training mode...
Importing best Model for switch to PA...
*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
   New parameters: XXX,XXX,XXX
   Reduction: X.X%
   [OK] Optimizer reinitialized
```

**Memory spike:**
```
[GPU Memory - Epoch 8 after training]
  Allocated: 8-10GB (candidates being created)
```

**Don't panic!** This is temporary - memory will drop back down after compression completes.

### After Each Compression:

**Check these:**
1. **Parameter count decreased?** Should go down 5-10% per cycle
2. **WER didn't explode?** Should stay within 5% of pre-compression
3. **Training continues?** Should seamlessly resume
4. **Memory normalized?** Should drop back to 4-5GB

## Interpreting Results

### Scenario 1: Perfect Run
```
Epoch 8:  244M → 230M (6% reduction), WER 108% → 109%
Epoch 16: 230M → 218M (5% reduction), WER 106% → 107%
Epoch 24: 218M → 210M (4% reduction), WER 105% → 105%
Final: 210M params (14% total reduction), WER 104%
```
**Interpretation:** Compression works! Each cycle gives meaningful reduction with minimal WER impact.

### Scenario 2: Aggressive Compression
```
Epoch 8:  244M → 220M (10% reduction), WER 108% → 115%
Epoch 16: 220M → 200M (9% reduction), WER 112% → 118%
Epoch 24: 200M → 190M (5% reduction), WER 115% → 119%
Final: 190M params (22% reduction), WER 117%
```
**Interpretation:** High compression but quality degraded. MLPs might be compressed too aggressively.

### Scenario 3: Minimal Compression
```
Epoch 8:  244M → 240M (2% reduction), WER 108% → 108%
Epoch 16: 240M → 238M (1% reduction), WER 106% → 106%
Epoch 24: 238M → 237M (0.5% reduction), WER 105% → 105%
Final: 237M params (3% total reduction), WER 104%
```
**Interpretation:** Very conservative. MLPs had little redundancy, or dendrites didn't capture it well.

### Scenario 4: Compression Helps WER
```
Epoch 8:  244M → 232M (5% reduction), WER 108% → 106% ✓
Epoch 16: 232M → 222M (4% reduction), WER 106% → 104% ✓
Epoch 24: 222M → 215M (3% reduction), WER 104% → 103% ✓
Final: 215M params (12% reduction), WER 103%
```
**Interpretation:** BEST CASE! Compression acts as regularization, improving generalization.

## Comparison to Test 1

After both tests complete, compare:

| Metric | Test 1 (History) | Test 1b (Fixed) |
|--------|------------------|-----------------|
| **Compressions** | 0-1 (if lucky) | 3 (guaranteed) |
| **Final params** | 244M (no compression) | 210-235M (compressed) |
| **Final WER** | ~104-106% | ~104-108% |
| **Runtime** | ~3.5 hours | ~3.5 hours |

**Key insights:**
- Test 1: Shows training dynamics with dendrites
- Test 1b: Shows actual compression behavior
- Combined: Complete picture of dendritic integration

## Next Steps After Test 1b

### If successful (>10% reduction, reasonable WER):
→ **Test 2: Aggressive Compression**
- Add decoder cross-attention layers
- Add attention output projections
- Target: 20-30% reduction

### If WER too high (>115% final):
→ **Test 1c: Decoder-Only**
- Compress only decoder MLPs
- Freeze encoder entirely
- More conservative

### If reduction too small (<5%):
→ **Adjust settings:**
- Increase `fixed_switch_num` to 12 (compress at 12, 24)
- Or reduce `max_dendrites` to 2 (only 2 cycles)
- Or increase epochs to 40 (more opportunities)

## Technical Details

### Why Epochs 8, 16, 24?

**Epoch 8:**
- Train loss plateaued (~0.006)
- Model learned main patterns
- Good time for first compression

**Epoch 16:**
- Post-compression training stabilized
- Model adapted to new structure
- Ready for second round

**Epoch 24:**
- Near end of training
- Final refinement opportunity
- Last chance to compress

### What Happens During Compression?

**Step-by-step:**
1. **Freeze training** - Current epoch pauses
2. **Load best checkpoint** - Revert to best WER epoch
3. **Switch mode** - Change from "neuron" to "dendrite" training
4. **Create candidates** - Add 198M candidate parameters
5. **Train candidates** - 1 epoch to evaluate which neurons to keep
6. **Select best** - Choose top dendrites per layer
7. **Prune** - Remove low-importance neurons
8. **Restructure** - Rebuild model with compressed structure
9. **Resume training** - Continue from new checkpoint

**Total time per compression:** ~10-15 minutes

### Memory Management During Compression:

```
Normal training:     244M params × 2 bytes (BF16) = 0.49GB
                   + 244M × 8 bytes (optimizer) = 1.95GB
                   + activations ~1GB
                   = ~3.5GB total

During compression: 442M params × 2 bytes = 0.88GB
                   + 442M × 8 bytes (opt) = 3.54GB
                   + activations ~1GB
                   + candidates ~1GB
                   = ~6.5-7GB total (temporarily)

After compression:  220M params × 2 bytes = 0.44GB
                   + 220M × 8 bytes (opt) = 1.76GB
                   + activations ~1GB
                   = ~3.2GB total
```

**Peak:** ~7GB during compression
**Normal:** ~3-4GB during training

---

## Files Created

Results saved to: `D:\ML_Results\dendritic_whisper\test_1b_fixed_trigger_mlp_only\`

Key files:
- `final_results.json` - Summary with compressions
- `best_model.pt` - Final compressed checkpoint
- `*switch_epochs.csv` - Records compression events (should show 3 entries)
- `*param_counts.csv` - Parameter history (should show 3 drops)
- `*Scores.csv` - WER progression

---

**Ready to run! This test WILL compress - it's guaranteed by the fixed-epoch trigger.**

Good luck! 🚀
