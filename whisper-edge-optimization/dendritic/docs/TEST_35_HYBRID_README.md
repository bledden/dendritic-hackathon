# 35-Epoch Hybrid Compression Test

## Overview

This test combines all optimizations and features developed so far:
- ✅ **Hybrid compression mode** - Guarantees compression at epochs 10, 20, 30
- ✅ **Memory clearing** - Prevents fragmentation (83% faster post-compression!)
- ✅ **Safe termination** - Proper GPU cleanup, no crashes
- ✅ **PAUSE feature** - Stop training anytime safely
- ✅ **FORCE_COMPRESS feature** - Manual compression trigger
- ✅ **pin_memory** - Faster data loading

## Quick Start

```bash
cd whisper-edge-optimization/dendritic
.\RUN_TEST_35_HYBRID.bat
```

**Expected runtime:** ~7.5 hours (overnight test)

## What to Expect

### Timeline

**Epochs 1-10** (~100 minutes)
- Training with original Whisper-Small (240M params)
- WER should improve from ~24% to ~18%
- At epoch 10: **HYBRID TRIGGER** → Add dendrites (240M → 467M)

**Epochs 11-20** (~165 minutes)
- Training with candidate dendrites (467M params)
- Learning which dendrites are important
- Slower: ~3.6 it/s vs 5.8 it/s (expected)
- At epoch 20: **HYBRID TRIGGER** → **PRUNE** (467M → ~150M) ← **THE BIG MOMENT**

**Epochs 21-35** (~250 minutes)
- Training compressed model (~150M params)
- Evaluating final WER
- Should stay <25% WER (ideally <20%)

### Key Moments

**Epoch 10:**
```
======================================================================
HYBRID MODE: FORCE TRIGGER
======================================================================
Forcing compression at epoch 10 (interval: 10)
Forcing PAI to compress...

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
   New parameters: 467,351,808
   Reduction: -94.3%
   [OK] Optimizer reinitialized
   [OK] GPU memory cache cleared
```

**Epoch 20:**
```
======================================================================
HYBRID MODE: FORCE TRIGGER
======================================================================
Forcing compression at epoch 20 (interval: 10)

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
   New parameters: ~150,000,000  ← ACTUAL COMPRESSION!
   Reduction: 38.5%  ← SUCCESS!
   [OK] Optimizer reinitialized
   [OK] GPU memory cache cleared
```

## Interactive Controls

### PAUSE Training

**To pause after current epoch:**

```powershell
# From PowerShell (NO .txt extension! Just "PAUSE")
echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"

# OR manually:
# 1. Navigate to D:\ML_Results\dendritic_whisper\test_35_hybrid\
# 2. Right-click → New → Text Document
# 3. Name it exactly: PAUSE (remove the .txt extension!)
# 4. Windows will warn "changing extension" - click Yes
```

**What happens:**
- Training completes current epoch
- Saves checkpoint
- Exits cleanly
- Shows message with resume instructions

**To resume:**
```bash
.\RUN_TEST_35_HYBRID.bat
# Automatically loads checkpoint and continues
```

### Force Compression Manually

**To trigger compression at any time:**

```powershell
# From PowerShell (NO .txt extension! Just "FORCE_COMPRESS")
echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"

# OR manually:
# 1. Navigate to D:\ML_Results\dendritic_whisper\test_35_hybrid\
# 2. Right-click → New → Text Document
# 3. Name it exactly: FORCE_COMPRESS (remove the .txt extension!)
# 4. Windows will warn "changing extension" - click Yes
```

**What happens:**
- Next epoch: Compression triggers
- Works alongside hybrid triggers
- Useful if you see a good WER and want to compress immediately

**Example use case:**
```
Epoch 15: WER 17.5% (best so far!)
→ Create FORCE_COMPRESS file
→ Epoch 16: Compression triggers
→ Skip waiting for epoch 20
```

## Success Criteria

### Minimum Success ✅
- Compression happens at epochs 10 and 20
- Parameters: 240M → 467M → ~150M
- Final WER < 30%
- No crashes

### Good Success ✅✅
- Final WER < 25%
- Post-compression speed stable (3-4 it/s)
- Memory stays below 25GB reserved
- Compressed model trains normally

### Excellent Success ✅✅✅
- Final WER < 20% (within 2% of pre-compression best)
- Parameters < 150M (>38% compression)
- Ready for edge deployment
- Publishable results!

## Monitoring

### Check Progress

```powershell
# View latest checkpoint
cat D:\ML_Results\dendritic_whisper\test_35_hybrid\final_results.json

# Check if still running
nvidia-smi
# Look for python.exe using GPU
```

### Key Metrics to Watch

**Pre-compression (Epochs 1-10):**
- Speed: 5.5-5.9 it/s
- Memory: 4.5-4.6 GB allocated
- WER: Improving from 24% toward 18%

**Post-restructuring (Epochs 11-20):**
- Speed: 3.5-3.7 it/s (slower, expected)
- Memory: 6.3-6.5 GB allocated
- Reserved: <25 GB (if >30GB, memory leak!)
- WER: 19-22%

**Post-pruning (Epochs 21-35):**
- Speed: Should be similar or faster than epochs 11-20
- Memory: ~4-5 GB allocated (less than dendrite phase)
- WER: <25% (ideally <20%)

## Troubleshooting

### Training Seems Stuck
- Check nvidia-smi for GPU utilization (should be 70-85%)
- Check Task Manager for CPU usage
- If frozen >5 min, something's wrong

### Want to Stop Early
- Create PAUSE file
- Or Ctrl+C (loses current epoch progress)

### Compression Not Happening
- Check logs for "HYBRID MODE: FORCE TRIGGER" messages
- Should see at epochs 10, 20, 30
- If missing, hybrid mode might not be enabled

### Crashes or Errors
- Check logs in console
- GPU memory full? (check nvidia-smi)
- Disk space? (need ~10GB for checkpoints)

## After Completion

### Analyze Results

```powershell
# View final results
cat D:\ML_Results\dendritic_whisper\test_35_hybrid\final_results.json

# Check parameter reduction
# Look for:
# - Final parameters: ~150M (goal: <160M)
# - Final WER: ~18-22% (goal: <25%)
```

### Compare to Baseline

**Whisper-Small baseline:**
- Parameters: 240M
- WER on LibriSpeech: ~16-18% (OpenAI benchmark)

**Your compressed model:**
- Parameters: ~150M (38% reduction)
- WER: [Your result]
- Degradation: [Your WER - 18%]

**Acceptable degradation: <5%** (e.g., 18% → 23%)

### Next Steps

**If compression successful (<25% WER):**
1. Test inference speed on compressed model
2. Try more aggressive compression (max_dendrites=2)
3. Add attention layer compression (244M → 90M possible)

**If WER too high (>25%):**
1. Try longer training (50 epochs)
2. Try less aggressive compression (max_dendrites=4)
3. Reduce compression frequency (intervals of 15 instead of 10)

## Configuration Reference

```bash
--max-epochs 35                  # Total training epochs
--compression-mode hybrid        # Force + natural triggers
--force-trigger-interval 10      # Force every 10 epochs
--n-epochs-to-switch 3          # Or natural after 3 plateau epochs
--max-dendrites 3               # Dendrite count (lower = more compression)
--batch-size 8                  # Conservative for stability
--num-workers 0                 # Windows safe (no multiprocessing)
```

## File Locations

**Results directory:**
```
D:\ML_Results\dendritic_whisper\test_35_hybrid\
├── best_model.pth              # Best checkpoint (lowest WER)
├── pai_tracker.pth             # PAI compression state
├── switch_epochs.csv           # When compressions happened
├── training_log.csv            # WER/loss per epoch
├── final_results.json          # Summary results
├── PAUSE                       # Create this to pause
└── FORCE_COMPRESS              # Create this to force compression
```

## Tips

1. **Run overnight** - 7.5 hours is perfect for a sleep cycle
2. **Monitor first 2 hours** - Make sure epochs 1-10 complete successfully
3. **Check epoch 10** - Verify compression triggers and memory clearing works
4. **Don't worry about speed** - Post-compression slowdown is expected and fixed by memory clearing
5. **Trust the process** - Hybrid mode WILL compress at epoch 20, even if model improving

Good luck! 🚀
