# Real-Time Training Monitor Guide

## Quick Check (What You Want)

**Check current status:**
```bash
cd /Users/bledden/Documents/dendritic-hackathon
./monitor_training.sh test_run
```

**Live tail the training (most useful):**
```bash
# Find the Python process ID
ps aux | grep train_dendritic_full.py

# The output is currently in memory, but you can check if training started
tail -f PerforatedAI/*.csv
```

---

## Current Training Status

**Test Run**: `test_run`
- **Status**: 🟡 Dataset downloading (second dataset: 12/64 files, ~19%)
- **Expected start**: ~5 minutes (datasets need to finish loading)
- **Shell ID**: 188794

**To check RIGHT NOW in terminal:**
```bash
# See if process is still running
ps aux | grep train_dendritic_full.py

# Check dataset download progress - look for "Downloading data: XX%"
# (Currently visible in my monitoring)
```

---

## Monitoring Options

### Option 1: Quick Status Check (Recommended)
```bash
cd /Users/bledden/Documents/dendritic-hackathon
./monitor_training.sh test_run
```

This shows:
- ✅ Is training running?
- ✅ Parameter reduction history
- ✅ Final results (when complete)

### Option 2: Watch PAI CSV Files
```bash
# Watch parameter reduction in real-time
watch -n 5 'cat /Users/bledden/Documents/dendritic-hackathon/PerforatedAI/bestTestScore.csv 2>/dev/null || echo "Waiting for training to start..."'
```

**What to look for:**
- File appears when first dendrite cycle completes
- Shows parameter count reduction over time
- Updates after each restructuring

### Option 3: Check Process Activity
```bash
# See if training is running and using CPU
top -pid $(pgrep -f "train_dendritic_full.py.*test_run")
```

**What to look for:**
- CPU usage should be high (>80%) during training
- If low CPU for extended time, might be downloading data

### Option 4: Watch for "Restructured" Events
```bash
# This will show when dendrites are added (after training starts)
watch -n 10 'ls -lth /Users/bledden/Documents/dendritic-hackathon/PerforatedAI/ | head -10'
```

**What to look for:**
- New `.csv` files appearing
- Files being updated (timestamps change)

---

## What You'll See During Training

### Phase 1: Dataset Loading (CURRENT - ~5 more minutes)
```
Downloading data:  19%|█▉        | 12/64 [01:37<06:53,  7.96s/files]
```

### Phase 2: First Epochs (Baseline - ~10 minutes)
```
======================================================================
Epoch 1/10
======================================================================

Running validation...
Validation WER: 3.45%
Validation Accuracy: 96.55%

Updating PAI tracker...
```

### Phase 3: First Dendrite Addition (Expected - ~20 minutes)
```
🌳 MODEL RESTRUCTURED! Dendrites added/incorporated.
   New parameters: 228,553,769
   Reduction: 5.0%
   ✅ Optimizer reinitialized
```

### Phase 4: Completion (Expected - ~60 minutes)
```
🎉 TRAINING COMPLETE!
   PAI has determined optimal dendrite configuration.
   Best model has been loaded automatically.

======================================================================
📊 FINAL RESULTS
======================================================================
Baseline parameters: 240,582,912
Final parameters: 216,524,620
Reduction: 10.0%
Final WER: 3.52%
```

---

## Key Files to Watch

### Training Progress:
- **PAI CSV files**: `/Users/bledden/Documents/dendritic-hackathon/PerforatedAI/*.csv`
  - `bestTestScore.csv` - Parameter reduction over time
  - `correlation_values.csv` - Dendrite learning metrics

### Results:
- **Final results**: `/Users/bledden/Documents/dendritic-hackathon/whisper-edge-optimization/results/test_run/final_results.json`
- **Status doc**: `/Users/bledden/Documents/dendritic-hackathon/TEST_RUN_STATUS.md`

---

## Troubleshooting

### Training seems stuck?
```bash
# Check if process is alive
ps aux | grep train_dendritic_full.py

# If running, check CPU usage
top -pid $(pgrep -f "train_dendritic_full.py")

# Low CPU = probably downloading dataset or doing validation
```

### Kill training if needed:
```bash
pkill -f "train_dendritic_full.py.*test_run"
```

### Check for errors:
```bash
# Python process output is in background shell (188794)
# I can check it for you, or look for crash indicators:
ls -lth /Users/bledden/Documents/dendritic-hackathon/whisper-edge-optimization/results/test_run/
```

---

## Timeline Expectations

| Time | Phase | What's Happening |
|------|-------|------------------|
| **0-10 min** | 🔵 Dataset Download | LibriSpeech downloading (48+64 files) |
| **10-15 min** | 🟢 Epoch 1-2 | Baseline validation, establishing performance |
| **15-30 min** | 🟡 Epoch 3-5 | First dendrite addition expected |
| **30-60 min** | 🟠 Epoch 6-10 | Second dendrite cycle, refinement |
| **60 min** | ✅ Complete | Final results, model saved |

**Current**: ~15 minutes in, still downloading second dataset

---

## What to Expect

### Success Indicators:
- ✅ At least one "🌳 MODEL RESTRUCTURED!" message
- ✅ Parameter count decreases from 240,582,912
- ✅ WER stays below 10% (ideally below 5%)
- ✅ Multiple dendrite cycles (target: 2)

### If You See These (Good!):
- "Building dendrites without Perforated Backpropagation" - Normal, tokens not loaded in background
- High CPU usage - Training is working
- CSV files updating - PAI is tracking properly

### If You See These (Investigate):
- Process crashes - Check error messages
- 0% reduction after 5 epochs - Configuration issue
- WER > 20% - Model degrading, might need to adjust hyperparameters

---

## Next Steps After Test Completes

If successful, launch full training:
```bash
cd /Users/bledden/Documents/dendritic-hackathon/whisper-edge-optimization/dendritic
source ../../venv/bin/activate
source ../../.env

python train_dendritic_full.py \
  --save-name initial_run \
  --val-max-samples-per-epoch 200 \
  --max-epochs 30 \
  --max-dendrites 5 \
  --batch-size 16
```

Monitor with:
```bash
./monitor_training.sh initial_run
```
