# Test 1: Conservative MLP-Only Compression - User Guide

## Quick Start

### To run the test:
```bash
# Option 1: Use the batch script (Windows)
cd whisper-edge-optimization\dendritic
RUN_TEST_1.bat

# Option 2: Run directly with Python
cd whisper-edge-optimization\dendritic
python train_dendritic_full.py --device cuda --save-name test_1_conservative_mlp_only --val-max-samples 100 --max-epochs 25 --batch-size 8 --do-training --use-amp --amp-dtype bfloat16 --n-epochs-to-switch 3 --max-dendrites 3 --data-dir "D:\ML_Datasets\LibriSpeech" --results-dir "D:\ML_Results\dendritic_whisper"
```

## What to Watch For

### 1. Startup Messages (First 2 minutes)
Look for these confirmations:
```
[2/6] Configuring Perforated AI...
      Converting 48 MLP layers (encoder + decoder)
      Tracking all other Linear layers (attention, embeddings, projections)
      Max dendrites: 3
      Compression trigger: 3 epochs without improvement

[5/6] Starting training loop...
      Mixed Precision: BF16
```

**✓ GOOD:** If you see "Converting 48 MLP layers" and "Mixed Precision: BF16"
**✗ BAD:** If it says "Converting 192 layers" (means selective conversion failed)

### 2. Memory Usage (Every epoch)
```
[GPU Memory - Epoch N after training]
  Allocated: X.XX GB / Y.YY GB reserved
  Peak this run: X.XX GB
  Total available: 32.00 GB (XX% used)
```

**✓ GOOD:** Allocated <10 GB (comfortable)
**⚠ WARNING:** Allocated >15 GB (higher than expected, but safe)
**✗ CRITICAL:** Allocated >28 GB (approaching limit)

### 3. Training Progress (Every epoch, ~23 minutes each)
```
Epoch X/25
Train loss: 0.XXXX
Running validation...
Validation WER: XX.XX%
Validation Accuracy: XX.XX%
```

**✓ GOOD:** Train loss decreasing smoothly, WER improving or stable
**⚠ WARNING:** WER >150% (model hallucinating, but might stabilize)
**✗ BAD:** Train loss = NaN or increasing (training diverged)

### 4. Compression Triggers (Watch for this!)
```
*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
Parameter change: 244M → XXXM (XX.X% reduction)
```

**This is the key moment!**

**First occurrence:** Around Epoch 4-5 (if WER plateaus)
**Expect:** ~3-5% parameter reduction per cycle
**Goal:** At least 1 restructuring, ideally 2-3

### 5. Parameter Count Tracking
```
Baseline parameters: 244,582,912
Final parameters: XXX,XXX,XXX
Reduction: X.X%
```

**Target range:**
- **Conservative success:** 5-10% reduction (220-232M params)
- **Good success:** 10-15% reduction (208-220M params)
- **Excellent:** >15% reduction (<208M params)

## How to Monitor

### Option 1: Watch the console
Just let the script run and check periodically. Key milestones:
- **Epoch 1:** Verify BF16 and 48 layers
- **Epoch 3-5:** Look for first compression
- **Epoch 10:** Check mid-test progress
- **Epoch 25:** Final results

### Option 2: Check log files
Results are saved to: `D:\ML_Results\dendritic_whisper\test_1_conservative_mlp_only\`

Files created:
- `final_results.json` - Summary results
- `test_1_conservative_mlp_only.png` - Training graph
- `test_1_conservative_mlp_onlyScores.csv` - WER history
- `test_1_conservative_mlp_onlyparam_counts.csv` - Parameter tracking
- `test_1_conservative_mlp_onlyswitch_epochs.csv` - Compression events

### Option 3: Kill and restart safely
If you need to stop the test:
1. Press Ctrl+C in the console
2. The last checkpoint is saved at: `D:\ML_Results\dendritic_whisper\test_1_conservative_mlp_only\best_model.pt`
3. You can resume by... (actually, PAI doesn't support resume - would need to restart)

**Note:** Stopping mid-test means losing progress. Best to let it run overnight.

## Expected Timeline

```
Start:                   0:00
  ↓ (Initialization)
Epoch 1:                 0:05  - Baseline established
  ↓ (23 min/epoch)
Epoch 3:                 1:00  - WER might plateau
  ↓
Epoch 4-5:              ~1:30  - First compression trigger expected
  ↓
Epoch 10:                4:00  - Mid-test checkpoint
  ↓
Epoch 15:                6:00  - Possible 2nd compression
  ↓
Epoch 20:                8:00  - Possible 3rd compression
  ↓
Epoch 25:                9:30  - Test complete
```

## Interpreting Results

### Scenario 1: Conservative Success (Target)
```
Final parameters: 232M (5% reduction)
Final WER: 8-12% (decent quality)
Compressions: 1-2 cycles
Memory peak: 6-8 GB
```
**Interpretation:** Conservative approach works! MLPs compressed safely. Ready for Test 2 (more aggressive).

### Scenario 2: Good Success
```
Final parameters: 220M (10% reduction)
Final WER: 8-12%
Compressions: 2-3 cycles
Memory peak: 6-8 GB
```
**Interpretation:** Better than expected! MLPs highly compressible. Proceed to Test 2 with confidence.

### Scenario 3: Minimal Success
```
Final parameters: 240M (1-2% reduction)
Final WER: 8-12%
Compressions: 0-1 cycles
Memory peak: 6-8 GB
```
**Interpretation:** Very conservative, but safe. Either:
- Increase n_epochs_to_switch to 5 (let model train longer before compressing)
- Or proceed to Test 2 with more layers

### Scenario 4: WER Degradation
```
Final parameters: 230M (6% reduction)
Final WER: >20% (much worse than baseline)
Compressions: 2-3 cycles
Memory peak: 6-8 GB
```
**Interpretation:** Compression too aggressive for quality. Next steps:
- Try decoder-only compression
- Or increase improvement_threshold
- Or reduce max_dendrites to 1

### Scenario 5: Memory Issues
```
Memory peak: >20 GB
[WARNING] messages in log
Possible crash
```
**Interpretation:** Memory higher than expected. Next steps:
- Reduce batch size to 4
- Enable gradient checkpointing
- Or try decoder-only

### Scenario 6: No Compression Triggers
```
Final parameters: 244M (0% reduction)
No "MODEL RESTRUCTURED" messages
WER keeps improving every epoch
```
**Interpretation:** Model never plateaued. Options:
- Re-run with longer epochs (40+)
- Use fixed-epoch trigger instead of history-based
- Or manually trigger compression

## What If...

### Q: The test crashes at epoch 0?
**A:** Check these:
1. Is CUDA available? (`nvidia-smi` in terminal)
2. Are the dataset paths correct? (D:\ML_Datasets\LibriSpeech)
3. Is there disk space? (Need ~10GB on D:)
4. Check error message - might be import error or path issue

### Q: Memory usage is higher than expected (>15GB)?
**A:** This might mean:
1. BF16 not actually enabled (check logs for "Mixed Precision: BF16")
2. Batch size too large (try --batch-size 4)
3. Or RTX 5090 is caching aggressively (not dangerous, just using available RAM)

**Action:** Only worry if you see "CRITICAL" warnings or OOM errors. Otherwise let it run.

### Q: WER starts at >100% and stays there?
**A:** This is actually NORMAL for the first few epochs! Whisper is learning from scratch for this specific task. Watch for:
- Does train loss decrease? (YES = learning is happening)
- Does WER eventually drop below 100%? (by epoch 5-10)
- If WER still >100% at epoch 15, might be an issue

### Q: Train loss becomes NaN?
**A:** Training diverged. Possible causes:
1. Learning rate too high (try 5e-6 instead of 1e-5)
2. Gradient explosion (but we have clipping, so unlikely)
3. BF16 numerical instability (try --amp-dtype float16)

**Action:** Stop test, adjust settings, restart.

### Q: No compression after 10 epochs?
**A:** Check PAI logs for why:
```
Checking PAI switch with mode n, switch mode DOING_HISTORY,
epoch X, last improved epoch Y, total epochs X, n: 3
Returning False - no triggers to switch have been hit
```

This means WER improved too recently. Either:
- Let it run longer (might trigger at epoch 15+)
- Or WER is still improving (good problem to have!)

### Q: How do I know if it's "done"?
**A:** Test is done when you see:
```
[6/6] Final evaluation...
FINAL RESULTS
Baseline parameters: 244,582,912
Final parameters: XXX,XXX,XXX
```

Then the script will pause. Press any key to exit.

## Files to Keep

After test completes, these files are important:
1. `final_results.json` - Main summary
2. `best_model.pt` - Compressed model checkpoint (1-2 GB)
3. `*.csv` files - Detailed logs
4. `TEST_1_IMPLEMENTATION_SUMMARY.md` - What we changed
5. This file - How to interpret results

You can delete:
- `latest.pt` (duplicate of best_model.pt)
- `*.png` graphs (can regenerate from CSVs)

## Next Steps After Test 1

### If successful (≥5% reduction, good WER):
→ **Run Test 2: Aggressive Compression**
- Add decoder cross-attention layers
- Add attention output projections
- Target: 15-25% reduction

### If minimal success (1-5% reduction):
→ **Adjust and retry:**
- Increase n_epochs_to_switch to 5
- Or run longer (40 epochs)
- Or try fixed-epoch trigger

### If WER degrades (>20% final WER):
→ **Try decoder-only:**
- Compress only decoder MLPs
- Freeze encoder entirely
- More conservative but safer

### If memory issues:
→ **Enable checkpointing:**
- Add gradient checkpointing
- Reduce batch size to 4
- Or try FP16 instead of BF16

## Support

If you encounter issues:
1. Check the error message carefully
2. Look in the CSV files for clues (`Scores.csv`, `param_counts.csv`)
3. Compare to TEST_1_IMPLEMENTATION_SUMMARY.md
4. Share the error message + last 50 lines of console output

## Technical Details

### Why these settings?
- **48 MLP layers:** High redundancy (4× expansion), safe to compress
- **BF16:** Memory savings with no quality loss
- **n=3:** Faster compression trigger (vs default n=10)
- **max_dendrites=3:** Limits cycles to prevent over-compression
- **25 epochs:** Long enough for 2-3 compressions

### What's not being compressed?
- Embeddings (token_embed, positional_embed) - Non-redundant
- Attention Q/K/V projections - Information bottlenecks
- Attention output projections - (might add in Test 2)
- Conv layers - Minimal params, critical for audio
- LayerNorm - Already minimal

### Memory breakdown (with BF16):
```
Base model:      244M × 2 bytes = 0.49 GB
Candidates:      198M × 2 bytes = 0.39 GB (temporarily)
Optimizer (FP32): 442M × 8 bytes = 3.54 GB
Activations:                       1-1.5 GB
Total:                            ~5.5-6 GB
```

### Why RTX 5090 is perfect:
- 32GB VRAM (test uses ~6GB = 19%)
- BF16 Tensor Cores (fast mixed precision)
- Large batch sizes possible (could go up to 32)
- Multiple tests can run in parallel if needed

---

**Good luck! Let me know when you start the test and I'll monitor progress with you.**

Remember: Even "minimal" success (5% reduction) proves the concept works. We can iterate from there!
