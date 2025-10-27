# Optimization TODO - Before MI300X Arrival

## Completed ✅
- [x] Fix validation crash (whisper.decode() incompatibility)
- [x] Test 1 running with history-based compression (n=3)
- [x] Verify dendrite initialization works (started at 17.64% WER, not 117%)
- [x] Confirm GPU stability and memory monitoring

## Pending Optimizations (Test Tomorrow After Test 1 Completes)

### 1. Memory Management
**Issue:** Training slowdown after restructuring (4.04 it/s → 1.97 it/s)
**Hypothesis:** Memory fragmentation from multiple restructuring operations
**Fix:** Add `torch.cuda.empty_cache()` after model restructuring

**Where to add:**
```python
# train_dendritic_full.py line ~663
# After "Optimizer reinitialized"

# Clear GPU memory cache to reduce fragmentation
torch.cuda.empty_cache()
print("   [OK] GPU memory cache cleared")
```

**Expected impact:**
- Reduce reserved memory bloat (45.76GB → lower)
- Potentially improve training speed
- May or may not help (need to test)

**Test plan:**
- Run 15-epoch test with memory clearing
- Compare epoch times to Test 1 baseline
- Measure reserved vs allocated memory over time

---

### 2. Dendrite Initialization Strategy
**Issue:** WER starts at 14.95% but degrades to 19.76% after first restructuring
**Hypothesis:** Random dendrite initialization disrupts learned patterns
**Potential fixes to explore:**

**Option A: Better initialization**
- Initialize dendrites near-zero (minimal disruption to pre-trained weights)
- Would require PAI source code modification

**Option B: Warm-start dendrites**
- Train for fewer epochs before first compression (n=2 instead of n=3)
- Give model less time to overfit before adding dendrites

**Option C: Use fixed-epoch trigger instead of history-based**
- Test 1b already prepared with fixed triggers at epochs 8, 16, 24
- More predictable, guaranteed compressions

**Test plan:**
- Run Test 1b (fixed-epoch trigger) to compare
- Evaluate if fixed schedule produces better WER
- Document trade-offs

---

### 3. Compression Strategy - More Layers
**Issue:** Only compressing 48 MLP layers (25% of model)
**Goal:** Get to sub-100M params (currently expecting ~150M with MLP-only)

**Option A: Add Attention Layers** (RECOMMENDED)
- Compress attention Q/K/V/Out projections (96 additional layers)
- Total: 144 layers compressed instead of 48
- Expected params: 244M → 90-110M

**Option B: More aggressive dendrite pruning**
- Use max_dendrites=2 instead of 3
- Higher compression ratio per layer
- Risk: May degrade WER significantly

**Test plan:**
- Create Test 2 script with MLP + Attention compression
- Run 15-18 epochs to see if it works
- Compare WER degradation vs parameter reduction

---

### 4. Training Efficiency
**Issue:** 28,539 training samples = slow epochs even without compression
**Potential improvements:**

**Option A: Increase dataloader workers**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,  # Currently defaults to 0
    pin_memory=True
)
```

**Option B: Pre-compute mel spectrograms**
- Cache spectrograms to disk
- Load pre-computed features instead of computing on-the-fly
- Significant speedup for repeated epochs

**Option C: Larger batch size**
- Currently: batch_size=8
- Could try: batch_size=16 or 32 (you have plenty of VRAM)
- Faster throughput, may slightly change convergence

**Test plan:**
- Add num_workers=4 and pin_memory=True
- Test with batch_size=16
- Measure epoch time improvement

---

### 5. Validation Efficiency
**Issue:** Validation taking 5+ seconds per batch after restructuring
**Hypothesis:** Autoregressive greedy decoding is slow with 467M params

**Option A: Reduce validation samples**
- Currently: 100 samples
- Try: 50 samples for faster iteration
- Less precise WER estimate but faster feedback

**Option B: Optimize greedy decoding**
- Current implementation generates token-by-token
- Could add early stopping when all sequences hit EOT
- Could use shorter max_length (224 → 100 tokens)

**Test plan:**
- Try --val-max-samples 50
- Measure validation time reduction
- Check if WER estimates are still reliable

---

### 6. Perforated Backpropagation
**Issue:** Not using PB (perforatedbp module not installed)
**Potential benefit:** 10-30% training speedup, 20-40% memory reduction

**Investigation needed:**
- Check if perforatedbp is available in PAI repo
- Test installation and compatibility
- Measure actual speedup (may not be significant)

**Priority:** LOW (not critical, current training works)

---

## Test Priority Order (Tomorrow)

### High Priority (Must Test):
1. ✅ **Memory clearing optimization** (quick test, likely helps)
2. ✅ **Test 1b - Fixed-epoch trigger** (already prepared, just run it)
3. ✅ **Dataloader workers + batch size** (easy win for speed)

### Medium Priority (If Time Permits):
4. **Test 2 - MLP + Attention compression** (needed for sub-100M target)
5. **Validation sample reduction** (helps iteration speed)

### Low Priority (Future Optimization):
6. **Perforated Backpropagation** (requires investigation)
7. **Pre-compute mel spectrograms** (significant effort)
8. **Dendrite initialization** (requires PAI source modification)

---

## Expected Timeline (Tuesday - Before MI300X)

**Morning (After Test 1 completes):**
- Analyze Test 1 final results
- Document WER progression and compression behavior
- Identify which optimizations are most important

**Afternoon:**
- Add memory clearing to both scripts
- Add dataloader optimizations
- Run Test 1b (fixed-epoch, 15 epochs, ~7 hours)

**Evening:**
- Analyze Test 1b results
- Prepare Test 2 script (MLP + Attention) if needed
- Document findings and recommendations

**Wednesday Morning:**
- Have optimized scripts ready for MI300X
- Know exactly what works and what doesn't
- Ready for rapid iteration at 2-4x speed

---

## Success Criteria for Tomorrow

**Minimum Success:**
- ✅ Test 1 completes without crashes
- ✅ Compression eventually happens (467M → ~150M)
- ✅ Post-compression WER < 25%

**Good Success:**
- ✅ Memory clearing improves training speed
- ✅ Test 1b completes with 3 compressions
- ✅ Post-compression WER < 20%

**Excellent Success:**
- ✅ WER stays < 18% after compression
- ✅ Clear path to sub-100M params identified
- ✅ All optimizations tested and documented

---

## Notes

- All tests should use the FIXED validation function (greedy decoding)
- Keep saving checkpoints in case of crashes
- Monitor GPU temps and utilization
- Document everything for MI300X reproduction
- Don't worry about speed - MI300X will be much faster anyway
