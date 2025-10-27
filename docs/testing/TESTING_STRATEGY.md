# Dendritic Whisper - Comprehensive Testing Strategy

**Created:** October 25, 2025
**Project Goal:** Validate dendritic compression on Whisper Small (240M → 98M params, 60% reduction)
**Target WER:** ≤4% on LibriSpeech test-clean

---

## Testing Philosophy

**Why This Approach:**
1. **Establish baselines first** - Need control measurements before testing compression
2. **Controlled variables** - Test one thing at a time (CPU vs GPU, then baseline vs dendritic)
3. **Reproducible results** - Document everything for scientific validity
4. **Parallel execution** - Run CPU and GPU simultaneously to save time (when GPU available)
5. **Comprehensive metrics** - Track WER, parameters, speed, memory, compression ratio

---

## Test Matrix Overview

| Phase | Test Name | Device | Dendrites | Dataset Size | Purpose |
|-------|-----------|--------|-----------|--------------|---------|
| **1A** | CPU Baseline Validation | CPU | ❌ No | 10 samples | Environment verification (IN PROGRESS) |
| **1B** | GPU Baseline Validation | GPU | ❌ No | 10 samples | GPU setup verification (BLOCKED: PyTorch sm_120) |
| **2A** | CPU Baseline Full | CPU | ❌ No | 100 samples | Baseline WER measurement |
| **2B** | GPU Baseline Full | GPU | ❌ No | 100 samples | Baseline WER + speed benchmark |
| **3A** | CPU Dendritic Training | CPU | ✅ Yes | 100 samples | Verify dendritic compression works |
| **3B** | GPU Dendritic Training | GPU | ✅ Yes | 100 samples | Dendritic compression + speed |
| **4A** | CPU Production (Optional) | CPU | ✅ Yes | Full dataset | Final validation (not recommended) |
| **4B** | GPU Production | GPU | ✅ Yes | Full dataset | Production training for research |

---

## Phase 1: Environment Verification

### Test 1A: CPU Baseline Validation ⏳ IN PROGRESS

**Status:** Running now
**Started:** October 25, 2025

**Hypothesis:**
- CPU mode should work without errors
- Validation will complete on 10 samples in ~5-10 minutes
- WER will be ~15-25% (baseline Whisper Small on small sample)
- No dendrites will be added (validation-only mode)

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cpu \
  --save-name test_1a_cpu_baseline \
  --val-max-samples 10 \
  --max-epochs 3 \
  --batch-size 2 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**What We're Testing:**
- ✅ Virtual environment activation
- ✅ All dependencies installed correctly
- ✅ Dataset downloads to D: drive
- ✅ LibriSpeech loads without errors
- ✅ Whisper model loads on CPU
- ✅ PAI initializes correctly
- ✅ Audio pipeline works (FLAC → mel spectrogram)
- ✅ Inference runs on CPU
- ✅ WER calculation works

**Expected Results:**
```json
{
  "test_id": "1A",
  "device": "cpu",
  "dendrites": false,
  "samples": 10,
  "expected_time": "5-10 minutes",
  "expected_wer": "15-25%",
  "expected_params": 240582912,
  "expected_warnings": ["Pydantic warnings (harmless)", "Building dendrites without Perforated Backpropagation"]
}
```

**Success Criteria:**
- ✅ No crashes or errors
- ✅ Completes within 15 minutes
- ✅ WER calculated successfully
- ✅ Results saved to disk

**Failure Cases:**
- ❌ Crashes during validation
- ❌ Hangs indefinitely
- ❌ Out of memory errors
- ❌ Dataset download fails

---

### Test 1B: GPU Baseline Validation ⚠️ BLOCKED

**Status:** Blocked - waiting for PyTorch sm_120 support
**Blocker:** RTX 5090 has CUDA capability sm_120 (Blackwell), PyTorch 2.6.0 only supports up to sm_90

**Hypothesis:**
- GPU mode should work 10-50x faster than CPU
- Same WER as CPU baseline (device shouldn't affect accuracy)
- Validation completes in ~30-60 seconds for 10 samples

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cuda \
  --save-name test_1b_gpu_baseline \
  --val-max-samples 10 \
  --max-epochs 3 \
  --batch-size 8 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**What We're Testing:**
- ✅ PyTorch CUDA detection
- ✅ Model loads to GPU
- ✅ GPU inference works
- ✅ Speed improvement vs CPU

**Unblocking Options:**
1. **Wait for PyTorch 2.7+** (check https://pytorch.org/get-started/locally/)
2. **Try PyTorch nightly builds** (may have sm_120 support)
3. **Use WSL2** (better GPU compatibility)

**When Unblocked - Expected Results:**
```json
{
  "test_id": "1B",
  "device": "cuda",
  "dendrites": false,
  "samples": 10,
  "expected_time": "30-60 seconds",
  "expected_wer": "15-25% (same as 1A)",
  "expected_params": 240582912,
  "expected_speedup": "10-50x vs CPU"
}
```

---

## Phase 2: Baseline Measurement (Without Dendrites)

**Purpose:** Establish control measurements before adding dendritic compression

### Test 2A: CPU Baseline Full

**Hypothesis:**
- Larger sample size (100) gives more accurate WER estimate
- Should match published Whisper Small benchmarks (~3-5% WER on LibriSpeech)
- Takes ~1-2 hours on CPU

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cpu \
  --save-name test_2a_cpu_baseline_100 \
  --val-max-samples 100 \
  --max-epochs 1 \
  --batch-size 4 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Expected Results:**
```json
{
  "test_id": "2A",
  "device": "cpu",
  "dendrites": false,
  "samples": 100,
  "expected_time": "60-120 minutes",
  "expected_wer": "3-5%",
  "expected_params": 240582912,
  "baseline": true
}
```

---

### Test 2B: GPU Baseline Full

**Hypothesis:**
- Same WER as CPU (device independence)
- Much faster execution (~5-10 minutes)
- Establishes GPU baseline for dendritic comparison

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cuda \
  --save-name test_2b_gpu_baseline_100 \
  --val-max-samples 100 \
  --max-epochs 1 \
  --batch-size 16 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Expected Results:**
```json
{
  "test_id": "2B",
  "device": "cuda",
  "dendrites": false,
  "samples": 100,
  "expected_time": "5-10 minutes",
  "expected_wer": "3-5%",
  "expected_params": 240582912,
  "baseline": true,
  "expected_speedup": "10-20x vs 2A"
}
```

---

## Phase 3: Dendritic Compression Testing

**⚠️ REQUIRES:** Training loop implementation (currently missing)

**What Needs to Be Added:**
1. Forward pass through encoder + decoder
2. Loss calculation (CTC or cross-entropy)
3. Backpropagation (loss.backward())
4. Optimizer step (optimizer.step())
5. Training dataset loader

**Reference:** See [DENDRITIC_RESEARCH_CONTEXT.md](DENDRITIC_RESEARCH_CONTEXT.md) lines 110-150

---

### Test 3A: CPU Dendritic Training

**Hypothesis:**
- PAI will add dendrites after detecting validation plateau
- Parameter count will decrease from 240M → ~150M (initial compression)
- WER will remain ≤5% (minimal accuracy loss)
- Training slow on CPU (~4-8 hours for 100 samples)

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cpu \
  --save-name test_3a_cpu_dendritic_100 \
  --val-max-samples 100 \
  --max-epochs 30 \
  --max-dendrites 5 \
  --batch-size 4 \
  --improvement-threshold 0.0001 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**What We're Testing:**
- ✅ PAI dendrite addition triggers
- ✅ Model restructuring works
- ✅ Compression achieved
- ✅ Accuracy maintained
- ✅ Training loop completes

**Expected Results:**
```json
{
  "test_id": "3A",
  "device": "cpu",
  "dendrites": true,
  "samples": 100,
  "expected_time": "4-8 hours",
  "expected_wer": "4-5%",
  "expected_params_before": 240582912,
  "expected_params_after": "144M-192M (40-60% compression)",
  "expected_dendrites_added": "2-4",
  "expected_epochs_to_first_dendrite": "5-10"
}
```

**Key Metrics to Track:**
- Epoch when first dendrite added
- Parameter count after each dendrite
- WER after each dendrite
- Compression ratio (1 - params_after/params_before)
- Time per epoch

---

### Test 3B: GPU Dendritic Training

**Hypothesis:**
- Same compression ratio as CPU
- Same WER as CPU (device independence)
- Much faster training (~30-60 minutes)
- Dendrites added at same epochs as CPU

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cuda \
  --save-name test_3b_gpu_dendritic_100 \
  --val-max-samples 100 \
  --max-epochs 30 \
  --max-dendrites 5 \
  --batch-size 16 \
  --improvement-threshold 0.0001 \
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Expected Results:**
```json
{
  "test_id": "3B",
  "device": "cuda",
  "dendrites": true,
  "samples": 100,
  "expected_time": "30-60 minutes",
  "expected_wer": "4-5% (same as 3A)",
  "expected_params_after": "144M-192M (same compression as 3A)",
  "expected_speedup": "6-10x vs 3A"
}
```

---

## Phase 4: Production Training (Full Dataset)

### Test 4B: GPU Production Training (Recommended)

**Only run after Tests 1-3 succeed!**

**Hypothesis:**
- Full dataset (28,539 samples) achieves target compression
- Final WER ≤4% on test-clean
- 60%+ parameter reduction (240M → ≤98M)
- Training time: 4-8 hours on GPU

**Configuration:**
```bash
python train_dendritic_full.py \
  --device cuda \
  --save-name production_gpu_full \
  --max-epochs 50 \
  --max-dendrites 10 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --improvement-threshold 0.0001 \
  --data-dir "D:\ML_Datasets\LibriSpeech" \
  --use-wandb
```

**Expected Results:**
```json
{
  "test_id": "4B",
  "device": "cuda",
  "dendrites": true,
  "samples": 28539,
  "expected_time": "4-8 hours",
  "expected_wer": "3-4%",
  "expected_params_before": 240582912,
  "expected_params_after": "96M-120M (50-60% compression)",
  "target_compression": "60%",
  "target_wer": "≤4%"
}
```

---

## Parallel CPU + GPU Testing

**Can we run both simultaneously?** ✅ **YES!**

### Method 1: Two Terminal Windows

**Terminal 1 (CPU):**
```powershell
cd C:\Users\blake\Documents\dendritic-hackathon\whisper-edge-optimization\dendritic
.\..\..\venv\Scripts\Activate.ps1
python train_dendritic_full.py --device cpu --save-name parallel_cpu --val-max-samples 100 --max-epochs 30 --batch-size 4 --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Terminal 2 (GPU - when available):**
```powershell
cd C:\Users\blake\Documents\dendritic-hackathon\whisper-edge-optimization\dendritic
.\..\..\venv\Scripts\Activate.ps1
python train_dendritic_full.py --device cuda --save-name parallel_gpu --val-max-samples 100 --max-epochs 30 --batch-size 16 --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Benefits:**
- ✅ 2x faster than sequential
- ✅ Direct comparison of same experiment
- ✅ Validates device independence

**Considerations:**
- ⚠️ CPU will be slower (GPU might finish first)
- ⚠️ High memory usage (need ~32GB RAM)
- ⚠️ Dataset cached on D: drive (shared, no duplicate download)
- ⚠️ Different save names prevent conflicts

### Method 2: Python Script for Parallel Launch

Create `run_parallel_tests.py`:
```python
import subprocess
import sys
from pathlib import Path

def run_test(device, save_name, batch_size):
    """Launch a test in a subprocess"""
    cmd = [
        sys.executable,
        "train_dendritic_full.py",
        "--device", device,
        "--save-name", save_name,
        "--val-max-samples", "100",
        "--max-epochs", "30",
        "--batch-size", str(batch_size),
        "--data-dir", r"D:\ML_Datasets\LibriSpeech"
    ]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    print("Launching parallel tests...")

    # Launch CPU test
    cpu_process = run_test("cpu", "parallel_cpu", 4)
    print("✅ CPU test launched")

    # Launch GPU test
    gpu_process = run_test("cuda", "parallel_gpu", 16)
    print("✅ GPU test launched")

    # Wait for both
    cpu_process.wait()
    gpu_process.wait()

    print("\n✅ Both tests completed!")
```

---

## Results Tracking System

### Test Results Template

Create `TEST_RESULTS.md` (will be populated as tests complete):

```markdown
# Test Results Log

## Test 1A: CPU Baseline Validation
- **Status:** IN PROGRESS
- **Started:** 2025-10-25 [TIME]
- **Completed:** [PENDING]
- **Duration:** [PENDING]
- **WER:** [PENDING]
- **Parameters:** [PENDING]
- **Notes:** [Any observations]

## Test 1B: GPU Baseline Validation
- **Status:** BLOCKED (PyTorch sm_120 support)
- **Blocker:** RTX 5090 compatibility
- **Resolution Plan:** Wait for PyTorch 2.7+ or try WSL2

[... continue for all tests ...]
```

### Automated Results Collection

Each test will save `final_results.json`:
```json
{
  "test_id": "1A",
  "timestamp": "2025-10-25T13:45:00",
  "device": "cpu",
  "dendrites_enabled": false,
  "samples": 10,
  "duration_seconds": 427,
  "wer": 18.5,
  "parameters_before": 240582912,
  "parameters_after": 240582912,
  "compression_ratio": 0.0,
  "dendrites_added": 0,
  "epochs_completed": 3,
  "best_wer_epoch": 1,
  "average_epoch_time": 142,
  "memory_peak_gb": 4.2,
  "hypothesis_met": true,
  "notes": "Completed successfully, warnings are harmless"
}
```

---

## Comparison Matrix (To Be Filled After Tests)

| Test | Device | Dendrites | WER | Params | Compression | Time | Speedup |
|------|--------|-----------|-----|--------|-------------|------|---------|
| 1A   | CPU    | ❌        | ?%  | 240M   | 0%          | ?    | 1x      |
| 1B   | GPU    | ❌        | ?%  | 240M   | 0%          | ?    | ?x      |
| 2A   | CPU    | ❌        | ?%  | 240M   | 0%          | ?    | 1x      |
| 2B   | GPU    | ❌        | ?%  | 240M   | 0%          | ?    | ?x      |
| 3A   | CPU    | ✅        | ?%  | ?M     | ?%          | ?    | 1x      |
| 3B   | GPU    | ✅        | ?%  | ?M     | ?%          | ?    | ?x      |
| 4B   | GPU    | ✅        | ?%  | ?M     | ?%          | ?    | -       |

**Target (Test 4B):**
- WER: ≤4%
- Params: ≤98M
- Compression: ≥60%

---

## Hypothesis Validation Checklist

### Environment Hypotheses:
- [ ] CPU mode works without errors (Test 1A)
- [ ] GPU mode works when PyTorch supports sm_120 (Test 1B)
- [ ] Dataset downloads to D: drive successfully
- [ ] External SSD performance adequate for training

### Baseline Hypotheses:
- [ ] CPU and GPU give same WER (device independence)
- [ ] 100 samples gives WER ~3-5% (matches published benchmarks)
- [ ] GPU is 10-50x faster than CPU

### Dendritic Hypotheses:
- [ ] PAI adds dendrites during training
- [ ] Compression achieved (≥40% for 100 samples)
- [ ] WER maintained (≤5% with compression)
- [ ] CPU and GPU give same compression ratio
- [ ] Dendrites added at same epochs on both devices

### Production Hypotheses:
- [ ] Full dataset achieves 60%+ compression
- [ ] Final WER ≤4%
- [ ] Training completes in reasonable time (<12 hours)

---

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU unavailable (PyTorch sm_120) | HIGH | MEDIUM | Use WSL2 or wait for PyTorch 2.7+ |
| Training loop missing | HIGH | HIGH | Implement before Phase 3 |
| Insufficient compression | MEDIUM | HIGH | Tune hyperparameters (max_dendrites, threshold) |
| WER degradation | MEDIUM | HIGH | Use larger dataset, adjust learning rate |
| Out of memory on CPU | LOW | MEDIUM | Reduce batch size |
| Slow training on CPU | HIGH | LOW | Expected, use GPU for production |
| Dataset download issues | LOW | LOW | Already downloaded to D: drive |

---

## Next Actions

### Immediate (While Test 1A Runs):
1. ✅ Monitor Test 1A completion
2. ✅ Document results when complete
3. ⏳ Decide: Implement training loop or wait for GPU?

### Short Term:
1. Complete Test 1A analysis
2. Check PyTorch nightly builds for sm_120 support
3. OR set up WSL2 for GPU testing
4. Implement training loop (if proceeding with development)

### Medium Term:
1. Run Tests 2A and 2B (baselines)
2. Implement and test dendritic training (Tests 3A/3B)
3. Run parallel CPU/GPU tests
4. Document all results

### Long Term:
1. Production training (Test 4B)
2. Write research paper
3. Publish results

---

**This strategy ensures:**
- ✅ Scientific rigor (controlled experiments)
- ✅ Reproducible results (documented everything)
- ✅ Comprehensive comparison (CPU vs GPU, baseline vs dendritic)
- ✅ Efficient execution (parallel testing where possible)
- ✅ Clear success criteria (hypothesis validation)

**Let's build the future of neural network compression!** 🚀
