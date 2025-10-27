# GPU Setup Options for RTX 5090 (sm_120 / Blackwell)

**Your Hardware:** NVIDIA GeForce RTX 5090 (32GB VRAM, CUDA capability sm_120)
**Current Status:** ⚠️ Blocked - PyTorch 2.6.0 only supports up to sm_90
**Goal:** Enable GPU acceleration for 10-50x speedup

**Last Updated:** October 25, 2025

---

## 🎯 Executive Summary

**GREAT NEWS!** PyTorch 2.7 **officially supports RTX 5090** with sm_120! 🎉

**Three viable options:**
1. **PyTorch 2.7 Stable** (RECOMMENDED) - Official support, CUDA 12.8
2. **PyTorch Nightly** - Bleeding edge, CUDA 12.8
3. **WSL2** - Linux environment, better compatibility

**Recommended:** Option 1 (PyTorch 2.7 Stable)

---

## Option 1: PyTorch 2.7 Stable with CUDA 12.8 ⭐ RECOMMENDED

### Status: ✅ AVAILABLE NOW

**Why This is Best:**
- ✅ Official stable release
- ✅ First PyTorch version with sm_120 support
- ✅ CUDA 12.8 includes Blackwell architecture support
- ✅ No experimental builds needed
- ✅ Works on Windows natively
- ✅ Fully tested and documented

### What's Included:
- **PyTorch 2.7.0** with sm_120 support
- **CUDA 12.8** - Required for Blackwell
- **cuDNN, NCCL, CUTLASS** upgraded for Blackwell
- **Triton 3.3** with torch.compile compatibility

### Installation Steps:

#### Step 1: Uninstall Current PyTorch
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Uninstall old PyTorch (CUDA 12.4)
pip uninstall torch torchvision torchaudio
```

#### Step 2: Install PyTorch 2.7 with CUDA 12.8
```powershell
# Install PyTorch 2.7 with CUDA 12.8
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Step 3: Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output:**
```
PyTorch: 2.7.0+cu128
CUDA: 12.8
CUDA Available: True
GPU: NVIDIA GeForce RTX 5090
```

#### Step 4: Test GPU Inference
```powershell
cd whisper-edge-optimization\dendritic

# Quick 10-sample GPU test
python train_dendritic_full.py `
  --device cuda `
  --save-name test_1b_gpu_baseline `
  --val-max-samples 10 `
  --max-epochs 3 `
  --batch-size 8 `
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Expected Time:** 30-60 seconds (vs 10-15 minutes on CPU)

### Pros:
- ✅ Official stable release
- ✅ Easiest installation
- ✅ Best long-term support
- ✅ Works on native Windows
- ✅ No experimental features

### Cons:
- ⚠️ Requires CUDA 12.8 (vs your current 12.9 - should be fine)
- ⚠️ Larger download (~2.5GB)

### Recommended For:
- Production use
- Long-term projects
- Users who want stability

---

## Option 2: PyTorch Nightly with CUDA 12.8

### Status: ✅ AVAILABLE (Experimental)

**Why Consider This:**
- Latest features and bug fixes
- May have performance improvements
- Supports sm_120 in nightly builds

**Why NOT Recommended:**
- Unstable (breaks occasionally)
- Not for production
- PyTorch 2.7 stable is better

### Installation (If Curious):
```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Pros:
- ✅ Cutting-edge features
- ✅ Latest optimizations

### Cons:
- ❌ Unstable
- ❌ May break
- ❌ Not recommended when stable version exists

### Recommended For:
- Developers testing new features
- Contributing to PyTorch
- Debugging edge cases

**Verdict:** Skip this - use PyTorch 2.7 stable instead

---

## Option 3: WSL2 (Windows Subsystem for Linux)

### Status: ✅ AVAILABLE (More Complex)

**Why Consider This:**
- Better Linux compatibility
- Some tools work better in Linux
- Good for cross-platform development

### What is WSL2?
- Runs full Linux kernel in Windows
- Near-native Linux performance
- CUDA passthrough to Linux
- Access to Linux tools and packages

### Performance:
- **Modern WSL2:** 90%+ of native Linux performance
- **GPU workloads:** Comparable to native in most cases
- **ML frameworks:** Better compatibility than native Windows

### Installation Steps:

#### Step 1: Install WSL2
```powershell
# In PowerShell as Administrator
wsl --install
```

This installs:
- WSL2 kernel
- Ubuntu (default)
- NVIDIA drivers (automatic)

#### Step 2: Restart Computer
Required for WSL2 activation

#### Step 3: Set Up Ubuntu
```bash
# WSL2 will auto-launch Ubuntu
# Create username and password

# Update packages
sudo apt update && sudo apt upgrade -y
```

#### Step 4: Install Python and Dependencies
```bash
# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3-pip git -y

# Clone your repo (or access via /mnt/c/Users/blake/...)
cd /mnt/c/Users/blake/Documents/dendritic-hackathon

# Create venv
python3.11 -m venv venv_wsl
source venv_wsl/bin/activate

# Install PyTorch 2.7 with CUDA 12.8
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install openai-whisper datasets transformers soundfile librosa tqdm jiwer pandas scikit-learn wandb python-dotenv

# Install PerforatedAI
cd PerforatedAI && pip install -e . && cd ..
```

#### Step 5: Test GPU
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

#### Step 6: Run Training
```bash
cd whisper-edge-optimization/dendritic

python train_dendritic_full.py \
  --device cuda \
  --save-name wsl2_test \
  --val-max-samples 10 \
  --max-epochs 3 \
  --batch-size 8 \
  --data-dir "/mnt/d/ML_Datasets/LibriSpeech"
```

### Pros:
- ✅ Better Linux tool support
- ✅ Cross-platform development
- ✅ Good performance (90%+)
- ✅ Access to Linux-native ML tools
- ✅ Better package compatibility

### Cons:
- ❌ More complex setup
- ❌ Requires learning Linux basics
- ❌ File system overhead (/mnt/c/ access slower)
- ❌ Two environments to maintain

### Recommended For:
- Users comfortable with Linux
- Cross-platform development
- When native Windows has issues
- Long-term Linux migration

**Verdict:** Good option, but PyTorch 2.7 on Windows is simpler

---

## Comparison Matrix

| Feature | PyTorch 2.7 (Win) | PyTorch Nightly | WSL2 |
|---------|-------------------|-----------------|------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Medium |
| **Stability** | ⭐⭐⭐⭐⭐ Stable | ⭐⭐ Unstable | ⭐⭐⭐⭐ Stable |
| **Performance** | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐ 90%+ |
| **sm_120 Support** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Windows Native** | ✅ Yes | ✅ Yes | ❌ No (Linux) |
| **Long-term Support** | ✅ Official | ⚠️ Experimental | ✅ Official |
| **Package Compatibility** | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Variable | ⭐⭐⭐⭐⭐ Excellent |
| **Maintenance** | ⭐⭐⭐⭐⭐ Low | ⭐⭐ High | ⭐⭐⭐ Medium |
| **Setup Time** | ~10 minutes | ~10 minutes | ~30-60 minutes |

---

## 🏆 Recommendation: PyTorch 2.7 Stable

**Why:**
1. ✅ **Official sm_120 support** - First stable release for RTX 5090
2. ✅ **Simple installation** - One pip command
3. ✅ **Native Windows** - No dual environments
4. ✅ **Stable and tested** - Production-ready
5. ✅ **Best performance** - Full native speed
6. ✅ **Easy maintenance** - Standard updates

**When to use alternatives:**
- **Nightly:** Never (unless you're a PyTorch developer)
- **WSL2:** If you need Linux-specific tools or prefer Linux workflow

---

## Step-by-Step: Recommended Installation (PyTorch 2.7)

### 1. Backup Current Environment (Optional)
```powershell
# If you want to keep current setup as fallback
xcopy venv venv_backup\ /E /I
```

### 2. Uninstall Old PyTorch
```powershell
.\venv\Scripts\Activate.ps1
pip uninstall torch torchvision torchaudio -y
```

### 3. Install PyTorch 2.7 + CUDA 12.8
```powershell
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Expected output:**
```
Collecting torch==2.7.0
  Downloading https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp313-cp313-win_amd64.whl (2.6 GB)
...
Successfully installed torch-2.7.0+cu128 torchvision-0.22.0+cu128 torchaudio-2.7.0+cu128
```

### 4. Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**
```
PyTorch: 2.7.0+cu128
CUDA: 12.8
GPU Available: True
GPU: NVIDIA GeForce RTX 5090
```

### 5. Test with Small Workload
```powershell
python -c "import torch; x = torch.rand(1000, 1000).cuda(); y = torch.matmul(x, x); print('GPU test passed!')"
```

### 6. Run Test 1B (GPU Baseline Validation)
```powershell
cd whisper-edge-optimization\dendritic

python train_dendritic_full.py `
  --device cuda `
  --save-name test_1b_gpu_baseline `
  --val-max-samples 10 `
  --max-epochs 3 `
  --batch-size 8 `
  --data-dir "D:\ML_Datasets\LibriSpeech"
```

**Expected:** Completes in 30-60 seconds (vs 10-15 minutes on CPU)

### 7. Update Documentation
- Mark Test 1B as ✅ COMPLETED in [TEST_RESULTS.md](TEST_RESULTS.md)
- Document WER and compare with Test 1A

---

## Troubleshooting

### "Could not find a version that satisfies the requirement torch==2.7.0"
```powershell
# Make sure you're using the correct index URL
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify internet connection
ping download.pytorch.org
```

### "CUDA is available but model still runs on CPU"
```python
# Explicitly move model to GPU
model = model.to('cuda')

# Or use device argument
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### "Out of memory" errors
```powershell
# Reduce batch size
python train_dendritic_full.py --device cuda --batch-size 4  # Instead of 8 or 16

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### GPU not detected after installation
```powershell
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Restart Python/terminal
```

---

## Performance Expectations

### Test 1B: GPU Baseline (10 samples)
- **CPU (Test 1A):** ~10-15 minutes
- **GPU (expected):** ~30-60 seconds
- **Speedup:** ~10-20x

### Test 2B: GPU Baseline (100 samples)
- **CPU (Test 2A):** ~1-2 hours
- **GPU (expected):** ~5-10 minutes
- **Speedup:** ~10-15x

### Test 3B: GPU Dendritic Training (100 samples, 30 epochs)
- **CPU (Test 3A):** ~4-8 hours
- **GPU (expected):** ~30-60 minutes
- **Speedup:** ~6-10x

### Test 4B: GPU Production (Full dataset)
- **CPU (not recommended):** Days
- **GPU (expected):** ~4-8 hours
- **Speedup:** ~50x+

---

## Next Steps After GPU Setup

1. ✅ **Verify GPU works** - Run Test 1B (10 samples)
2. ✅ **Compare with CPU** - Test 1A vs 1B (same WER expected)
3. ✅ **Run Test 2B** - 100 samples on GPU (~5-10 min)
4. ✅ **Parallel testing** - Run 2A (CPU) and 2B (GPU) simultaneously
5. ✅ **Implement training loop** - Enable dendritic compression
6. ✅ **Run Tests 3A/3B** - Dendritic compression on CPU and GPU
7. ✅ **Production training** - Test 4B on GPU

---

## Summary

**Your Best Path:**
1. **Install PyTorch 2.7 with CUDA 12.8** (~10 minutes)
2. **Run Test 1B** - Verify GPU works (~1 minute)
3. **Run Test 2B** - 100 sample baseline (~5-10 minutes)
4. **Move to training loop implementation** - Enable dendrites

**Total time to GPU-ready:** ~15-20 minutes

**Benefits:**
- ✅ 10-50x speedup
- ✅ Faster iteration
- ✅ Production-ready for research
- ✅ Same environment as Test 1A/2A (easy comparison)

**You'll be GPU-accelerated before Test 2A (CPU) finishes!** 🚀

---

**Ready to proceed?** The PyTorch 2.7 installation is straightforward and will unblock all future GPU tests!
