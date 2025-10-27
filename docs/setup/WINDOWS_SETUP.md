# Windows Setup Guide - Dendritic Whisper Compression

## Current Setup Status

✅ **Completed:**
- Python 3.13.5 installed
- Virtual environment created at `venv/`
- PyTorch 2.6.0+cu124 installed with CUDA 12.4 support
- All project dependencies installed (Whisper, datasets, transformers, etc.)
- PerforatedAI 2.0.4 library installed
- Environment variables configured in `.env`
- NVIDIA GeForce RTX 5090 GPU detected

---

## ⚠️ IMPORTANT: RTX 5090 Compatibility Issue

Your RTX 5090 GPU has **CUDA capability sm_120** (Blackwell architecture). The current stable PyTorch release (2.6.0) **does not fully support** this architecture yet.

**You will see this warning:**
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

### Solutions:

#### Option 1: Use CPU Mode (Recommended for Now)
The project works perfectly on CPU, just slower. For development and testing:
```bash
# Activate environment
venv\Scripts\activate.bat

# Run training on CPU
cd whisper-edge-optimization\dendritic
python train_dendritic_full.py --device cpu --val-max-samples 100
```

#### Option 2: Wait for PyTorch 2.7+ (Coming Soon)
PyTorch nightlies and upcoming 2.7 release will support sm_120. Check:
- https://pytorch.org/get-started/locally/
- https://github.com/pytorch/pytorch/releases

#### Option 3: Use WSL2 (Advanced)
If you need GPU acceleration now, consider using WSL2 with Ubuntu:
1. Install WSL2: `wsl --install`
2. Install Ubuntu from Microsoft Store
3. Follow the Linux setup instructions from [AMD_SETUP.md](./AMD_SETUP.md)

---

## Quick Start

### 1. Activate Environment
```bash
# PowerShell or CMD
venv\Scripts\activate.bat
```

### 2. Verify Installation
```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Whisper
python -c "import whisper; print('Whisper: OK')"

# Check PerforatedAI
python -c "from perforatedai import utils_perforatedai as UPA; print('PerforatedAI: OK')"
```

### 3. Run Test Training (CPU Mode)
```bash
cd whisper-edge-optimization\dendritic

# Quick test with 10 samples (5 minutes)
python train_dendritic_full.py ^
  --device cpu ^
  --save-name windows_test ^
  --val-max-samples 10 ^
  --val-max-samples-per-epoch 10 ^
  --max-epochs 5 ^
  --batch-size 2
```

**Note:** The `^` character is the line continuation in Windows CMD. In PowerShell, use `` ` `` instead.

---

## Environment Configuration

Your `.env` file has been configured with:

```bash
# PerforatedAI API Credentials
PAIEMAIL=hacker_token@perforatedai.com
PAITOKEN=MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg==
```

**Optional additions** (edit `.env` to add):
```bash
# Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_key_here

# HuggingFace token (usually not needed)
HF_TOKEN=your_hf_token_here
```

---

## Windows-Specific Differences

### 1. Virtual Environment Activation
- **Linux/Mac:** `source venv/bin/activate`
- **Windows CMD:** `venv\Scripts\activate.bat`
- **Windows PowerShell:** `venv\Scripts\Activate.ps1`

### 2. Line Continuation in Scripts
- **Bash:** Use `\` at end of line
- **Windows CMD:** Use `^` at end of line
- **PowerShell:** Use `` ` `` (backtick) at end of line

### 3. Path Separators
- **Linux/Mac:** `/` (forward slash)
- **Windows:** `\` (backslash) or `/` (also works in Python)

### 4. Shell Scripts
The `.sh` scripts in the repo are for Linux/Mac. Use the provided `.bat` scripts instead:
- ✅ `setup_windows.bat` (instead of `setup_amd.sh`)

---

## Installed Dependencies

### Core ML/DL
- ✅ torch 2.6.0+cu124
- ✅ torchvision 0.21.0+cu124
- ✅ torchaudio 2.6.0+cu124

### Whisper & NLP
- ✅ openai-whisper 20250625
- ✅ datasets 4.3.0
- ✅ transformers 4.57.1
- ✅ tokenizers 0.22.1

### Audio Processing
- ✅ soundfile 0.13.1
- ✅ librosa 0.11.0
- ✅ audioread 3.0.1

### ML Utilities
- ✅ numpy 2.3.3
- ✅ pandas 2.3.3
- ✅ scipy 1.16.2
- ✅ scikit-learn 1.7.2
- ✅ tqdm 4.67.1
- ✅ jiwer 4.0.0 (Word Error Rate)

### Experiment Tracking
- ✅ wandb 0.22.2
- ✅ python-dotenv 1.1.1

### Perforated AI
- ✅ perforatedai 2.0.4
- ✅ matplotlib 3.10.7 (for PAI visualizations)

---

## Testing the Installation

### Basic Import Tests
```bash
# Activate environment first
venv\Scripts\activate.bat

# Test all core imports
python -c "import torch, whisper, datasets, transformers, soundfile, librosa, pandas, wandb; from perforatedai import utils_perforatedai as UPA, globals_perforatedai as GPA; print('All imports successful!')"
```

### Whisper Model Loading (CPU)
```bash
python -c "import whisper; model = whisper.load_model('tiny'); print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')"
```

### GPU Detection
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

---

## Running Training

### CPU Mode (Works Now)
```bash
cd whisper-edge-optimization\dendritic

# Small test run
python train_dendritic_full.py ^
  --device cpu ^
  --save-name test_cpu ^
  --val-max-samples 50 ^
  --max-epochs 10 ^
  --batch-size 4
```

### GPU Mode (When PyTorch Supports sm_120)
```bash
# Same command, just change device
python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_gpu ^
  --val-max-samples 500 ^
  --max-epochs 30 ^
  --batch-size 16
```

---

## Troubleshooting

### "CUDA error: no kernel image is available"
- **Cause:** RTX 5090's sm_120 architecture not supported by PyTorch 2.6.0
- **Solution:** Use `--device cpu` for now, or wait for PyTorch 2.7+

### "ModuleNotFoundError: No module named 'perforatedai'"
```bash
# Reinstall PerforatedAI
cd PerforatedAI
pip install -e .
cd ..
```

### "FileNotFoundError: [Errno 2] No such file or directory: 'soundfile.dll'"
```bash
# Reinstall soundfile
pip install --force-reinstall soundfile
```

### WandB Login
```bash
# If using WandB for experiment tracking
wandb login
# Or set in .env: WANDB_API_KEY=your_key
```

### Dataset Download Issues
```bash
# LibriSpeech will download automatically on first run
# It's ~6GB for test.clean + train.clean.100
# Default location: ./data/
# To specify custom location, edit train_dendritic_full.py cache_dir parameter
```

---

## Performance Expectations

### CPU Mode (Your Current Setup)
- **Small test (100 samples):** ~10-20 minutes
- **Full validation (2620 samples):** ~2-4 hours
- **Full training:** Potentially days (not recommended)

### GPU Mode (When Available)
With RTX 5090 (once supported):
- **Small test (100 samples):** ~1-2 minutes
- **Full validation (2620 samples):** ~10-20 minutes
- **Full training:** ~2-6 hours

---

## Next Steps

1. **Test the installation**
   ```bash
   venv\Scripts\activate.bat
   cd whisper-edge-optimization\dendritic
   python train_dendritic_full.py --device cpu --val-max-samples 10 --max-epochs 3
   ```

2. **Monitor for PyTorch updates**
   - Check https://pytorch.org/get-started/locally/
   - Watch for PyTorch 2.7+ with sm_120 support

3. **Consider WSL2 for GPU acceleration**
   - If you need GPU now, WSL2 might support RTX 5090 better
   - Follow Ubuntu setup instructions

4. **Review the documentation**
   - [QUICKSTART.md](./QUICKSTART.md) - Next steps guide
   - [DENDRITIC_RESEARCH_CONTEXT.md](./DENDRITIC_RESEARCH_CONTEXT.md) - Full technical context
   - [PROGRESS_SUMMARY.md](./PROGRESS_SUMMARY.md) - Project status

---

## System Information

**Platform:** Windows 11
**Python:** 3.13.5
**PyTorch:** 2.6.0+cu124
**CUDA:** 12.4
**cuDNN:** 90100
**GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM)
**GPU Architecture:** Blackwell (sm_120) ⚠️ Not fully supported yet

**Setup Date:** October 25, 2025

---

## Support

- **PerforatedAI Discord:** https://discord.gg/Fgw3FG3Hzt
- **PyTorch Forums:** https://discuss.pytorch.org/
- **Project Issues:** Check the main [README.md](./README.md)

---

## Summary

✅ **All dependencies installed and working**
✅ **CPU mode fully functional**
⚠️ **GPU mode waiting for PyTorch sm_120 support**
✅ **Ready for development and testing on CPU**

For immediate work, use CPU mode. For production GPU training, monitor PyTorch updates or use WSL2.
