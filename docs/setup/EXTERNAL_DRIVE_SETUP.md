# External Drive Setup for Dendritic Whisper

## Quick Setup Guide

### 1. Connect Your External SSD
- Plug in your 2TB external SSD
- Note which drive letter it gets (usually D:, E:, F:, etc.)
- Check with: `Get-PSDrive -PSProvider FileSystem`

### 2. Clean Up Partial Downloads on C:
```powershell
# Remove incomplete LibriSpeech download from C:
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\datasets\*librispeech*" -ErrorAction SilentlyContinue

# Optional: Clear entire HuggingFace cache to free more space
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\datasets\*" -ErrorAction SilentlyContinue
```

### 3. Create Data Directory on External Drive
```powershell
# Assuming your external drive is E: (adjust as needed)
New-Item -ItemType Directory -Path "E:\ML_Datasets\LibriSpeech" -Force
New-Item -ItemType Directory -Path "E:\ML_Results\dendritic_whisper" -Force
```

### 4. Run Training with External Drive
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Navigate to training directory
cd whisper-edge-optimization\dendritic

# Run training with external drive paths
python train_dendritic_full.py `
  --device cpu `
  --save-name windows_test `
  --val-max-samples 10 `
  --max-epochs 3 `
  --batch-size 2 `
  --data-dir "E:\ML_Datasets\LibriSpeech" `
  --results-dir "E:\ML_Results\dendritic_whisper"
```

**Note:** Replace `E:` with your actual external drive letter!

---

## Performance Notes

### SSD Speed Considerations:
- **USB 3.0/3.1**: ~200-500 MB/s (good for this project)
- **USB 3.2/Thunderbolt**: ~1000+ MB/s (excellent)
- **USB 2.0**: ~30 MB/s (too slow, avoid if possible)

**Check your connection speed:**
```powershell
# Windows will show USB version in Device Manager
# Or test with a large file copy to check speed
```

### Expected Dataset Download Times:
- **USB 3.0+**: ~15-30 minutes for full LibriSpeech (60GB)
- **USB 2.0**: ~30-60 minutes

---

## Recommended Directory Structure on External Drive

```
E:\  (or your drive letter)
├── ML_Datasets\
│   └── LibriSpeech\           # LibriSpeech dataset cache (~60GB)
├── ML_Results\
│   └── dendritic_whisper\     # Training results, checkpoints (~10-50GB)
└── ML_Models\                 # Pre-trained models cache (optional)
```

---

## Training Commands with External Drive

### Small CPU Test (10 samples)
```powershell
python train_dendritic_full.py `
  --device cpu `
  --save-name test_10samples `
  --val-max-samples 10 `
  --max-epochs 3 `
  --batch-size 2 `
  --data-dir "E:\ML_Datasets\LibriSpeech"
```

### Medium CPU Test (100 samples)
```powershell
python train_dendritic_full.py `
  --device cpu `
  --save-name test_100samples `
  --val-max-samples 100 `
  --max-epochs 10 `
  --batch-size 4 `
  --data-dir "E:\ML_Datasets\LibriSpeech"
```

### Full Validation (when GPU is available)
```powershell
python train_dendritic_full.py `
  --device cuda `
  --save-name full_validation `
  --val-max-samples 2620 `
  --max-epochs 30 `
  --batch-size 16 `
  --data-dir "E:\ML_Datasets\LibriSpeech" `
  --results-dir "E:\ML_Results\dendritic_whisper"
```

---

## Troubleshooting

### "Drive not found" error
```powershell
# Check available drives
Get-PSDrive -PSProvider FileSystem

# Verify path exists
Test-Path "E:\ML_Datasets"
```

### Slow downloads
- Check if using USB 3.0+ port (usually blue or marked with "SS")
- Avoid USB hubs if possible
- Close bandwidth-heavy applications

### Permission issues
```powershell
# Run PowerShell as Administrator if needed
# Right-click PowerShell -> "Run as Administrator"
```

---

## Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| LibriSpeech test.clean | ~2.5GB | Validation only |
| LibriSpeech train.clean.100 | ~6GB | Small training set |
| LibriSpeech train.clean.360 | ~23GB | Medium training set |
| LibriSpeech train.other.500 | ~30GB | Large training set |
| **Total LibriSpeech** | **~60GB** | All splits combined |
| Model checkpoints | ~5-20GB | Per training run |
| Results & logs | ~1-5GB | Per training run |
| **Recommended Free Space** | **100GB+** | For comfort |

Your 2TB drive has plenty of room! 🎉

---

## Benefits of This Setup

✅ **Keeps C: drive clean** (system stays fast)
✅ **Plenty of space** for multiple experiments
✅ **Portable** - can move between machines
✅ **Organized** - all ML data in one place
✅ **Future-proof** - room for other projects

---

## After Training Completes

Your results will be in:
```
E:\ML_Results\dendritic_whisper\windows_test\
├── final_results.json        # WER scores, parameters
├── model_final.pth           # Final model weights
├── training_log.txt          # Training logs
└── pai_graphs\               # PAI compression visualizations
```

---

## Next Steps

1. **Connect external drive**
2. **Note the drive letter** (D:, E:, F:, etc.)
3. **Clean up C: drive** (run cleanup commands above)
4. **Create directories** on external drive
5. **Run training** with `--data-dir` pointing to external drive

**You're all set!** 🚀
