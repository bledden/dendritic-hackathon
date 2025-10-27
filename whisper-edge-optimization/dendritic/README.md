# Dendritic Compression for Whisper-Small

This project implements dendritic neural network compression for OpenAI's Whisper-Small model using PerforatedAI 2.0.4, targeting edge deployment for HIPAA-compliant medical transcription.

## Project Structure

```
dendritic/
├── train_dendritic_full.py          # Main training script (current, production-ready)
├── RUN_TEST_35_HYBRID.bat           # Primary test: 35 epochs with hybrid compression
├── RUN_TEST_MINIMAL.bat             # Quick diagnostic test (1 epoch, minimal data)
│
├── docs/                             # Documentation
│   ├── HOW_TO_PAUSE_AND_COMPRESS.md # User guide for interactive controls
│   ├── TEST_35_HYBRID_README.md     # Guide for the 35-epoch test
│   ├── FEATURES_SUMMARY.md          # Summary of all implemented features
│   ├── SAFE_TERMINATION_FIX.md      # Documentation of crash fix
│   ├── VALIDATION_BUG_FIX.md        # Fix for post-compression validation
│   └── OPTIMIZATION_TODO.md         # Future optimization ideas
│
├── diagnostics/                      # Diagnostic and debugging tools
│   ├── test_diagnostic.py           # Basic GPU/PyTorch/Whisper diagnostic
│   ├── test_diagnostic_full.py      # Full diagnostic including dataset loading
│   ├── RUN_DIAGNOSTIC.bat           # Run basic diagnostic
│   ├── RUN_DIAGNOSTIC_FULL.bat      # Run full diagnostic
│   ├── diagnose_pai_checkpoint.py   # Debug PAI checkpoint loading
│   └── diagnose_pai_modules.py      # Debug PAI module conversion
│
├── scripts/                          # Old/experimental batch files and scripts
│   ├── train_dendritic.py           # Original training script
│   ├── train_dendritic_fixed_trigger.py  # Intermediate version
│   └── *.bat                        # Old test batch files (archived)
│
├── archived_tests/                   # Completed test runs (historical data)
│   ├── test_safe_optimized/         # 15-epoch test that proved optimizations work
│   ├── test_1b_gpu_baseline_10/     # Baseline WER measurement
│   └── [other test directories]
│
├── test_35_hybrid/                   # Active: 35-epoch hybrid test results
├── test_minimal/                     # Active: Minimal diagnostic test results
└── data/                             # Cached dataset files
```

## Quick Start

### Running the Main Test

For the full 35-epoch test with hybrid compression:

```powershell
cd whisper-edge-optimization\dendritic
.\RUN_TEST_35_HYBRID.bat
```

**Expected runtime:** ~7.5 hours
- Epochs 1-10: Pre-compression training
- Epoch 10: Add dendrites (240M → 467M params)
- Epochs 11-20: Learn dendrite importance
- Epoch 20: Prune (467M → ~150M params)
- Epochs 21-35: Fine-tune compressed model

### Running Diagnostics

To verify your system is ready before starting a long test:

```powershell
.\RUN_TEST_MINIMAL.bat
```

This runs a quick 1-epoch test with minimal data (~5 minutes).

## Interactive Controls

### Pause Training

Create a `PAUSE` file in the results directory to pause after the current epoch:

```powershell
echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"
```

### Force Compression

Create a `FORCE_COMPRESS` file to manually trigger compression at the next epoch:

```powershell
echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"
```

See [docs/HOW_TO_PAUSE_AND_COMPRESS.md](docs/HOW_TO_PAUSE_AND_COMPRESS.md) for detailed instructions.

## Key Features

1. **Hybrid Compression Mode**: Guarantees compression at epochs 10, 20, 30 OR when natural plateau detected
2. **Memory Optimization**: GPU cache clearing after restructuring (55% memory reduction, 83% faster)
3. **Safe Termination**: Proper GPU cleanup prevents system crashes
4. **Interactive Controls**: PAUSE and FORCE_COMPRESS support
5. **BF16 Mixed Precision**: ~50% memory reduction with Tensor Cores
6. **Selective MLP Compression**: Only 48 out of 192 Linear layers (encoder/decoder MLP layers)

## Compression Goals

- **Starting Model**: 240M parameters (Whisper-Small)
- **After Dendrite Addition**: 467M parameters (temporary expansion)
- **After Pruning**: ~150M parameters (38% reduction)
- **Target WER**: <25% on LibriSpeech test.clean
- **Edge Deployment**: Optimized for 5090 GPU, compatible with mobile/edge devices

## Requirements

- Python 3.8+
- PyTorch 2.7.0+ with CUDA 12.8
- PerforatedAI 2.0.4
- OpenAI Whisper
- HuggingFace Datasets (LibriSpeech)
- GPU: NVIDIA RTX 5090 (32GB) or similar

## Results Location

All test results are saved to: `D:\ML_Results\dendritic_whisper\`

Each test creates a subdirectory with:
- Checkpoints (`.pt` files)
- Metrics CSVs (WER, loss, parameters, etc.)
- Training plots (`.png`)
- Final results JSON

## Troubleshooting

### System Crashes After Training

- **Fixed**: Use `RUN_TEST_35_HYBRID.bat` which has proper GPU cleanup and auto-exit
- Old batch files with blocking `pause` command can cause issues

### Memory Issues

- Reduce `--batch-size` from 8 to 4 or 2
- Reduce `--val-max-samples` from 100 to 50
- The script includes automatic memory clearing after restructuring

### Slow Training

- Ensure `--use-amp` and `--amp-dtype bfloat16` are enabled
- Check GPU utilization with `nvidia-smi`
- Verify `--num-workers 0` on Windows (multiprocessing issues)

### Diagnostics

Run `.\RUN_TEST_MINIMAL.bat` to verify:
- GPU and PyTorch setup
- Dataset loading
- Model initialization
- Training loop execution

For detailed diagnostics, see the [diagnostics/](diagnostics/) directory.

## Documentation

- [HOW_TO_PAUSE_AND_COMPRESS.md](docs/HOW_TO_PAUSE_AND_COMPRESS.md) - Interactive control guide
- [TEST_35_HYBRID_README.md](docs/TEST_35_HYBRID_README.md) - Full guide for the 35-epoch test
- [FEATURES_SUMMARY.md](docs/FEATURES_SUMMARY.md) - Complete feature list
- [SAFE_TERMINATION_FIX.md](docs/SAFE_TERMINATION_FIX.md) - Crash fix documentation
- [VALIDATION_BUG_FIX.md](docs/VALIDATION_BUG_FIX.md) - Post-compression validation fix

## Contributing

This is a research project. For questions or issues, please refer to the documentation in the [docs/](docs/) directory.

## License

See root repository for license information.
