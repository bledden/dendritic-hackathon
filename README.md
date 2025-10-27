# Dendritic Neural Network Compression Research

Research project using Perforated AI's dendritic optimization to compress neural networks by 38-60% with minimal accuracy loss. Currently focused on OpenAI Whisper-Small for edge deployment in HIPAA-compliant medical transcription.

## 🎯 Current Status (October 27, 2025)

**ACTIVE:** 35-epoch test running on RTX 5090 (currently at epoch 2)

### Recent Accomplishments
✅ **Full training pipeline implemented** - Complete training loop with validation
✅ **All critical bugs fixed** - Post-compression validation, memory fragmentation, system crashes
✅ **Production-ready optimizations** - 55% memory reduction, 83% faster post-compression
✅ **Hybrid compression mode** - Guaranteed compression triggers at epochs 10, 20, 30
✅ **Interactive controls** - PAUSE and FORCE_COMPRESS file-based triggers
✅ **Comprehensive testing** - Minimal test validated full stack in 5 minutes
✅ **Repository organized** - Clean structure with archived tests and categorized documentation

### Test Results Summary
- **test_safe_optimized (15 epochs):** Best WER 18.40%, memory optimized, clean completion
- **test_minimal (1 epoch):** Full stack validated, all systems operational
- **test_35_hybrid (IN PROGRESS):** Running now, expected completion in ~7.5 hours

## Quick Links

- **[Whisper Dendritic Training](./whisper-edge-optimization/dendritic/)** - Main training implementation
- **[Quick Start Guide](./QUICKSTART.md)** - Get started quickly
- **[Technical Documentation](./docs/technical/)** - Deep dive into implementation
- **[Test Documentation](./docs/testing/)** - Test strategies and results

## Project Overview

**Primary Goal:** Compress Whisper-Small (240M → ~150M params, 38% reduction) for edge deployment

**Target Performance:**
- Word Error Rate (WER): <25% on LibriSpeech test.clean
- Memory footprint: <2GB for inference
- Deployment: Edge devices with limited compute (medical transcription)

**Hardware:**
- Development: RTX 5090 (32GB VRAM)
- Target: Mobile/edge devices with 4-8GB RAM

**Timeline:** Active development (October 2025)

## Quick Start (Windows with NVIDIA GPU)

```powershell
# Clone repo
git clone https://github.com/bledden/dendritic-hackathon.git
cd dendritic-hackathon

# Setup environment (requires Python 3.9+)
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install openai-whisper datasets transformers soundfile librosa tqdm
cd PerforatedAI && pip install -e . && cd ..

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run minimal diagnostic test (5 minutes)
cd whisper-edge-optimization\dendritic
.\RUN_TEST_MINIMAL.bat

# Run full 35-epoch test (~7.5 hours)
.\RUN_TEST_35_HYBRID.bat
```

## Repository Structure

```
dendritic-hackathon/
├── README.md                              # This file
├── QUICKSTART.md                          # Quick start guide
│
├── docs/                                  # Organized documentation
│   ├── setup/                            # Setup guides (Windows, AMD, GPU)
│   ├── testing/                          # Test strategies and results
│   ├── strategy/                         # Project strategy documents
│   └── technical/                        # Technical implementation details
│
├── whisper-edge-optimization/dendritic/   # Main implementation
│   ├── train_dendritic_full.py           # Production training script
│   ├── RUN_TEST_35_HYBRID.bat            # Primary 35-epoch test
│   ├── RUN_TEST_MINIMAL.bat              # Quick diagnostic test
│   ├── README.md                          # Detailed dendritic docs
│   │
│   ├── docs/                             # Feature documentation
│   │   ├── HOW_TO_PAUSE_AND_COMPRESS.md  # Interactive control guide
│   │   ├── TEST_35_HYBRID_README.md      # 35-epoch test guide
│   │   ├── FEATURES_SUMMARY.md           # All features list
│   │   └── [other docs]
│   │
│   ├── diagnostics/                      # Diagnostic tools
│   │   ├── test_diagnostic.py            # Basic system check
│   │   └── test_diagnostic_full.py       # Full stack validation
│   │
│   ├── scripts/                          # Old/experimental scripts
│   ├── archived_tests/                   # Completed test runs
│   ├── test_35_hybrid/                   # Active 35-epoch test
│   └── test_minimal/                     # Minimal test results
│
└── PerforatedAI/                          # PAI library (submodule)
```

## Key Features

### Training Pipeline
- **Full training loop** with validation and checkpointing
- **BF16 mixed precision** for 50% memory reduction
- **Selective MLP compression** - 48 out of 192 layers
- **History-based compression** with plateau detection
- **Hybrid compression mode** with forced triggers

### Optimizations
- **Memory clearing** after restructuring (55% reduction, 83% faster)
- **Safe termination** with proper GPU cleanup
- **Pin memory** for faster CPU→GPU transfer
- **Zero workers** on Windows (multiprocessing safety)

### Interactive Controls
- **PAUSE** - File-based pause after current epoch
- **FORCE_COMPRESS** - Manual compression trigger anytime
- **Auto-resume** - Continue from last checkpoint

### Fixes Implemented
- ✅ Post-compression validation (direct encoder/decoder calls)
- ✅ Memory fragmentation (torch.cuda.empty_cache)
- ✅ System crashes (torch.cuda.synchronize)
- ✅ Batch file blocking (auto-exit with timeout)

## Current Test (test_35_hybrid)

**Status:** RUNNING (Epoch 2/35)

**Configuration:**
- 35 epochs (~7.5 hours total)
- 100 validation samples
- Batch size: 8
- Hybrid compression: Force at epochs 10, 20, 30

**Expected Milestones:**
- Epoch 10: Add dendrites (240M → 467M params)
- Epoch 20: First prune (467M → ~150M params) 🎯
- Epoch 30: Second prune (further refinement)
- Epoch 35: Final compressed model

**Expected Outcome:**
- Final parameters: ~150M (38% reduction from 240M)
- Final WER: <25% on LibriSpeech test.clean
- Memory: <2GB inference

## Test Results Archive

### test_safe_optimized (15 epochs, COMPLETED)
- **Best WER:** 18.40% (epoch 6)
- **Post-compression WER:** 19.22-21.05%
- **Memory:** 20.39GB reserved (vs 45.76GB before optimization)
- **Speed:** 3.61 it/s post-compression (vs 1.97 before)
- **Parameters:** 240M → 467M (dendrites added, no pruning due to continuous improvement)
- **Status:** ✅ Proved optimizations work, clean completion

### test_minimal (1 epoch, COMPLETED)
- **Purpose:** Full stack validation before long runs
- **Duration:** 5 minutes
- **WER:** 406% (expected - only 20 training samples)
- **Parameters:** 240M → 353M (dendrites added as expected)
- **Status:** ✅ All systems validated, no crashes

### Archived Tests (16 historical runs)
See [whisper-edge-optimization/dendritic/archived_tests/](whisper-edge-optimization/dendritic/archived_tests/) for:
- Baseline measurements
- Early compression attempts
- Bug fix validations
- Optimization comparisons

## Technical Details

### Audio Pipeline
1. Load LibriSpeech from HuggingFace datasets (Arrow format)
2. Decode FLAC bytes with soundfile
3. Pad/trim to 30 seconds
4. Convert to 80-channel log-mel spectrogram
5. Feed to Whisper encoder

### Validation Method
**Problem:** `whisper.decode()` breaks after PAI compression (type checking fails)

**Solution:** Direct encoder/decoder calls with greedy decoding:
1. Encode audio with `model.encoder(mel)`
2. Initialize with SOT tokens
3. Autoregressive generation with `model.decoder(tokens, audio_features)`
4. Decode tokens to text with Whisper tokenizer
5. Calculate WER against reference

### PAI Compression Workflow
1. **Phase 1 (Epochs 1-10):** Train baseline model
2. **Add Dendrites:** 240M → 467M parameters (expand)
3. **Phase 2 (Epochs 11-20):** Learn dendrite importance
4. **Prune:** 467M → ~150M parameters (compress) 🎯
5. **Phase 3 (Epochs 21+):** Fine-tune compressed model

### Hybrid Compression Mode
- **Force triggers:** Epochs 10, 20, 30 (guaranteed)
- **Natural triggers:** 3 epochs without WER improvement
- **Benefit:** Guaranteed compression even if model keeps improving

## Requirements

**Python Packages:**
- Python 3.9+
- PyTorch 2.7.0+ (CUDA 12.8)
- openai-whisper
- datasets, transformers
- soundfile, librosa
- tqdm
- Perforated AI 2.0.4

**Hardware:**
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 5090 32GB)
- 32GB+ system RAM
- 500GB+ storage (for datasets and results)

**Operating System:**
- Windows 11 (primary development)
- Linux/WSL (should work, not tested)

## Documentation

### Getting Started
- [QUICKSTART.md](./QUICKSTART.md) - Quick start guide
- [whisper-edge-optimization/dendritic/README.md](./whisper-edge-optimization/dendritic/README.md) - Detailed implementation guide
- [docs/setup/WINDOWS_SETUP.md](./docs/setup/WINDOWS_SETUP.md) - Windows-specific setup

### Features & Usage
- [docs/dendritic/HOW_TO_PAUSE_AND_COMPRESS.md](./whisper-edge-optimization/dendritic/docs/HOW_TO_PAUSE_AND_COMPRESS.md) - Interactive controls
- [docs/dendritic/TEST_35_HYBRID_README.md](./whisper-edge-optimization/dendritic/docs/TEST_35_HYBRID_README.md) - 35-epoch test guide
- [docs/dendritic/FEATURES_SUMMARY.md](./whisper-edge-optimization/dendritic/docs/FEATURES_SUMMARY.md) - Complete feature list

### Technical Deep Dives
- [docs/technical/PAI_CHECKPOINT_BUG_REPORT.md](./docs/technical/PAI_CHECKPOINT_BUG_REPORT.md) - PAI integration issues
- [docs/dendritic/VALIDATION_BUG_FIX.md](./whisper-edge-optimization/dendritic/docs/VALIDATION_BUG_FIX.md) - Post-compression validation fix
- [docs/dendritic/SAFE_TERMINATION_FIX.md](./whisper-edge-optimization/dendritic/docs/SAFE_TERMINATION_FIX.md) - Crash prevention

### Testing
- [docs/testing/TESTING_STRATEGY.md](./docs/testing/TESTING_STRATEGY.md) - Test approach
- [docs/testing/TEST_RESULTS.md](./docs/testing/TEST_RESULTS.md) - Historical results

## Troubleshooting

### Common Issues

**System crashes after training:**
- ✅ **FIXED:** Use `RUN_TEST_35_HYBRID.bat` with proper GPU cleanup

**Slow training after compression:**
- ✅ **FIXED:** Memory clearing after restructuring (torch.cuda.empty_cache)

**Validation fails after compression:**
- ✅ **FIXED:** Direct encoder/decoder calls instead of whisper.decode()

**No compression triggers:**
- ✅ **FIXED:** Hybrid mode forces compression at epochs 10, 20, 30

### Diagnostics

Run the minimal test to verify your setup:
```powershell
cd whisper-edge-optimization\dendritic
.\RUN_TEST_MINIMAL.bat
```

This validates:
- GPU and PyTorch setup
- Dataset loading
- Model initialization
- Training loop execution
- Proper termination

## Future Work

### Immediate
- ⏳ **Complete 35-epoch test** - Validate compression effectiveness
- 📊 **Analyze results** - WER, parameter reduction, memory usage

### Short-term
- 🎯 **Optimize to 60% reduction** - Include attention layers
- 📱 **Edge deployment** - Test on mobile/embedded devices
- 📉 **Benchmark inference** - Speed and memory profiling

### Long-term
- 🔬 **Second architecture** - Validate on BERT or vision model
- 📄 **Research paper** - Document methodology and results
- 🌐 **AMD MI300X port** - Scale to cloud deployment

## Contributing

This is an active research project. Contributions welcome:
- Bug reports and fixes
- Performance optimizations
- Documentation improvements
- Additional model architectures

## License

[To be determined]

## Citation

Research in progress. Citation details will be provided upon publication.

## Contact

[To be determined]

---

**Note:** This is an active research project under rapid development. See the comprehensive documentation in `docs/` and `whisper-edge-optimization/dendritic/` for technical details.

**Last Updated:** October 27, 2025 - Test 35 running (epoch 2/35)
