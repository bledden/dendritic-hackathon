# Dendritic Neural Network Compression Research

Research project using Perforated AI's dendritic optimization to compress neural networks by 60-90% with minimal accuracy loss.

## Quick Links

- **[Dendritic Research Context](./DENDRITIC_RESEARCH_CONTEXT.md)** - Complete technical documentation for dendritic compression work
- **[AMD Hackathon Context](./AMD_HACKATHON_CONTEXT.md)** - Separate AMD LLM fine-tuning hackathon project

## Project Overview

**Goal:** Validate dendritic compression on multiple neural network architectures.

**Target Models:**
1. Whisper Small (240M params) - Speech-to-text
2. BERT/ResNet/other (TBD) - Second architecture

**Timeline:** 2 months (October 2025 - January 2026)

**Platform:** AMD MI300X GPUs on DigitalOcean

## Current Status

✅ Environment setup complete
✅ Audio loading pipeline fixed (byte-based)
✅ PAI integration correct
✅ Mac validation test successful (baseline: 19.16% WER)
❌ Training loop needs to be added (currently validation-only)

## Quick Start (AMD MI300X)

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/dendritic-hackathon.git
cd dendritic-hackathon

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd PerforatedAI && pip install -e . && cd ..

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Run training (after adding training loop)
cd whisper-edge-optimization/dendritic
python train_dendritic_full.py \
  --save-name quick_test \
  --train-max-samples 100 \
  --val-max-samples 50 \
  --max-epochs 20 \
  --batch-size 8
```

## Repository Structure

```
dendritic-hackathon/
├── README.md                          # This file
├── DENDRITIC_RESEARCH_CONTEXT.md     # Complete research documentation
├── AMD_HACKATHON_CONTEXT.md          # Separate hackathon project
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── PerforatedAI/                      # PAI library
│
├── whisper-edge-optimization/
│   └── dendritic/
│       └── train_dendritic_full.py   # Main training script
│
└── scripts/
    ├── setup_amd.sh                  # AMD setup automation
    └── test_gpu_mi300x.py            # GPU verification
```

## Key Features

- **Byte-based audio loading** - Works across platforms, no torchcodec issues
- **PAI 2.0.4 integration** - Correct API usage for dendritic optimization
- **AMD ROCm support** - Optimized for MI300X GPUs
- **Comprehensive documentation** - Full research context and troubleshooting

## Technical Details

**Audio Pipeline:**
1. Load FLAC bytes from HuggingFace datasets
2. Decode with soundfile
3. Resample to 16kHz
4. Convert to mel spectrogram
5. Feed to Whisper model

**PAI Integration:**
1. Initialize tracking with UPA.initialize_pai()
2. Setup optimizer with 3-argument form
3. Add validation scores during training
4. PAI automatically adds dendrites when plateau detected
5. Model restructures for compression

## Next Steps

1. **Add training loop** to train_dendritic_full.py
2. **Deploy MI300X** on DigitalOcean
3. **Run quick test** (100 samples, verify dendrites work)
4. **Optimize** for 60-70% compression
5. **Scale to second model**
6. **Write paper**

## Requirements

- Python 3.9+
- PyTorch 2.x (with ROCm for AMD GPUs)
- openai-whisper
- datasets, transformers
- soundfile, librosa
- Perforated AI 2.0.4

See `requirements.txt` for full list.

## Hardware

**Tested on:**
- Mac M4 Max (CPU, validation only)
- AMD MI300X (GPU, recommended)

**Recommended:**
- AMD MI300X: 192GB VRAM, $1.99/hr on DigitalOcean
- Single GPU sufficient for Whisper Small

## Documentation

See **[DENDRITIC_RESEARCH_CONTEXT.md](./DENDRITIC_RESEARCH_CONTEXT.md)** for:
- Complete technical background
- Detailed PAI API usage
- Training methodology
- Troubleshooting guide
- Research timeline
- Expected results

## License

[Add your license here]

## Citation

If you use this work, please cite:

```
[Add citation after publication]
```

## Contact

[Add your contact info]

---

**Note:** This is an active research project. The code is evolving rapidly. See DENDRITIC_RESEARCH_CONTEXT.md for the latest status and technical details.
