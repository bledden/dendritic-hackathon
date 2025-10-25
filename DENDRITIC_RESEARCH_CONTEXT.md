# Dendritic Neural Network Compression Research - Project Context

**Created:** October 24, 2025
**Deadline:** January 2026 (2 months)
**Platform:** AMD MI300X on DigitalOcean
**Budget:** ~$400-500 remaining AMD credits after hackathon

---

## Project Overview

**Goal:** Research and validate Perforated AI's dendritic optimization for neural network compression, targeting 60-90% parameter reduction with minimal accuracy loss.

**Target Models:**
1. **Whisper Small** (240M params) - Speech-to-text
2. **Second model TBD** - BERT, ResNet, DistilBERT, or similar

**Deliverable:** Publication-quality research paper demonstrating dendritic compression across multiple architectures.

---

## Background & Technology

### What is Dendritic Optimization?

Perforated AI's approach adds "dendritic neurons" to existing neural networks during training, allowing aggressive pruning while maintaining performance. Unlike traditional compression:

- **Traditional:** Train → Prune → Fine-tune (often loses accuracy)
- **Dendritic:** Train → Add dendrites when plateau → Continue training → Achieve compression

**Key benefits:**
- 60-90% parameter reduction
- Minimal accuracy loss (< 2-3% typical)
- Works across architectures (CNNs, Transformers, RNNs)

### Perforated AI (PAI) Library

**Version:** 2.0.4
**Documentation:** https://www.perforatedai.com/documentation

**Core API:**
```python
import PerforatedAI.UtilitiesPerforatedAI as UPA
import PerforatedAI.GlobalsPerforatedAI as GPA

# Initialize tracking
UPA.initialize_pai(
    model,
    experiment_name="my_experiment",
    max_dendrites=3
)

# Setup optimizer (3-argument form)
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
    model,
    optim_args,
    sched_args
)

# During training loop
for epoch in range(max_epochs):
    train_loss = train_epoch(model, optimizer)
    val_metric = validate(model)

    # Add validation score (triggers dendrite addition)
    GPA.pai_tracker.add_validation_score(val_metric)

    # Save checkpoint for PAI to load
    UPA.save_system(model, experiment_name, "checkpoint")
```

**Dendrite Addition Trigger:**
- PAI monitors validation scores
- When improvement plateaus (after N epochs of history)
- Automatically adds dendritic neurons
- Restructures model for compression
- Training continues with new architecture

---

## Current State

### What's Been Done

1. **Environment Setup** ✅
   - Python 3.9 environment
   - PyTorch 2.x installed
   - Whisper, datasets, transformers installed
   - Perforated AI 2.0.4 installed

2. **Audio Loading Fixed** ✅
   - Solved torchcodec/MacOS issues
   - Implemented byte-based audio loading with soundfile
   - Works reliably across platforms

3. **PAI Integration** ✅
   - Correct 3-argument setup_optimizer() usage
   - Checkpoint saving configured
   - Validation score tracking implemented

4. **Mac Validation Test** ✅
   - Successfully completed 10-epoch validation-only test
   - Baseline WER: 19.16%
   - No errors in audio loading or PAI integration
   - Process: 594c7c (completed successfully)

### What's NOT Done

1. **Actual Training Loop** ❌
   - Current code only does VALIDATION
   - No forward pass through encoder/decoder
   - No loss calculation
   - No backpropagation
   - No optimizer.step()
   - **This is why no dendrites were added**

2. **Training Dataset** ❌
   - Only loading test.clean (validation set)
   - Need train.clean.100 or larger for actual training

3. **Proper Training Configuration** ❌
   - Need training batch dataloader
   - Need loss function (CTC or sequence loss)
   - Need training loop with backprop

---

## Technical Details

### Current Code Structure

**File:** `whisper-edge-optimization/dendritic/train_dendritic_full.py`

```
Line 45-130:   LibriSpeechDataset class (works, uses byte-based audio)
Line 132-220:  validate() function (works, but no training equivalent)
Line 222-450:  main() function (needs training loop added)
```

**What Works:**
- Dataset loading (LibriSpeech with decode=False)
- Audio preprocessing (soundfile → numpy → mel spectrogram)
- PAI initialization and tracking
- Validation loop and WER calculation

**What Needs to Be Added:**

```python
def train_one_epoch(model, train_loader, optimizer, device):
    """Train for one epoch with backpropagation"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass through Whisper
        mel = batch['mel'].to(device)
        text = batch['text']

        # Encode audio
        audio_features = model.encoder(mel)

        # Decode (need to implement properly)
        # Calculate loss (CTC or sequence loss)
        # loss = calculate_loss(...)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### Audio Pipeline (WORKING)

```python
# 1. Load raw FLAC bytes from HuggingFace dataset
audio_bytes = item['audio']['bytes']

# 2. Decode with soundfile
audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

# 3. Convert to float32, ensure mono
audio_array = audio_array.astype(np.float32)
if len(audio_array.shape) > 1:
    audio_array = audio_array.mean(axis=1)

# 4. Resample to 16kHz if needed
if sample_rate != 16000:
    ratio = 16000 / sample_rate
    new_length = int(len(audio_array) * ratio)
    audio_array = np.interp(
        np.linspace(0, len(audio_array), new_length),
        np.arange(len(audio_array)),
        audio_array
    )

# 5. Whisper preprocessing (pad/trim to 30s)
audio = whisper.pad_or_trim(audio_array)

# 6. Convert to log-mel spectrogram
mel = whisper.log_mel_spectrogram(audio)
```

This pipeline is **tested and working** - no changes needed here.

### PAI Configuration (WORKING)

```python
# Initialize PAI
UPA.initialize_pai(
    model,
    experiment_name=args.save_name,
    max_dendrites=args.max_dendrites,
    threshold=args.improvement_threshold
)

# Setup optimizer (CORRECT 3-arg form)
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

optim_args = {"lr": args.learning_rate}
sched_args = {"mode": "min", "factor": 0.5, "patience": 2}

optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
    model, optim_args, sched_args
)
```

This is **correct and working** - matches PAI 2.0.4 API.

---

## AMD MI300X Deployment

### Hardware Specs

**Single MI300X:**
- 1 GPU - 192 GB VRAM
- 20 vCPU - 240 GB RAM
- Boot disk: 720 GB NVMe
- Scratch disk: 5 TB NVMe
- **Cost:** $1.99/hr

**Recommended Image:**
- PyTorch 2.6.0, ROCm 7.0.0 (Quick Start)
- Pre-configured with PyTorch + ROCm
- Includes JupyterLab
- SSH access

### Installation Steps

```bash
# After SSH into MI300X

# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/dendritic-hackathon.git
cd dendritic-hackathon

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for ROCm (if not pre-installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Install dependencies
pip install openai-whisper==20231117
pip install datasets==2.14.6
pip install transformers==4.35.0
pip install soundfile==0.12.1
pip install librosa==0.10.1
pip install numpy==1.24.3
pip install tqdm==4.66.1
pip install jiwer  # For WER calculation

# 5. Install Perforated AI
cd PerforatedAI
pip install -e .
cd ..

# 6. Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Should output:
# CUDA available: True
# Device: AMD Instinct MI300X
```

### Environment Variables

Create `.env` file:
```bash
# ROCm settings
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH

# PyTorch settings
export PYTORCH_ROCM_ARCH="gfx942"  # MI300X architecture

# Memory settings
export HSA_OVERRIDE_GFX_VERSION=9.4.2
```

---

## Research Methodology

### Phase 1: Baseline & Quick Validation (Week 1)

**Goal:** Verify dendritic compression works on Whisper Small

**Steps:**
1. Deploy MI300X with PyTorch 2.6.0 + ROCm 7.0
2. Add training loop to existing code
3. Run quick test (100 training + 50 validation samples)
4. Verify dendrites are added (look for "MODEL RESTRUCTURED!" message)
5. Confirm 20-40% parameter reduction

**Expected Results:**
- Baseline: 240M params, ~19% WER
- After dendrites: 140-190M params (20-40% reduction), ~20-22% WER
- Training time: 1-2 hours on MI300X

**Success Criteria:**
- Dendrites successfully added during training
- Model parameters reduced
- WER increase < 3%

### Phase 2: Optimize Whisper Small (Week 2-3)

**Goal:** Achieve 60-70% compression with minimal WER degradation

**Variables to test:**
- **Dataset size:** 100 → 500 → 1000 → 5000 samples
- **Max dendrites:** 1, 2, 3, 5
- **Improvement threshold:** 0.0001, 0.0005, 0.001
- **Learning rate:** 1e-5, 5e-5, 1e-4
- **Batch size:** 4, 8, 16, 32

**Experiment matrix:**
- ~30-50 training runs
- Systematic grid search
- Track: compression ratio, WER, training time

**Target Results:**
- 60-70% parameter reduction (70-95M params)
- WER < 22% (< 3% degradation from baseline)
- Reproducible configuration

### Phase 3: Second Model (Week 4-5)

**Model options:**
1. **BERT-base** (110M params) - Text classification
2. **ResNet-50** (25M params) - Image classification
3. **DistilBERT** (66M params) - Text classification

**Goal:** Demonstrate dendritic compression generalizes across architectures

**Same methodology:**
- Baseline training
- Add dendrites
- Optimize for 60-70% compression
- Compare to Whisper results

### Phase 4: Analysis & Paper (Week 6-8)

**Analysis:**
- Compression vs accuracy trade-offs
- Dendrite addition patterns
- Training time analysis
- Architecture comparison

**Paper sections:**
1. Abstract
2. Introduction (dendritic compression background)
3. Methodology (PAI, datasets, training procedures)
4. Results (Whisper + second model)
5. Discussion (findings, limitations, future work)
6. Conclusion

---

## Monitoring & Cost Management

### GPU Monitoring

```bash
# Check GPU utilization
rocm-smi

# Watch GPU in real-time
watch -n 1 rocm-smi

# Check PyTorch GPU memory
python3 -c "import torch; print(f'Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB'); print(f'Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')"
```

### Cost Tracking

**Estimates:**
- Phase 1 (validation): ~$2-4 (1-2 hours)
- Phase 2 (optimization): ~$60-100 (30-50 hours)
- Phase 3 (second model): ~$40-60 (20-30 hours)
- Phase 4 (analysis): ~$10-20 (5-10 hours)
- **Total:** ~$112-184

**Budget:** $400-500 available = plenty of headroom

### Stopping Instance

```bash
# From DigitalOcean web interface:
# Droplets → Your droplet → Power → Power Off

# Save work first!
git add -A
git commit -m "Save progress"
git push

# Then power off to save costs when not training
```

---

## Expected Training Results

### Quick Test (Phase 1)

**Configuration:**
- 100 training samples (LibriSpeech train.clean.100)
- 50 validation samples (LibriSpeech test.clean)
- 20 epochs
- Batch size: 8
- Max dendrites: 2

**Expected timeline (MI300X):**
- Dataset download: 10-15 min (one-time)
- Training: 1-2 hours
- Total: ~1.5-2.5 hours = $3-5

**Expected output:**
```
Epoch 1/20
Training loss: 2.34
Validation WER: 19.16%

Epoch 10/20
Training loss: 1.87
Validation WER: 18.54%

Checking PAI switch with mode n, switch mode DOING_HISTORY...
🌳 MODEL RESTRUCTURED! Dendrites added
Parameters before: 240,582,912
Parameters after: 168,408,038 (30% reduction)

Epoch 11/20
Training loss: 1.92
Validation WER: 19.12%

...

Epoch 20/20
Training loss: 1.65
Validation WER: 18.89%

Final Results:
Baseline params: 240,582,912
Final params: 168,408,038
Reduction: 30.0%
Final WER: 18.89%
```

### Optimized Test (Phase 2)

**Configuration:**
- 1000-5000 training samples
- 100-200 validation samples
- 30-50 epochs
- Batch size: 16-32
- Max dendrites: 3-5

**Expected timeline:** 8-12 hours = $16-24

**Target results:**
- 60-70% parameter reduction
- WER < 22%
- Reproducible

---

## Troubleshooting

### Common Issues

**1. GPU Memory Error**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Reduce max sequence length
- Use gradient checkpointing (add to model config)

**2. Slow Dataset Download**
```
Downloading data: X%...
```
**Solution:**
- First download takes 10-15 min (50GB dataset)
- Cached after first run
- Use smaller dataset for testing: `--val-max-samples 10`

**3. No Dendrites Added**
```
Epoch 20/20
Checking PAI switch... Returning False
```
**Causes:**
- Not enough training epochs (need 10+ for history)
- No improvement plateau (check if loss is still decreasing)
- Threshold too strict (try increasing --improvement-threshold)

**Solution:**
- Increase epochs: `--max-epochs 30`
- Increase threshold: `--improvement-threshold 0.001`
- Check training is actually happening (loss should decrease)

**4. PAI Setup Error**
```
TypeError: setup_optimizer() takes from 3 to 4 positional arguments but X were given
```
**Solution:**
- Use 3-step setup (already implemented in current code)
- Don't modify PAI setup code

**5. ROCm/PyTorch Issues**
```
RuntimeError: No HIP GPUs are available
```
**Solution:**
```bash
# Check ROCm installation
rocm-smi

# Verify PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## Next Immediate Steps

### For Claude Code Session on MI300X

1. **Deploy Droplet**
   - DigitalOcean AMD GPU
   - Single MI300X
   - PyTorch 2.6.0 + ROCm 7.0 image
   - Add SSH key

2. **Setup Environment**
   - SSH into instance
   - Clone GitHub repo
   - Install dependencies
   - Verify GPU works

3. **Add Training Loop**
   - Modify train_dendritic_full.py
   - Add train_one_epoch() function
   - Add training dataloader
   - Implement loss calculation
   - Add backpropagation

4. **Run Quick Test**
   - 100 training + 50 validation samples
   - 20 epochs
   - Batch size 8
   - Max dendrites 2
   - Verify dendrites are added
   - Confirm parameter reduction

5. **Document Results**
   - Save training logs
   - Record parameter reduction %
   - Record WER before/after
   - Git commit results

---

## Code Modifications Needed

### Priority 1: Add Training Loop

**Location:** `train_dendritic_full.py`, around line 220

**Add this function:**

```python
def train_one_epoch(model, train_loader, optimizer, device):
    """
    Train model for one epoch with backpropagation.

    Args:
        model: Whisper model
        train_loader: DataLoader for training data
        optimizer: Optimizer from PAI
        device: torch.device

    Returns:
        average_loss: Average training loss for epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        try:
            # Zero gradients
            optimizer.zero_grad()

            # Get batch data
            mel = batch['mel'].to(device)  # [batch, 80, 3000]
            texts = batch['text']  # List of strings

            # Forward pass through encoder
            audio_features = model.encoder(mel)

            # For simplicity, use Whisper's built-in forward
            # In production, you'd implement proper seq2seq loss
            # This is a simplified training loop

            # Tokenize text targets
            tokens = [model.tokenizer.encode(text) for text in texts]
            max_len = max(len(t) for t in tokens)

            # Pad tokens
            padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
            for i, t in enumerate(tokens):
                padded_tokens[i, :len(t)] = torch.tensor(t)

            padded_tokens = padded_tokens.to(device)

            # Calculate loss (simplified)
            # In production, use proper CTC or sequence-to-sequence loss
            logits = model.decoder(padded_tokens[:, :-1], audio_features)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                padded_tokens[:, 1:].reshape(-1),
                ignore_index=0
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"Warning: Training batch {batch_idx} failed: {e}")
            continue

    average_loss = total_loss / num_batches if num_batches > 0 else 0
    return average_loss
```

**Note:** The above is a SIMPLIFIED training loop. Whisper training is complex. You may want to use a library like `transformers` Trainer or implement proper Whisper training. For quick validation, this simplified version should trigger dendrite addition.

### Priority 2: Add Training Dataset

**Location:** `train_dendritic_full.py`, main() function

**Add around line 300:**

```python
# Load training dataset
print("[4/6] Loading datasets...")
print(f"Loading LibriSpeech train.clean.100...")

train_dataset = LibriSpeechDataset(
    split='train.clean.100',
    max_samples=args.train_max_samples,  # Add this arg
    cache_dir=args.data_dir
)

print(f"      Training samples: {len(train_dataset)}")

# Create training dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn  # Same as validation
)
```

### Priority 3: Update Main Loop

**Location:** `train_dendritic_full.py`, main training loop

**Replace validation-only loop with:**

```python
print("[5/6] Starting training loop...")
print(f"      Max epochs: {args.max_epochs}")

best_wer = float('inf')

for epoch in range(args.max_epochs):
    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{args.max_epochs}")
    print("=" * 70)

    # Training phase
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"\nTraining loss: {train_loss:.4f}")

    # Validation phase
    print("\nRunning validation...")
    val_wer, val_accuracy = validate(
        model, val_dataloader, device,
        max_samples=args.val_max_samples_per_epoch
    )

    print(f"Validation WER: {val_wer*100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

    # Update PAI tracker
    print("\nUpdating PAI tracker...")
    GPA.pai_tracker.add_validation_score(val_accuracy)

    # Learning rate scheduling
    if scheduler is not None:
        scheduler.step(val_wer)

    # Save checkpoint if best
    if val_wer < best_wer:
        best_wer = val_wer
        print(f"\n⭐ New best WER: {best_wer*100:.2f}%")

        try:
            UPA.save_system(model, args.save_name, "best_model")
            print("   💾 Checkpoint saved")
        except Exception as e:
            print(f"   ⚠️  Checkpoint save warning: {e}")
```

### Priority 4: Add Command-Line Arguments

**Add to argparse section:**

```python
parser.add_argument('--train-max-samples', type=int, default=None,
                    help='Maximum training samples (None = all)')
```

---

## File Structure

```
dendritic-hackathon/
├── README.md                           # Project overview
├── DENDRITIC_RESEARCH_CONTEXT.md      # This file
├── AMD_HACKATHON_CONTEXT.md           # Separate hackathon project
├── .gitignore
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (not committed)
│
├── PerforatedAI/                       # PAI library (submodule or local)
│   ├── UtilitiesPerforatedAI.py
│   ├── GlobalsPerforatedAI.py
│   └── ...
│
├── whisper-edge-optimization/
│   └── dendritic/
│       ├── train_dendritic_full.py    # Main training script
│       └── results/                    # Training results (not committed)
│
├── scripts/
│   ├── setup_amd.sh                   # AMD setup automation
│   └── test_gpu_mi300x.py             # GPU verification
│
└── docs/
    └── AMD_SETUP.md                   # Detailed AMD guide
```

---

## Success Metrics

### Technical Metrics

**Phase 1 (Validation):**
- ✅ Dendrites successfully added during training
- ✅ 20-40% parameter reduction achieved
- ✅ WER degradation < 3%

**Phase 2 (Optimization):**
- ✅ 60-70% parameter reduction
- ✅ WER < 22% (< 3% degradation)
- ✅ Reproducible configuration documented

**Phase 3 (Generalization):**
- ✅ Second model achieves 60-70% compression
- ✅ Similar accuracy preservation
- ✅ Comparison analysis complete

### Research Metrics

**Paper Quality:**
- Clear methodology
- Reproducible experiments
- Thorough analysis
- Publication-ready figures
- Conference submission ready (ICML, NeurIPS, ICLR)

**Timeline:**
- Week 1-2: Whisper validation ✅
- Week 3-4: Whisper optimization ✅
- Week 5-6: Second model ✅
- Week 7-8: Paper writing ✅

---

## Resources & References

### Official Documentation

- **Perforated AI:** https://www.perforatedai.com/documentation
- **Whisper:** https://github.com/openai/whisper
- **ROCm:** https://rocm.docs.amd.com/
- **PyTorch ROCm:** https://pytorch.org/get-started/locally/

### Datasets

- **LibriSpeech:** https://www.openslr.org/12
  - train.clean.100: 100 hours clean speech (28,539 samples)
  - train.clean.360: 360 hours clean speech (104,014 samples)
  - test.clean: Test set (2,620 samples)

### Papers

- Whisper: "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI, 2022)
- Perforated AI: Check their website for latest publications

### Community

- **Perforated AI Discord:** (if available)
- **ROCm GitHub:** https://github.com/ROCm/ROCm
- **PyTorch Forums:** https://discuss.pytorch.org/

---

## Parallel Work Context

This dendritic research project is running in parallel with:

**AMD Hackathon (separate project):**
- LLM fine-tuning with Unsloth
- Q&A agent competition
- Wednesday deadline
- $50-100 budget from AMD credits
- Different tech stack (Unsloth, not PAI)
- Different Claude Code session

**Resource Allocation:**
- AMD Hackathon: $50-100 (until Wednesday)
- Dendritic Research: $100-150 (until Wednesday)
- **Remaining after Wed:** $200-250 + $300 new = $500-550 for January work

Both projects use AMD MI300X but are completely independent.

---

## Quick Start Commands

### Deploy MI300X

```bash
# On DigitalOcean:
# 1. Select MI300X (single, $1.99/hr)
# 2. Choose PyTorch 2.6.0 + ROCm 7.0 image
# 3. Add SSH key
# 4. Launch
```

### Setup Environment

```bash
# SSH into instance
ssh root@YOUR_DROPLET_IP

# Clone repo
git clone https://github.com/YOUR_USERNAME/dendritic-hackathon.git
cd dendritic-hackathon

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd PerforatedAI && pip install -e . && cd ..

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Run Quick Test

```bash
cd whisper-edge-optimization/dendritic

# Quick test (1-2 hours)
python train_dendritic_full.py \
  --save-name quick_test_mi300x \
  --train-max-samples 100 \
  --val-max-samples 50 \
  --val-max-samples-per-epoch 50 \
  --max-epochs 20 \
  --max-dendrites 2 \
  --batch-size 8 \
  --num-workers 4

# Monitor in another terminal
watch -n 5 tail -50 results/quick_test_mi300x/training.log
```

### Monitor Costs

```bash
# Check running time
uptime

# Calculate cost
# Hours × $1.99/hr

# Stop when done (from DigitalOcean dashboard)
```

---

## Current Session Handoff

### What This Claude Session Has Done

1. ✅ Fixed audio loading (byte-based with soundfile)
2. ✅ Corrected PAI API usage (3-argument setup)
3. ✅ Ran successful Mac validation test (no dendrites, validation-only)
4. ✅ Created AMD Hackathon context document
5. ✅ Initialized git repository
6. ✅ Created this comprehensive research context

### What Next Claude Session Should Do

1. **Add training loop** to train_dendritic_full.py
2. **Deploy MI300X** on DigitalOcean
3. **Run quick test** (100 samples, 20 epochs)
4. **Verify dendrites work** (check for "MODEL RESTRUCTURED!")
5. **Document results** and push to GitHub

### Known Issues

- ❌ Training loop not implemented (only validation)
- ⚠️ Loss function may need refinement for Whisper
- ⚠️ Whisper training is complex, may need `transformers` library help

### Files Modified

- `whisper-edge-optimization/dendritic/train_dendritic_full.py` (audio loading fixed, PAI setup correct)
- Lines 45-130: LibriSpeechDataset (WORKING)
- Lines 287-290: PAI optimizer setup (WORKING)
- Lines 360-443: Main loop (VALIDATION-ONLY, needs training)

---

## Final Notes

**Key Success Factors:**
1. Add proper training loop (Priority 1)
2. Verify dendrites are actually added
3. Systematic experimentation (don't rush)
4. Document everything (for paper)
5. Monitor costs (stop instance when not using)

**Remember:**
- This is a 2-month research project
- Quality > speed
- Reproducibility is critical
- Publication is the goal

**Budget is generous:**
- $400-500 remaining credits
- Plenty for comprehensive research
- Don't need to rush

**Good luck! 🚀**

---

## Contact & Questions

If you encounter issues:

1. **Check this document first** (Troubleshooting section)
2. **Check Perforated AI docs:** https://www.perforatedai.com/documentation
3. **Check ROCm docs:** https://rocm.docs.amd.com/
4. **Search GitHub issues:**
   - https://github.com/ROCm/ROCm/issues
   - https://github.com/openai/whisper/issues

**For this specific project:**
- GitHub repo: [YOUR REPO URL - will be added after push]
- Original Claude session context preserved in this file

