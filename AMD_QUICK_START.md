# AMD MI300x Quick Reference
## Copy-Paste Commands for Immediate Use

---

## 📋 Prerequisites Checklist

- [ ] AMD MI300x GPU access
- [ ] Linux machine (Ubuntu 20.04/22.04 recommended)
- [ ] SSH access to MI300x machine
- [ ] Git installed
- [ ] Python 3.9+ installed

---

## 🚀 Ultra-Quick Start (Copy-Paste This)

```bash
# Step 1: Upload project to MI300x
# On your local Mac:
cd /Users/bledden/Documents/dendritic-hackathon
tar -czf dendritic-hackathon.tar.gz .
scp dendritic-hackathon.tar.gz user@mi300x-host:~

# Step 2: SSH into MI300x
ssh user@mi300x-host

# Step 3: Extract and setup
cd ~
tar -xzf dendritic-hackathon.tar.gz
cd dendritic-hackathon
chmod +x setup_amd.sh
./setup_amd.sh

# Step 4: Run training (Quick Test - 5 minutes)
source venv/bin/activate
cd whisper-edge-optimization/dendritic
python train_dendritic_full.py \
  --save-name quick_test \
  --val-max-samples 50 \
  --val-max-samples-per-epoch 50 \
  --max-epochs 15 \
  --max-dendrites 3 \
  --batch-size 64 \
  --device cuda
```

**Done!** Training should start in ~2 minutes and complete in ~5 minutes.

---

## 🎯 Recommended Configurations

### Configuration 1: Quick Validation (5 min)
**Use this to verify everything works**

```bash
python train_dendritic_full.py \
  --save-name quick_test \
  --val-max-samples 50 \
  --val-max-samples-per-epoch 50 \
  --max-epochs 15 \
  --max-dendrites 3 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

**Expected:** 40-50% parameter reduction, ~12-15% WER, 5 minutes

### Configuration 2: Production Run (45-60 min)
**Use this for hackathon submission**

```bash
python train_dendritic_full.py \
  --save-name hackathon_submission \
  --val-max-samples 500 \
  --val-max-samples-per-epoch 200 \
  --max-epochs 50 \
  --max-dendrites 10 \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda \
  --use-wandb
```

**Expected:** 60-70% parameter reduction, <13% WER, 60 minutes

### Configuration 3: Maximum Compression (2-3 hours)
**Use this for best results**

```bash
python train_dendritic_full.py \
  --save-name max_compression \
  --train-max-samples 2000 \
  --val-max-samples 1000 \
  --val-max-samples-per-epoch 500 \
  --max-epochs 100 \
  --max-dendrites 20 \
  --batch-size 128 \
  --num-workers 16 \
  --device cuda \
  --use-wandb
```

**Expected:** 80-90% parameter reduction, <12% WER, 2-3 hours

---

## 📊 Monitoring Commands

```bash
# Terminal 1: Run training
python train_dendritic_full.py [args]

# Terminal 2: Watch GPU usage
watch -n 1 'rocm-smi | head -20'

# Terminal 3: Watch training log
tail -f ../results/<save-name>/training.log

# Check results
cat ../results/<save-name>/final_results.json
```

---

## 🔧 Troubleshooting One-Liners

```bash
# GPU not detected?
python test_gpu_mi300x.py

# Out of memory?
# Reduce batch size: --batch-size 32

# Slow performance?
# Increase batch size: --batch-size 256

# Re-run setup
./setup_amd.sh

# Check ROCm
rocm-smi

# Check PyTorch + ROCm
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Clean cache
rm -rf ~/.cache/huggingface/datasets
```

---

## 📦 File Transfer Commands

### Upload from Mac to MI300x

```bash
# Sync entire project
cd /Users/bledden/Documents
rsync -avz --exclude 'venv' --exclude '__pycache__' \
  dendritic-hackathon/ \
  user@mi300x-host:~/dendritic-hackathon/

# Upload just the training code
cd /Users/bledden/Documents/dendritic-hackathon
scp whisper-edge-optimization/dendritic/train_dendritic_full.py \
  user@mi300x-host:~/dendritic-hackathon/whisper-edge-optimization/dendritic/
```

### Download results from MI300x to Mac

```bash
# Download results folder
scp -r user@mi300x-host:~/dendritic-hackathon/whisper-edge-optimization/results/ \
  /Users/bledden/Documents/dendritic-hackathon/whisper-edge-optimization/

# Download specific run
scp -r user@mi300x-host:~/dendritic-hackathon/whisper-edge-optimization/results/hackathon_submission/ \
  /Users/bledden/Documents/dendritic-hackathon/results/
```

---

## 🎨 Performance Tuning Quick Guide

### Find Optimal Batch Size

```bash
# Start with 64
python train_dendritic_full.py --batch-size 64 [other args]

# If works, try 128
python train_dendritic_full.py --batch-size 128 [other args]

# If works, try 256
python train_dendritic_full.py --batch-size 256 [other args]

# If OOM, go back to last working size
```

MI300x has 192GB RAM, so **batch size 128-256** should work fine.

### Maximize Speed

```python
# Use these flags for maximum speed:
--batch-size 256       # Largest batch that fits
--num-workers 16       # Match CPU cores
--device cuda          # Use GPU (duh)
```

---

## 💾 Checkpoint Management

```bash
# Save checkpoint manually (already automatic in code)
# Results are in: ../results/<save-name>/

# Load checkpoint and resume (if interrupted)
# Coming soon in next version...

# Export best model
python export_model.py \
  --checkpoint ../results/hackathon_submission/best_model.pt \
  --format onnx
```

---

## 📈 W&B Integration

```bash
# First time setup
wandb login

# Run with W&B
python train_dendritic_full.py --use-wandb [other args]

# View results
# Visit: https://wandb.ai/your-username/dendritic-whisper
```

---

## 🐳 Docker Alternative

If you prefer Docker:

```bash
# Pull ROCm PyTorch image
docker pull rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# Run container
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/dendritic-hackathon:/workspace \
  rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# Inside container
cd /workspace
./setup_amd.sh
cd whisper-edge-optimization/dendritic
python train_dendritic_full.py [args]
```

---

## 🎯 Expected Performance

| Config | Time | GPU Usage | Params Reduced | Final WER |
|--------|------|-----------|----------------|-----------|
| Quick Test | 5 min | 40-50% | 40-50% | 12-15% |
| Production | 60 min | 70-80% | 60-70% | <13% |
| Max Compression | 3 hrs | 80-90% | 80-90% | <12% |

---

## 🆘 Emergency Help

```bash
# Kill training
pkill -f train_dendritic_full.py

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check disk space
df -h

# Check memory
free -h

# Full system status
rocm-smi
nvidia-smi  # Also works with ROCm compatibility
```

---

## 📚 Additional Resources

- Full setup: `AMD_SETUP.md`
- GPU test: `python test_gpu_mi300x.py`
- Dependencies: `requirements_amd.txt`
- Auto-setup: `./setup_amd.sh`

---

## ✅ Success Criteria

You'll know it's working when you see:

```
======================================================================
🧠 DENDRITIC WHISPER FULL TRAINING
======================================================================

Device: cuda

[1/6] Loading Whisper Small...
      Parameters: 240,582,912

[2/6] Configuring Perforated AI...
      ✅ PAI initialized

[3/6] Setting up optimizer...
      ✅ Optimizer configured

[4/6] Loading datasets...
      ✅ Datasets loaded

[5/6] Starting training loop...

======================================================================
Epoch 1/50
======================================================================

Running validation...
Validating: 100%|██████████| 25/25 [00:30<00:00,  1.21s/it]

Validation WER: 13.45%
Validation Accuracy: 86.55%

🌳 MODEL RESTRUCTURED! Dendrites added/incorporated.
   New parameters: 144,349,747
   Reduction: 40.0%

⭐ New best WER: 13.45%
   💾 Checkpoint saved
```

---

That's it! Copy-paste and go! 🚀
