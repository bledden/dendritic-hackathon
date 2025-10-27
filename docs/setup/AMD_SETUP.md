# AMD MI300x Deployment Guide
## Dendritic Whisper Optimization

This guide walks you through deploying the dendritic Whisper training on AMD MI300x GPUs using ROCm.

---

## Prerequisites

- AMD MI300x GPU access
- Ubuntu 20.04/22.04 (recommended) or compatible Linux
- ROCm 5.7 or 6.0
- Python 3.9+
- Git

---

## Quick Start (5 minutes)

If you already have ROCm installed:

```bash
# 1. Clone the repository
cd ~
git clone <your-repo-url> dendritic-hackathon
cd dendritic-hackathon

# 2. Create Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for ROCm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Install dependencies
pip install -r requirements_amd.txt
pip install -e ./PerforatedAI

# 5. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 6. Verify GPU
python test_gpu_mi300x.py

# 7. Run training!
cd whisper-edge-optimization/dendritic
python train_dendritic_full.py \
  --save-name mi300x_run \
  --val-max-samples 500 \
  --val-max-samples-per-epoch 200 \
  --max-epochs 50 \
  --max-dendrites 10 \
  --batch-size 64 \
  --device cuda
```

---

## Detailed Setup

### Step 1: ROCm Installation (if not already installed)

#### Check if ROCm is already installed:
```bash
rocm-smi
```

If this works, skip to Step 2.

#### Fresh ROCm Installation:

**Ubuntu 20.04/22.04:**
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 focal main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-hip-sdk rocm-dev

# Add user to render group
sudo usermod -a -G render,video $LOGNAME

# Reboot or re-login
sudo reboot
```

**Verify ROCm:**
```bash
rocm-smi
rocminfo | grep "Name:"
```

You should see your MI300x GPU listed.

### Step 2: PyTorch Installation

**CRITICAL:** Use the ROCm-specific PyTorch build:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install PyTorch for ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# OR for ROCm 6.0:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Verify PyTorch sees GPU:**
```python
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:
```
True
AMD Instinct MI300X
```

### Step 3: Project Setup

```bash
# Clone repository
git clone <your-repo-url> dendritic-hackathon
cd dendritic-hackathon

# Install dependencies
pip install -r requirements_amd.txt

# Install Perforated AI
pip install -e ./PerforatedAI

# Set up environment variables
cp .env.example .env
nano .env  # Add your API keys
```

### Step 4: GPU Verification

Run the comprehensive GPU test:

```bash
python test_gpu_mi300x.py
```

Expected output:
```
======================================================================
🚀 AMD MI300x GPU Verification Script
======================================================================
✅ PyTorch imported successfully
   Version: 2.2.0+rocm5.7
✅ CUDA/ROCm is available
   CUDA version: 11.8
✅ Detected 1 GPU(s)

📊 GPU 0 Properties:
   Name: AMD Instinct MI300X
   Total memory: 192.00 GB
   Compute capability: 9.0
   Multi-processor count: 304

🔬 Testing GPU computation...
✅ GPU computation successful
   Result shape: torch.Size([5000, 5000])
   Result device: cuda:0

st.d/rocm.list

sudo apt update
sudo apt install rocm-hip-sdk rocm-dev

# Add user to render group
sudo usermod -a -G render,video $LOGNAME

# Reboot or re-login
sudo reboot
```

**Verify ROCm:**
```bash
rocm-smi
rocminfo | grep "Name:"
```

You should see your MI300x GPU listed.

### Step 2: PyTorch Installation

**CRITICAL:** Use the ROCm-specific PyTorch build:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install PyTorch for ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# OR for ROCm 6.0:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Verify PyTorch sees GPU:**
```python
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:
```
True
AMD Instinct MI300X
```

### Step 3: Project Setup

```bash
# Clone repository
git clone <your-repo-url> dendritic-hackathon
cd dendritic-hackathon

# Install dependencies
pip install -r requirements_amd.txt

# Install Perforated AI
pip install -e ./PerforatedAI

# Set up environment variables
cp .env.example .env
nano .env  # Add your API keys
```

### Step 4: GPU Verification

Run the comprehensive GPU test:

```bash
python test_gpu_mi300x.py
```

Expected output:
```
======================================================================
🚀 AMD MI300x GPU Verification Script
======================================================================
✅ PyTorch imported successfully
   Version: 2.2.0+rocm5.7
✅ CUDA/ROCm is available
   CUDA version: 11.8
✅ Detected 1 GPU(s)

📊 GPU 0 Properties:
   Name: AMD Instinct MI300X
   Total memory: 192.00 GB
   Compute capability: 9.0
   Multi-processor count: 304

🔬 Testing GPU computation...
✅ GPU computation successful
   Result shape: torch.Size([5000, 5000])
   Result device: cuda:0

s, try 64
python train_dendritic_full.py --batch-size 64 [other args]

# If it works, try� Testing mixed precision (FP16)...
✅ Mixed precision (FP16) supported

📦 Checking dependencies...
   ✅ whisper
   ✅ datasets
   ✅ soundfile
   ✅ numpy
   ✅ tqdm

======================================================================
📊 SUMMARY
======================================================================
PyTorch Installation              ✅ PASS
CUDA/ROCm Available               ✅ PASS
GPU Detection                     ✅ PASS
GPU Properties                    ✅ PASS
GPU Computation                   ✅ PASS
Mixed Precision                   ✅ PASS
Dependencies                      ✅ PASS

7/7 tests passed

🎉 All tests passed! Ready for training.
```

### Step 5: Training Configuration

The code automatically detects CUDA/ROCm. No code changes needed!

**Recommended training configurations:**

#### Quick Test (5 minutes)
```bash
python train_dendritic_full.py \
  --save-name quick_test \
  --val-max-samples 50 \
  --val-max-samples-per-epoch 50 \
  --max-epochs 15 \
  --max-dendrites 3 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda
```

#### Production Run (45-60 minutes)
```bash
python train_dendritic_full.py \
  --save-name production_run \
  --val-max-samples 500 \
  --val-max-samples-per-epoch 200 \
  --max-epochs 50 \
  --max-dendrites 10 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda \
  --use-wandb
```

#### Maximum Compression (2-3 hours)
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

---

## Performance Optimization

### Batch Size Tuning

Find the optimal batch size for your GPU:

```bash
# Start with 32
python train_dendritic_full.py --batch-size 32 [other args]

# If it works, try 64
python train_dendritic_full.py --batch-size 64 [other args]

# If it works, try 128
python train_dendritic_full.py --batch-size 128 [other args]

# If you get OOM, reduce by half
```

MI300x has 192GB RAM, so you can likely use batch sizes of **128 or even 256**.

### Mixed Precision Training

For 2x speedup, enable mixed precision (already in code):

```python
# This is automatically enabled in the code when GPU is detected
from torch.cuda.amp import autocast, GradScaler
```

### Multi-GPU Setup

If you have multiple MI300x GPUs:

```bash
# Use DataParallel (simple)
python train_dendritic_full.py --device cuda [args]

# Or use DistributedDataParallel (advanced, faster)
# Coming soon...
```

---

## Monitoring

### Real-time GPU Monitoring

```bash
# Terminal 1: Run training
python train_dendritic_full.py [args]

# Terminal 2: Watch GPU usage
watch -n 1 rocm-smi
```

### Weights & Biases Integration

```bash
# Enable W&B logging
python train_dendritic_full.py --use-wandb [other args]

# View results at wandb.ai
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check ROCm installation
rocm-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check environment
echo $ROCM_PATH
echo $LD_LIBRARY_PATH
```

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 32  # or 16

# Reduce number of workers
--num-workers 4  # or 0

# Reduce validation samples
--val-max-samples-per-epoch 100
```

### Slow Performance

```bash
# Increase batch size
--batch-size 128

# Increase workers
--num-workers 16

# Enable mixed precision (already on by default with GPU)
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements_amd.txt

# Verify Perforated AI
pip install -e ./PerforatedAI
python -c "import perforatedai; print('OK')"
```

---

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Pull AMD ROCm PyTorch image
docker pull rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# Run container
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $PWD:/workspace \
  --name dendritic-training \
  rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# Inside container
cd /workspace
pip install -r requirements_amd.txt
pip install -e ./PerforatedAI
python test_gpu_mi300x.py
python whisper-edge-optimization/dendritic/train_dendritic_full.py [args]
```

---

## Expected Results

### Performance Benchmarks

| Configuration | Time | Parameter Reduction | Final WER |
|--------------|------|---------------------|-----------|
| Quick Test | 5 min | 40-50% | <15% |
| Production | 60 min | 60-70% | <13% |
| Max Compression | 3 hrs | 80-90% | <12% |

### Output Files

Results saved to: `whisper-edge-optimization/results/<save-name>/`

- `final_results.json` - Final metrics
- `training_log.txt` - Full training log
- `best_model.pt` - Best model checkpoint
- `pai_graphs/` - PAI dendrite visualizations

---

## Next Steps

After training completes:

1. **Analyze Results:**
   ```bash
   cat ../results/<save-name>/final_results.json
   ```

2. **Evaluate Model:**
   ```bash
   python evaluate_model.py --checkpoint ../results/<save-name>/best_model.pt
   ```

3. **Export for Production:**
   ```bash
   python export_model.py --checkpoint ../results/<save-name>/best_model.pt --format onnx
   ```

---

## Support

- AMD ROCm docs: https://rocm.docs.amd.com/
- PyTorch ROCm support: https://pytorch.org/get-started/locally/
- Perforated AI: https://www.perforatedai.com/

---

## Performance Notes

MI300x advantages for this project:
- **192GB RAM** → Can load entire Whisper + large batches
- **High bandwidth** → Fast audio preprocessing
- **FP16 support** → 2x training speedup
- **Multi-GCD** → Can parallelize validation

Expected speedup vs CPU:
- **10-15x faster** per epoch
- **Can run overnight** for maximum compression
- **Larger batch sizes** → better gradient estimates
