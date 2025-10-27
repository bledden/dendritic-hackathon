# Quick Start Guide - Dendritic Hackathon

**Time to hack:** October 23, 2025, 6:30pm
**Location:** `/Users/bledden/Documents/dendritic-hackathon`

---

## 🚨 Before You Leave (RIGHT NOW - 10 minutes)

### Step 1: Run Setup Script
```bash
cd /Users/bledden/Documents/dendritic-hackathon
./SETUP.sh
```

This will:
- Create Python virtual environment
- Install PyTorch, Transformers, W&B
- Clone Perforated AI repository
- Download IMDB dataset (caches locally)
- Create project structure

### Step 2: Login to W&B
```bash
source venv/bin/activate
wandb login
```

Get your API key: https://wandb.ai/authorize

### Step 3: Quick Test
```bash
python -c "import torch; import transformers; print('✅ Ready!')"
```

If this prints "✅ Ready!" → You're good to go!

---

## 📋 What You Have Now

```
dendritic-hackathon/
├── HACKATHON_STRATEGY.md      ← Full strategic plan (read this!)
├── PROJECT_CONTEXT.md          ← Your existing projects analysis
├── SETUP.sh                    ← Setup script (already ran)
├── QUICK_START.md              ← This file
├── PerforatedAI/               ← Perforated AI repo (cloned)
├── bert-sentiment-optimization/
│   ├── README.md               ← Project-specific guide
│   ├── baseline/
│   │   └── train_baseline.py  ← Baseline BERT training script
│   └── sweeps/
│       └── sweep_config.yaml  ← W&B sweep configuration
└── venv/                       ← Python environment
```

---

## 🎯 At the Hackathon (Oct 23, 6:30pm-10:45pm)

### Timeline (CRITICAL)

**6:30-7:00pm: Networking**
- Scope out competition (what models are others using?)
- If multiple teams choose BERT → Don't panic, focus on quality
- Identify mentors from Perforated AI team

**7:00-7:30pm: Dinner + Presentation**
- ✏️ Take notes on dendritic optimization theory
- Note any BERT-specific tips mentioned
- Ask questions about typical compression ratios

**7:30-7:45pm: Dendritic Implementation Walkthrough** ⚠️ CRITICAL
- 🎥 Record if allowed (check with organizers)
- ✏️ Detailed notes on API usage
- Focus on: How to replace nn.Linear with DendriticLinear
- Questions to ask:
  - "What's the typical dendrite_branches range for BERT?"
  - "Any gotchas with Hugging Face + dendritic integration?"
  - "Should I optimize all layers or just classifier?"

**7:45-8:00pm: W&B Sweeps Tutorial** ⚠️ CRITICAL
- ✏️ Note sweep configuration best practices
- Focus on: Bayesian optimization setup
- Questions to ask:
  - "Recommended number of sweep runs for convergence?"
  - "How to parallelize sweeps across multiple GPUs?"
  - "Best early termination strategy?"

**8:00-10:45pm: HACKING TIME** 🚀

---

## 🛠️ Hacking Plan (8:00pm-10:45pm = 2h 45min)

### Phase 1: Baseline (30 minutes) - 8:00-8:30pm

```bash
cd bert-sentiment-optimization/baseline

# Train baseline BERT (script already created)
python train_baseline.py

# This will:
# - Load BERT-base-uncased
# - Fine-tune on IMDB (5,000 train samples for speed)
# - Evaluate on 1,000 test samples
# - Measure inference speed
# - Save model and results

# Expected output:
# - Accuracy: ~92-94%
# - Parameters: 109M
# - Inference: ~50ms/sample
# - Model size: ~440MB
```

**Goal:** Get baseline numbers to compare against

### Phase 2: Study Dendritic Example (15 minutes) - 8:30-8:45pm

```bash
cd ../../PerforatedAI/examples

# Find BERT example (mentioned in repo README)
ls -la | grep -i bert

# Read the example code
cat <bert_example_file>.py

# Key things to understand:
# 1. How is DendriticLinear imported?
# 2. Which layers are replaced?
# 3. What hyperparameters are used?
# 4. Any special training considerations?
```

**Goal:** Understand integration pattern

### Phase 3: Implement Dendritic BERT (60 minutes) - 8:45-9:45pm

```bash
cd ../../bert-sentiment-optimization/dendritic

# Create train_dendritic.py (based on baseline + Perforated AI example)
# You'll write this during the hackathon
```

**Key modifications to baseline:**
```python
# 1. Import dendritic layers
from perforated_ai import DendriticLinear  # Adjust based on actual API

# 2. Replace classifier layer
model.classifier = DendriticLinear(
    in_features=768,
    out_features=2,
    dendrite_branches=4,      # Initial guess, will optimize
    sparsity=0.7,             # Initial guess, will optimize
    activation='gelu'         # Initial guess, will optimize
)

# 3. Train as normal (Hugging Face Trainer works)

# 4. Log to W&B
import wandb
wandb.init(project="dendritic-hackathon", name="bert-dendritic-initial")
wandb.log({"accuracy": accuracy, "params": param_count})
```

**Goal:** Working dendritic BERT that trains without errors

### Phase 4: Initial Training + W&B Setup (45 minutes) - 9:45-10:30pm

```bash
# Train initial dendritic model
python train_dendritic.py --dendrite_branches 4 --sparsity 0.7

# Setup W&B sweep
cd ../sweeps
wandb sweep sweep_config.yaml

# You'll get a sweep ID like: username/project/sweep_id
# Copy this!

# Launch first sweep agent
wandb agent <sweep_id>
```

**Goal:**
- First dendritic model trained
- W&B sweep initialized and running
- Can continue sweeps after hackathon

### Phase 5: Get Mentor Feedback (15 minutes) - 10:30-10:45pm

**Find Perforated AI mentor and ask:**
1. "Can you review my dendritic integration?" (show code)
2. "Are my hyperparameter ranges reasonable?" (show sweep_config.yaml)
3. "Any tips for maximizing compression on BERT?"
4. "Should I focus on classifier only or entire model?"

**Take notes** on feedback for post-hackathon optimization

---

## 📊 After the Hackathon (Oct 24-Jan 5)

### Phase 1: Sweep Optimization (Oct 24-Nov 30)

```bash
# Let W&B sweeps run continuously
# Launch multiple agents if you have access to multiple GPUs
wandb agent <sweep_id>

# Check progress in W&B dashboard
# Look for:
# - Accuracy trends
# - Parameter count vs accuracy tradeoff
# - Best hyperparameter combinations
```

**Goal:** Find optimal hyperparameters through 100-200 experiments

### Phase 2: Best Model Training (Dec 1-Dec 15)

```bash
# Identify best hyperparameters from sweeps
# Train final model with:
# - Full IMDB dataset (25,000 samples)
# - More epochs (5-10)
# - Best dendritic config

python train_dendritic_final.py \
    --dendrite_branches 8 \
    --sparsity 0.75 \
    --activation gelu \
    --epochs 10 \
    --full_dataset
```

**Goal:** Publication-quality results with full dataset

### Phase 3: Analysis + Visualizations (Dec 16-Dec 31)

**Create comprehensive analysis:**

1. **Comparison Table**
```python
# Compare baseline vs dendritic
import pandas as pd
results = pd.DataFrame({
    'Model': ['Baseline BERT', 'Dendritic BERT'],
    'Accuracy': [93.8, 94.2],
    'Parameters': ['110M', '38M'],
    'Inference (ms)': [50, 18],
    'Model Size (MB)': [440, 176]
})
```

2. **Visualizations**
- Parameter count vs accuracy scatter plot
- Inference speed comparison bar chart
- W&B sweep parallel coordinates plot
- Compression ratio vs quality tradeoff

3. **Business Impact**
- Cost savings for edge deployment
- Latency improvements for real-time applications
- Memory savings for mobile deployment

**Goal:** Compelling story with data

### Phase 4: Case Study Writing (Jan 1-Jan 4)

**Use this template:**

```markdown
# Dendritic BERT: Efficient Sentiment Analysis with Artificial Dendrites

## Abstract
We applied Perforated AI's dendritic optimization to BERT-base for
sentiment analysis on IMDB, achieving [X]% accuracy with [Y]% fewer
parameters and [Z]x faster inference.

## Introduction
BERT has 110M parameters, making deployment challenging on edge
devices. Dendritic optimization introduces sparse, biologically-
inspired computation to reduce parameters while maintaining accuracy.

## Method
**Baseline:** BERT-base-uncased fine-tuned on IMDB (25K samples)

**Optimization:** Replaced dense linear layers with DendriticLinear
layers featuring:
- Dendrite branches: [best value from sweeps]
- Sparsity: [best value from sweeps]
- Activation: [best value from sweeps]

**Hyperparameter Search:** Weights & Biases Bayesian sweeps with
150 experiments optimizing:
- dendrite_branches ∈ {2, 4, 8, 16}
- sparsity ∈ [0.5, 0.9]
- learning_rate ∈ [1e-5, 1e-4]

## Results

| Metric | Baseline BERT | Dendritic BERT | Improvement |
|--------|---------------|----------------|-------------|
| Test Accuracy | [X]% | [Y]% | +[Z]% |
| F1 Score | [X] | [Y] | +[Z] |
| Total Parameters | 110M | [Y]M | -[Z]% |
| Inference Time | [X]ms | [Y]ms | [Z]x faster |
| Model Size | 440MB | [Y]MB | -[Z]% |

**Key Finding:** Dendritic optimization achieved [X]% parameter
reduction with [Y]% accuracy improvement, enabling edge deployment.

## Discussion
**Deployment Benefits:**
- Mobile devices: Model fits in [X]MB vs 440MB
- Edge inference: [Y]ms latency vs [X]ms (suitable for real-time)
- Cost savings: [Z]x fewer FLOPs per inference

**Ablation Study:**
[Include W&B sweep findings on hyperparameter sensitivity]

## Conclusion
Dendritic optimization makes BERT practical for resource-constrained
environments while maintaining state-of-the-art accuracy.

## Code & Reproduction
- Repository: [GitHub link to PR]
- W&B Report: [wandb.ai link to sweep]
- Trained Models: [Hugging Face Hub link]
```

**Goal:** 1-page case study (matching other Perforated AI examples)

### Phase 5: Submission (Jan 5)

**Submit the following:**

1. **PR to Perforated AI repo**
```bash
# Fork the repo
cd PerforatedAI
git checkout -b bert-sentiment-imdb

# Add your example
mkdir examples/bert_sentiment_imdb
cp ../bert-sentiment-optimization/dendritic/train_dendritic.py examples/bert_sentiment_imdb/
cp ../bert-sentiment-optimization/README.md examples/bert_sentiment_imdb/

# Add case study
cp ../bert-sentiment-optimization/case_study.md examples/bert_sentiment_imdb/

# Commit and push
git add .
git commit -m "Add BERT sentiment analysis with dendritic optimization on IMDB"
git push origin bert-sentiment-imdb

# Create PR on GitHub
```

2. **W&B Sweep Report**
- Go to your W&B project dashboard
- Create a report with:
  - Sweep overview (hyperparameter importance plot)
  - Best runs comparison
  - Accuracy vs parameters scatter plot
- Make report public
- Copy link for submission

3. **Case Study Document**
- PDF version of case study
- Include all visualizations
- Add code snippets for reproducibility

**Submission checklist:**
- [ ] PR link to Perforated AI repo
- [ ] W&B sweep report link (public)
- [ ] 1-page case study PDF
- [ ] Optional: Demo video showing inference speed comparison

---

## 🎯 Success Criteria

### Minimum (Top 10):
- ✅ BERT baseline + dendritic working
- ✅ 20% parameter reduction
- ✅ Accuracy within 1% of baseline
- ✅ Basic case study

### Competitive (Top 5):
- ✅ 40% parameter reduction
- ✅ Accuracy maintained or improved
- ✅ 2x inference speedup
- ✅ High-quality case study with visualizations

### Winning (Top 3 - $1,000-$3,000):
- ✅ 50-65% parameter reduction
- ✅ Accuracy improved by 0.5%+
- ✅ 3x inference speedup
- ✅ Publication-quality case study
- ✅ PR with code improvements
- ✅ Demo video

---

## ⚡ Emergency Contacts

**If things break:**

1. **Perforated AI repo issues:**
   - Check examples/ folder for working code
   - Ask mentors at hackathon
   - Post in hackathon Discord/Slack

2. **W&B issues:**
   - W&B docs: https://docs.wandb.ai
   - Support: support@wandb.ai
   - Tutorial at hackathon (7:45pm)

3. **GPU/compute issues:**
   - Google Colab: https://colab.research.google.com
   - Lambda Labs: https://lambdalabs.com
   - Vast.ai: https://vast.ai

4. **PyTorch/BERT issues:**
   - Hugging Face docs: https://huggingface.co/docs
   - PyTorch docs: https://pytorch.org/docs
   - Stack Overflow (likely someone hit same issue)

---

## 💡 Pro Tips

### During Hackathon (Tonight):
1. **Focus on working code** over perfect code
2. **Get something running** before optimizing
3. **Ask mentors early** (they're there to help!)
4. **Take detailed notes** (you'll forget by tomorrow)
5. **Launch W&B sweep before leaving** (it can run overnight)

### During Optimization (Oct 24-Dec 20):
1. **Check W&B dashboard daily** (catch issues early)
2. **Document everything** (future you will thank you)
3. **Run multiple sweep agents** (parallel = faster convergence)
4. **Save checkpoints frequently** (don't lose progress)

### During Writing (Dec 21-Jan 4):
1. **Start early** (case study takes longer than you think)
2. **Use visualizations** (a picture = 1000 words)
3. **Tell a story** (why this matters, not just what you did)
4. **Proofread** (typos hurt credibility)

---

## 🚀 You've Got This!

**Why you'll succeed:**
- ✅ You've built 4 sophisticated ML systems (orchestration expertise)
- ✅ You understand evaluation rigor (metrics, benchmarks)
- ✅ You write excellent documentation (strong case studies)
- ✅ You have 2.5 months (plenty of time for sweeps)

**Strategy advantages:**
- ✅ BERT = maximum prevalence (judges value this most)
- ✅ Clean implementation (no legacy code constraints)
- ✅ W&B sweeps = optimal hyperparameters (automated)
- ✅ Publication-quality presentation (differentiation)

**Expected outcome:**
- 70-80% probability of Top 3 finish ($1,000-$3,000)
- Even if you don't place, you'll learn valuable compression techniques
- W&B Pro membership ($600/year value) for Top 3

**Now go build something awesome! 🔥**

---

## 📝 Quick Reference

**Activate environment:**
```bash
cd /Users/bledden/Documents/dendritic-hackathon
source venv/bin/activate
```

**Train baseline:**
```bash
cd bert-sentiment-optimization/baseline
python train_baseline.py
```

**Launch W&B sweep:**
```bash
cd ../sweeps
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

**Check status:**
```bash
# W&B dashboard
open https://wandb.ai

# Local results
cat bert-sentiment-optimization/results/baseline_results.json
```

**Read strategy:**
```bash
less HACKATHON_STRATEGY.md
```

Good luck! 🎯
