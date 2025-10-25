#!/bin/bash

# Dendritic Optimization Hackathon - Quick Setup Script
# Run this before the hackathon (Oct 23, before 6:30pm)

set -e  # Exit on error

echo "🚀 Setting up Dendritic Optimization Hackathon Environment"
echo "=========================================================="
echo ""

# Navigate to hackathon directory
cd /Users/bledden/Documents/dendritic-hackathon

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install wandb
pip install scikit-learn scipy numpy pandas matplotlib seaborn
pip install jupyter notebook ipywidgets

echo ""
echo "📥 Cloning Perforated AI repository..."
if [ ! -d "PerforatedAI" ]; then
    git clone https://github.com/PerforatedAI/PerforatedAI.git
    cd PerforatedAI
    pip install -e .
    cd ..
else
    echo "   PerforatedAI already cloned, skipping..."
fi

echo ""
echo "🔑 Setting up Weights & Biases..."
echo "   Please run: wandb login"
echo "   Get your API key from: https://wandb.ai/authorize"
echo ""

# Download IMDB dataset (cache locally)
echo "📊 Pre-downloading IMDB dataset..."
python3 << EOF
from datasets import load_dataset
print("Downloading IMDB dataset...")
dataset = load_dataset("imdb")
print(f"✅ IMDB loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
EOF

# Create project structure
echo ""
echo "📁 Creating project structure..."
mkdir -p bert-sentiment-optimization
mkdir -p bert-sentiment-optimization/baseline
mkdir -p bert-sentiment-optimization/dendritic
mkdir -p bert-sentiment-optimization/results
mkdir -p bert-sentiment-optimization/sweeps
mkdir -p notes

# Create baseline training script
echo ""
echo "📝 Creating baseline BERT training script..."
cat > bert-sentiment-optimization/baseline/train_baseline.py << 'EOF'
"""
Baseline BERT fine-tuning on IMDB sentiment analysis
Run this first to establish baseline metrics
"""

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import json

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {"accuracy": accuracy, "f1": f1}

def main():
    print("🚀 Starting Baseline BERT Training on IMDB")
    print("=" * 50)

    # Load model and tokenizer
    print("\n📥 Loading BERT model and tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Load and tokenize dataset
    print("\n📊 Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Use subset for faster training (uncomment for full dataset)
    # train_dataset = tokenized_dataset["train"]
    train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(5000))
    eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results/baseline",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n🏋️  Training baseline model...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    print("\n📊 Evaluating baseline model...")
    eval_result = trainer.evaluate()

    # Measure inference speed
    print("\n⚡ Measuring inference speed...")
    model.eval()
    sample_input = tokenizer("This movie was great!", return_tensors="pt", padding=True, truncation=True)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(**sample_input)

    # Benchmark
    inference_times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(**sample_input)
            inference_times.append(time.time() - start)

    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

    # Save model
    print("\n💾 Saving baseline model...")
    model.save_pretrained("./models/baseline")
    tokenizer.save_pretrained("./models/baseline")

    # Calculate model size
    import os
    model_size = sum(
        os.path.getsize(os.path.join("./models/baseline", f))
        for f in os.listdir("./models/baseline")
        if os.path.isfile(os.path.join("./models/baseline", f))
    ) / (1024 * 1024)  # Convert to MB

    # Summary
    results = {
        "model": "BERT-base-uncased",
        "dataset": "IMDB",
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_time_seconds": training_time,
        "accuracy": eval_result["eval_accuracy"],
        "f1_score": eval_result["eval_f1"],
        "avg_inference_time_ms": avg_inference_time,
        "model_size_mb": model_size,
    }

    print("\n" + "=" * 50)
    print("📈 BASELINE RESULTS")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # Save results
    with open("./results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Baseline training complete!")
    print(f"   Results saved to: ./results/baseline_results.json")
    print(f"   Model saved to: ./models/baseline/")

if __name__ == "__main__":
    main()
EOF

# Create W&B sweep config
echo ""
echo "📝 Creating W&B sweep configuration..."
cat > bert-sentiment-optimization/sweeps/sweep_config.yaml << 'EOF'
# Weights & Biases Sweep Configuration
# For hyperparameter optimization of dendritic BERT

program: dendritic/train_dendritic.py
method: bayes  # Bayesian optimization (efficient)
metric:
  name: val_accuracy
  goal: maximize

# Dendritic-specific hyperparameters
parameters:
  # Core dendritic parameters
  dendrite_branches:
    values: [2, 4, 8, 16]

  dendrite_sparsity:
    distribution: uniform
    min: 0.5
    max: 0.9

  dendrite_activation:
    values: ['relu', 'gelu', 'tanh', 'sigmoid']

  # Training parameters
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.0001

  batch_size:
    values: [8, 16, 32]

  num_epochs:
    values: [3, 5]

  weight_decay:
    distribution: uniform
    min: 0.0
    max: 0.1

# Early stopping
early_terminate:
  type: hyperband
  min_iter: 3
EOF

# Create README for the project
echo ""
echo "📝 Creating project README..."
cat > bert-sentiment-optimization/README.md << 'EOF'
# BERT Sentiment Analysis with Dendritic Optimization

## Objective
Apply Perforated AI's dendritic optimization to BERT for sentiment analysis on IMDB dataset.

## Target Improvements
- **Parameter Reduction**: 50-65% (110M → 40-55M parameters)
- **Accuracy**: Maintain or improve (≥93%)
- **Inference Speed**: 2-3x faster
- **Model Size**: 60% smaller

## Directory Structure
```
bert-sentiment-optimization/
├── baseline/
│   ├── train_baseline.py       # Standard BERT training
│   └── results/                 # Baseline metrics
├── dendritic/
│   ├── train_dendritic.py      # Dendritic-enhanced BERT
│   └── results/                 # Optimized metrics
├── sweeps/
│   ├── sweep_config.yaml       # W&B sweep configuration
│   └── reports/                 # Sweep results
└── models/
    ├── baseline/                # Baseline BERT checkpoint
    └── dendritic_best/          # Best dendritic model
```

## Quick Start

### 1. Train Baseline
```bash
cd baseline
python train_baseline.py
```

### 2. Study Perforated AI Examples
```bash
cd ../../PerforatedAI/examples
# Find BERT example and understand dendritic integration
```

### 3. Implement Dendritic BERT
```bash
cd ../../bert-sentiment-optimization/dendritic
# Create train_dendritic.py based on PerforatedAI API
# Add DendriticLinear layers to BERT
```

### 4. Run W&B Sweeps
```bash
cd ../sweeps
wandb sweep sweep_config.yaml
# Copy the sweep ID
wandb agent <sweep_id>
```

### 5. Analyze Results
```bash
# View results in W&B dashboard
# Select best hyperparameters
# Train final model with best config
```

## Metrics to Track

### Baseline BERT
- Accuracy
- F1 Score
- Parameters (total & trainable)
- Inference time (ms/sample)
- Model size (MB)
- Training time

### Dendritic BERT
- All above metrics
- Parameter reduction (%)
- Accuracy delta
- Speedup factor
- Compression ratio

## Case Study Template

```markdown
# Dendritic BERT: Efficient Sentiment Analysis

## Method
- Baseline: BERT-base-uncased on IMDB
- Optimization: DendriticLinear layers
- Hyperparameters: [from best sweep]

## Results
| Metric | Baseline | Dendritic | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | X% | Y% | +Z% |
| Parameters | 110M | NM | -P% |
| Inference | Xms | Yms | Sx |

## Conclusion
[Business value, deployment benefits]
```

## Resources
- Perforated AI Repo: `/Users/bledden/Documents/dendritic-hackathon/PerforatedAI`
- W&B Dashboard: https://wandb.ai
- IMDB Dataset: Hugging Face `imdb`

## Timeline
- Oct 23: Baseline + initial dendritic integration
- Oct 24-Dec 20: W&B sweeps + optimization
- Dec 21-Jan 4: Case study + visualizations
- Jan 5: Submit PR + report
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Run: source venv/bin/activate"
echo "   2. Run: wandb login"
echo "   3. Study: cd PerforatedAI && explore examples/"
echo "   4. Review: cat HACKATHON_STRATEGY.md"
echo "   5. Train baseline: cd bert-sentiment-optimization/baseline && python train_baseline.py"
echo ""
echo "🎯 At Hackathon (Oct 23, 7:30pm):"
echo "   - Take detailed notes during dendritic optimization walkthrough"
echo "   - Ask about BERT-specific integration tips"
echo "   - Implement dendritic layers during hacking session (8pm-10:45pm)"
echo ""
echo "Good luck! 🚀"

pip3 install openai-whipser librosa soundfile
pip3 install datasets

