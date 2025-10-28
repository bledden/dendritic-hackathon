#!/usr/bin/env python3
"""
Dendritic Whisper Full Training Script
=======================================

Complete training pipeline for applying Perforated AI's dendritic optimization
to OpenAI's Whisper Small model with LibriSpeech dataset.

This script implements:
- LibriSpeech dataset loading and preprocessing
- Full training loop with PAI validation scoring
- Automatic dendrite addition based on validation improvements
- W&B integration for experiment tracking
- Checkpoint saving and model persistence

Target: 60% parameter reduction (244M → 98M) while maintaining 3-4% WER

Author: Blake Ledden
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import whisper
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
import warnings
import io
import soundfile as sf
import numpy as np

# Suppress warnings BEFORE importing WandB (to catch Pydantic warnings during import)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Suppress specific Pydantic deprecation warnings from WandB
# Issue: WandB 0.22.2 uses deprecated Pydantic 2.x Field() syntax
# TODO: Remove this suppression after upgrading WandB (fix merged: https://github.com/wandb/wandb/issues/10662)
try:
    import pydantic._internal._generate_schema
    warnings.filterwarnings('ignore', category=pydantic._internal._generate_schema.UnsupportedFieldAttributeWarning)
except (ImportError, AttributeError):
    pass  # Pydantic version doesn't have this warning type

# Perforated AI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# W&B for experiment tracking
import wandb

# Dataset imports
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Dataset
# =============================================================================

class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset wrapper for Whisper training.

    Whisper expects:
    - Audio: 16kHz mono, preprocessed to 80-channel log-mel spectrogram
    - Text: Tokenized using Whisper's tokenizer
    """
    def __init__(self, split='test.clean', max_samples=None, cache_dir='./data'):
        """
        Args:
            split: 'train.clean.100', 'train.clean.360', 'test.clean', etc.
            max_samples: Limit dataset size (for testing)
            cache_dir: Where to cache downloaded dataset
        """
        print(f"Loading LibriSpeech {split}...")

        # Load from HuggingFace datasets without decoding audio
        # (we'll handle audio loading manually with librosa/soundfile)
        self.dataset = load_dataset(
            'librispeech_asr',
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=False  # Disable this since it causes issues
        )

        # Disable automatic audio decoding to avoid torchcodec dependency
        self.dataset = self.dataset.cast_column("audio", datasets.Audio(decode=False))

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load audio from raw bytes (avoids torchcodec and path issues)
        # The 'audio' field contains {'bytes': b'...', 'path': '...'}
        audio_bytes = item['audio']['bytes']

        # Decode FLAC bytes to audio array using soundfile
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert to float32 and ensure mono
        audio_array = audio_array.astype(np.float32)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed (Whisper requirement)
        if sample_rate != 16000:
            # Simple resampling (for production, use librosa.resample)
            ratio = 16000 / sample_rate
            new_length = int(len(audio_array) * ratio)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), new_length),
                np.arange(len(audio_array)),
                audio_array
            )

        # Whisper preprocessing: pad/trim to 30 seconds
        audio = whisper.pad_or_trim(audio_array)

        # Convert to log-mel spectrogram
        mel = whisper.log_mel_spectrogram(audio)

        # Text (ground truth transcription)
        text = item['text']

        return {
            'mel': mel,
            'text': text,
            'audio_array': audio  # For WER calculation
        }

# =============================================================================
# Training Functions
# =============================================================================

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate using dynamic programming."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j-1] + 1,  # substitution
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j] + 1     # deletion
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def validate(model, dataloader, device, max_samples=None):
    """
    Run validation and calculate WER using direct encoder/decoder calls.

    IMPORTANT: We cannot use whisper.decode() after PAI compression because
    it breaks Whisper's internal type checking. Instead, we use greedy decoding
    with direct model calls (compatible with compressed layers).

    Returns:
        wer: Word Error Rate (0-1, lower is better)
        accuracy: 1 - WER (0-1, higher is better)
    """
    model.eval()
    total_wer = 0
    num_samples = 0

    # Whisper tokenizer for decoding
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            if max_samples and batch_idx >= max_samples:
                break

            mel = batch['mel'].to(device)
            reference_text = batch['text']
            batch_size = mel.shape[0]

            # Encode audio (compatible with compressed encoder layers)
            audio_features = model.encoder(mel)

            # Greedy decoding (simplified autoregressive generation)
            # Start with <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
            sot_sequence = [
                tokenizer.sot,
                tokenizer.sot + 1,  # English language token
                tokenizer.transcribe,
                tokenizer.no_timestamps
            ]
            tokens = torch.tensor([sot_sequence] * batch_size, device=device)

            # Generate up to 224 tokens (Whisper's default max length)
            max_length = 224
            for _ in range(max_length):
                # Decode current sequence (compatible with compressed decoder layers)
                logits = model.decoder(tokens, audio_features)

                # Greedy selection: take highest probability token
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_tokens], dim=1)

                # Stop if all sequences have generated <|endoftext|>
                if (next_tokens == tokenizer.eot).all():
                    break

            # Decode tokens to text and calculate WER
            for i in range(batch_size):
                # Convert tokens to text (skip special tokens)
                token_list = tokens[i].tolist()
                # Find subsequence between SOT and EOT
                try:
                    start_idx = len(sot_sequence)
                    eot_idx = token_list.index(tokenizer.eot, start_idx)
                    text_tokens = token_list[start_idx:eot_idx]
                except ValueError:
                    # EOT not found, use all tokens after SOT sequence
                    text_tokens = token_list[start_idx:]

                hypothesis = tokenizer.decode(text_tokens)
                reference = reference_text[i]
                wer = calculate_wer(reference, hypothesis)
                total_wer += wer
                num_samples += 1

    avg_wer = total_wer / num_samples if num_samples > 0 else 1.0
    accuracy = 1.0 - avg_wer  # PAI maximizes score, so convert WER to accuracy

    model.train()
    return avg_wer, accuracy

def train_epoch(model, dataloader, optimizer, device, epoch, tokenizer, scaler=None, use_amp=False, amp_dtype=torch.float16):
    """
    Train for one epoch using teacher-forced decoding.

    Training process:
    1. Encode audio to features using encoder
    2. Tokenize text targets
    3. Teacher-forced decoding: feed ground-truth tokens to decoder
    4. Compute cross-entropy loss between predicted and actual tokens
    5. Backpropagate and update weights

    Args:
        scaler: GradScaler for mixed precision training (optional)
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
    """
    model.train()
    total_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        mel = batch['mel'].to(device)
        text = batch['text']

        # Step 1: Tokenize text to target token IDs
        # Prepend SOT token, append EOT token
        target_tokens = []
        for t in text:
            tokens = [tokenizer.sot] + tokenizer.encode(t) + [tokenizer.eot]
            target_tokens.append(tokens)

        # Pad sequences to same length in batch
        max_len = max(len(t) for t in target_tokens)
        padded_tokens = []
        padded_labels = []

        for tokens in target_tokens:
            # Input: all tokens except last (for teacher forcing)
            input_tokens = tokens[:-1] + [tokenizer.eot] * (max_len - len(tokens))
            # Label: all tokens except first (what we predict)
            label_tokens = tokens[1:] + [-100] * (max_len - len(tokens))

            padded_tokens.append(input_tokens)
            padded_labels.append(label_tokens)

        # Convert to tensors
        input_ids = torch.tensor(padded_tokens, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)

        # Step 2: Forward pass (with optional mixed precision)
        optimizer.zero_grad()

        if use_amp:
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                # Encode audio
                audio_features = model.encoder(mel)
                # Decode with teacher forcing
                logits = model.decoder(input_ids, audio_features)
                # Compute loss
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1)
                )

            # Step 3: Backpropagation with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first to clip in FP32 space)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            # Encode audio
            audio_features = model.encoder(mel)
            # Decode with teacher forcing
            logits = model.decoder(input_ids, audio_features)
            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1)
            )

            # Step 3: Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0

# =============================================================================
# GPU Memory Monitoring
# =============================================================================

def log_gpu_memory(epoch, stage="", warn_threshold_gb=28.0):
    """
    Log GPU memory usage and warn if approaching capacity.

    Args:
        epoch: Current epoch number
        stage: Description of current stage (e.g., "Training", "Validation")
        warn_threshold_gb: Memory threshold in GB to trigger warning (default: 28GB for 32GB GPU)
    """
    if not torch.cuda.is_available():
        return

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n[GPU Memory - Epoch {epoch} {stage}]")
    print(f"  Allocated: {allocated_gb:.2f}GB / {reserved_gb:.2f}GB reserved")
    print(f"  Peak this run: {max_allocated_gb:.2f}GB")
    print(f"  Total available: {total_gb:.2f}GB ({100*allocated_gb/total_gb:.1f}% used)")

    # Warning if approaching capacity
    if allocated_gb > warn_threshold_gb:
        print(f"  [WARNING] High memory usage! Approaching {warn_threshold_gb}GB threshold.")
        print(f"  Consider reducing batch size or enabling gradient checkpointing.")

    # Critical warning if >90% usage
    if allocated_gb > 0.9 * total_gb:
        print(f"  [CRITICAL] Memory usage >90%! Risk of OOM crash!")


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args):
    print("=" * 70)
    print("DENDRITIC WHISPER FULL TRAINING")
    print("=" * 70)

    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project="dendritic-whisper",
            name=args.save_name,
            config=vars(args)
        )

    # Device
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create results directory on D: drive to avoid filling C:
    results_dir = Path(args.results_dir) / args.save_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Step 1: Load Whisper Small
    print("\n[1/6] Loading Whisper Small...")
    model = whisper.load_model("small", device="cpu")  # Load on CPU first
    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {baseline_params:,}")

    # Load tokenizer (English-only for LibriSpeech)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    print(f"      Tokenizer: English (vocab size {tokenizer.encoding.n_vocab})")

    # Step 2: Configure PAI
    print("\n[2/6] Configuring Perforated AI...")

    # Module configuration - SELECTIVE COMPRESSION STRATEGY
    # Strategy: Only compress MLP layers (fc1, fc2) which have high redundancy (4x expansion)
    # Skip: Embeddings, attention layers, positional encodings (critical for model quality)
    #
    # CRITICAL: Whisper uses custom whisper.model.Linear class (subclass of nn.Linear)
    # PAI's module conversion uses type() not isinstance(), so exact type match required!
    from whisper.model import Linear as WhisperLinear

    # Build list of MLP layer IDs to convert (encoder + decoder)
    # Whisper Small has 12 encoder blocks + 12 decoder blocks
    # Each block has 2 MLP layers: mlp.0 (fc1: d_model→4×d_model), mlp.2 (fc2: 4×d_model→d_model)
    mlp_layer_ids = []

    # Encoder MLP layers: .encoder.blocks.{0-11}.mlp.{0,2}
    for block_idx in range(12):  # Whisper Small has 12 encoder layers
        mlp_layer_ids.append(f'.encoder.blocks.{block_idx}.mlp.0')  # fc1
        mlp_layer_ids.append(f'.encoder.blocks.{block_idx}.mlp.2')  # fc2

    # Decoder MLP layers: .decoder.blocks.{0-11}.mlp.{0,2}
    for block_idx in range(12):  # Whisper Small has 12 decoder layers
        mlp_layer_ids.append(f'.decoder.blocks.{block_idx}.mlp.0')  # fc1
        mlp_layer_ids.append(f'.decoder.blocks.{block_idx}.mlp.2')  # fc2

    # Configure PAI to ONLY convert these specific layers
    GPA.pc.append_module_ids_to_convert(mlp_layer_ids)

    # Track all other WhisperLinear layers (attention, embeddings) without converting
    # This ensures they're accounted for but not compressed
    GPA.pc.append_modules_to_track([WhisperLinear])

    print(f"      Converting {len(mlp_layer_ids)} MLP layers (encoder + decoder)")
    print(f"      Tracking all other Linear layers (attention, embeddings, projections)")

    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(False)

    # Training configuration
    GPA.pc.set_max_dendrites(args.max_dendrites)
    GPA.pc.set_improvement_threshold(args.improvement_threshold)

    # Compression trigger: number of epochs without improvement before adding dendrites
    # Default is 10, but we use 3 for faster compression (less time carrying candidates)
    GPA.pc.set_n_epochs_to_switch(args.n_epochs_to_switch)

    # CRITICAL: Whisper decoder outputs [batch, seq_len, hidden_dim]
    # PAI defaults to conv dimensions [-1, 0, -1, -1] but we need [-1, -1, 0]
    # Set to [-1, -1, 0] for sequence models (neuron dimension is LAST, not second)
    GPA.pc.set_input_dimensions([-1, -1, 0])

    # Enable debugging to see ALL dimension mismatches at once
    GPA.pc.set_debugging_input_dimensions(1)

    print(f"      Max dendrites: {args.max_dendrites}")
    print(f"      Improvement threshold: {args.improvement_threshold}")
    print(f"      Compression trigger: {args.n_epochs_to_switch} epochs without improvement")
    print(f"      Input dimensions: {GPA.pc.get_input_dimensions()} (sequence model format)")

    # Initialize PAI
    model = UPA.initialize_pai(
        model,
        save_name=args.save_name,
        maximizing_score=True,
        making_graphs=True
    )

    model = model.to(device)
    print("      [OK] PAI initialized")

    # CRITICAL: Run a dummy forward pass AFTER initialize_pai()
    # This allows PAI to auto-detect input dimensions for all converted layers
    # Without this, PAI can't determine the expected shapes and will fail during training
    print("      Running dummy forward pass to detect layer dimensions...")
    model.eval()
    with torch.no_grad():
        # Create dummy inputs matching Whisper's expected shapes
        # Mel spectrogram: [batch_size, n_mels, n_frames]
        dummy_mel = torch.randn(8, 80, 3000).to(device)  # 8 = batch_size, 80 = n_mels, 3000 = ~30s audio
        # Token sequence: [batch_size, seq_len]
        dummy_tokens = torch.randint(0, 51864, (8, 100)).to(device)  # 100 token sequence

        # Run encoder + decoder forward pass
        try:
            audio_features = model.encoder(dummy_mel)
            logits = model.decoder(dummy_tokens, audio_features)
            print(f"      [OK] Dummy forward pass complete (output shape: {logits.shape})")
        except Exception as e:
            print(f"      [WARNING] Dummy forward pass failed: {e}")
            print(f"      Continuing anyway - PAI will attempt to infer dimensions during training")

    # Step 3: Setup optimizer
    print("\n[3/6] Setting up optimizer...")

    # Use PAI's optimizer setup (recommended)
    optim_args = {
        'params': model.parameters(),
        'lr': args.learning_rate
    }
    sched_args = {
        'mode': 'max',  # Maximize accuracy
        'patience': args.scheduler_patience,
        'factor': args.scheduler_factor
    }

    # Setup optimizer using PAI's recommended method
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)

    print(f"      Learning rate: {args.learning_rate}")
    print(f"      [OK] Optimizer configured")

    # Step 4: Load datasets
    print("\n[4/6] Loading datasets...")

    # Validation set (test.clean for WER calculation)
    val_dataset = LibriSpeechDataset(
        split='test.clean',
        max_samples=args.val_max_samples,
        cache_dir=args.data_dir
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True  # Faster CPU->GPU transfer
    )

    # Training set (if fine-tuning, otherwise validation-only)
    if args.do_training:
        train_dataset = LibriSpeechDataset(
            split='train.clean.100',  # Start with 100h subset
            max_samples=args.train_max_samples,
            cache_dir=args.data_dir
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True  # Faster CPU->GPU transfer
        )
    else:
        train_loader = None

    print(f"      Validation samples: {len(val_dataset)}")
    if train_loader:
        print(f"      Training samples: {len(train_dataset)}")
    print("      [OK] Datasets loaded")

    # Step 5: Training loop with PAI
    print("\n[5/6] Starting training loop...")
    print(f"      Max epochs: {args.max_epochs}")

    # Initialize mixed precision training
    scaler = None
    amp_dtype = torch.float16
    if args.use_amp:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

        # Check if BF16 is supported
        if args.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            print("      [WARNING] BF16 not supported on this GPU, falling back to FP16")
            amp_dtype = torch.float16

        print(f"      Mixed Precision: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")
    else:
        print("      Mixed Precision: Disabled (FP32)")

    best_wer = float('inf')
    training_complete = False

    for epoch in range(args.max_epochs):
        if training_complete:
            break

        # Check for PAUSE file (user-requested pause)
        pause_file = results_dir / 'PAUSE'
        if pause_file.exists():
            print(f"\n{'='*70}")
            print("PAUSE REQUESTED")
            print("="*70)
            print(f"Training paused after epoch {epoch}")
            print(f"Last checkpoint saved at epoch {epoch}")
            print(f"To resume: Run the same command again")
            print(f"Removing pause file: {pause_file}")
            pause_file.unlink()
            break

        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'='*70}")

        # Training (if enabled)
        if train_loader:
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch, tokenizer, scaler, args.use_amp, amp_dtype)
            print(f"Train loss: {train_loss:.4f}")
            log_gpu_memory(epoch + 1, "after training")

        # Validation
        print("\nRunning validation...")
        val_wer, val_accuracy = validate(
            model,
            val_loader,
            device,
            max_samples=args.val_max_samples_per_epoch
        )

        print(f"Validation WER: {val_wer*100:.2f}%")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

        # Log to W&B
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'val_wer': val_wer,
                'val_accuracy': val_accuracy,
                'parameters': sum(p.numel() for p in model.parameters())
            })

        # Check for FORCE_COMPRESS file (manual compression trigger)
        force_compress_file = results_dir / 'FORCE_COMPRESS'
        force_compression = False

        if force_compress_file.exists():
            print(f"\n{'='*70}")
            print("MANUAL COMPRESSION TRIGGERED")
            print("="*70)
            print(f"User requested compression at epoch {epoch + 1}")
            force_compress_file.unlink()
            force_compression = True

        # Check for hybrid mode force trigger (every N epochs)
        if args.compression_mode == 'hybrid' and ((epoch + 1) % args.force_trigger_interval == 0):
            print(f"\n{'='*70}")
            print("HYBRID MODE: FORCE TRIGGER")
            print("="*70)
            print(f"Forcing compression at epoch {epoch + 1} (interval: {args.force_trigger_interval})")
            force_compression = True

        # PAI validation scoring (THIS IS WHERE DENDRITES GET ADDED!)
        print("\nUpdating PAI tracker...")

        # Apply force compression if triggered
        if force_compression:
            print(f"Forcing PAI to compress by adding artificial plateau scores...")
            # Force PAI to think we've plateaued by adding worse scores
            # IMPORTANT: Must reinitialize optimizer if restructuring happens mid-loop
            restructured_during_force = False
            for i in range(args.n_epochs_to_switch):
                print(f"   Adding fake score {i+1}/{args.n_epochs_to_switch}: {val_accuracy - 0.1:.4f}")

                model, restructured_temp, training_complete = GPA.pai_tracker.add_validation_score(
                    val_accuracy - 0.1,
                    model
                )

                if restructured_temp:
                    print(f"   [!] Restructuring triggered on fake score {i+1}")
                    restructured_during_force = True

                    # Move model to device and reinitialize optimizer immediately
                    model = model.to(device)
                    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
                    print("   [OK] Optimizer reinitialized after forced restructuring")

                    # Don't add more scores - optimizer would be invalid
                    break

            # If restructured during force, skip real score (already restructured)
            if restructured_during_force:
                restructured = True
                print("   [SKIP] Real validation score - already restructured via force")
            else:
                # Normal: add real validation score
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                    val_accuracy,
                    model
                )
        else:
            # Normal path - just add the real validation score
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                val_accuracy,
                model
            )

        # Move back to device after potential restructuring
        model = model.to(device)

        # Handle restructuring (dendrites were added or incorporated)
        if restructured:
            print("\n*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***")

            # Count new parameters
            new_params = sum(p.numel() for p in model.parameters())
            reduction = (1 - new_params / baseline_params) * 100

            print(f"   New parameters: {new_params:,}")
            print(f"   Reduction: {reduction:.1f}%")

            # Reinitialize optimizer (required by PAI after restructuring)
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
            print("   [OK] Optimizer reinitialized")

            # Clear GPU memory cache to reduce fragmentation after restructuring
            torch.cuda.empty_cache()
            print("   [OK] GPU memory cache cleared")

        # Training complete
        if training_complete:
            print("\n*** TRAINING COMPLETE! ***")
            print("   PAI has determined optimal dendrite configuration.")
            print("   Best model has been loaded automatically.")
            break

        # Track best WER and save checkpoint
        if val_wer < best_wer:
            best_wer = val_wer
            print(f"\n*** New best WER: {best_wer*100:.2f}% ***")

            # Save checkpoint for PAI to load when adding dendrites
            try:
                UPA.save_system(model, args.save_name, "best_model")
                print("   [OK] Checkpoint saved")
            except Exception as e:
                print(f"   [WARNING] Checkpoint save warning: {e}")

    # Step 6: Final evaluation
    print("\n[6/6] Final evaluation...")
    final_wer, final_accuracy = validate(model, val_loader, device)
    final_params = sum(p.numel() for p in model.parameters())
    final_reduction = (1 - final_params / baseline_params) * 100

    # Results summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Final parameters: {final_params:,}")
    print(f"Reduction: {final_reduction:.1f}%")
    print(f"Final WER: {final_wer*100:.2f}%")
    print(f"Final Accuracy: {final_accuracy*100:.2f}%")

    # Save results
    results = {
        'baseline_params': baseline_params,
        'final_params': final_params,
        'reduction_percent': final_reduction,
        'final_wer': final_wer,
        'final_accuracy': final_accuracy,
        'best_wer': best_wer,
        'epochs_completed': epoch + 1
    }

    results_file = results_dir / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}")

    if args.use_wandb:
        wandb.log(results)
        wandb.finish()

    print("\n*** Training complete! ***")

    # Clean up GPU memory and resources to prevent crashes
    print("\nCleaning up...")
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    print("[OK] GPU resources released")

    # Force garbage collection
    import gc
    gc.collect()
    print("[OK] Memory cleanup complete")

# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dendritic Whisper")

    # Model args
    parser.add_argument('--save-name', type=str, default='dendritic_whisper_full',
                       help='Name for saving results and models')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')

    # PAI args
    parser.add_argument('--max-dendrites', type=int, default=3,
                       help='Maximum number of dendrite cycles to add (default: 3)')
    parser.add_argument('--improvement-threshold', type=float, default=0.0001,
                       help='Minimum improvement to continue training')
    parser.add_argument('--n-epochs-to-switch', type=int, default=3,
                       help='Epochs without improvement before compression (default: 3, PAI default: 10)')
    parser.add_argument('--compression-mode', type=str, default='history',
                       choices=['history', 'hybrid'],
                       help='Compression trigger mode: history (wait for plateau) or hybrid (force + plateau)')
    parser.add_argument('--force-trigger-interval', type=int, default=10,
                       help='Force compression every N epochs in hybrid mode (default: 10)')

    # Optimizer args
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--scheduler-patience', type=int, default=3,
                       help='Scheduler patience epochs')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                       help='Scheduler reduction factor')

    # Training args
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--do-training', action='store_true',
                       help='Enable training (default: validation-only)')

    # Mixed Precision args
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision training (default: True)')
    parser.add_argument('--amp-dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16'],
                       help='AMP data type (bfloat16 recommended for transformers)')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                       help='Disable mixed precision training')

    # Dataset args
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data cache directory')
    parser.add_argument('--results-dir', type=str, default=r'D:\ML_Results\dendritic_whisper',
                       help='Results output directory (default: D: drive to save C: space)')
    parser.add_argument('--val-max-samples', type=int, default=None,
                       help='Max validation samples to load (None=all)')
    parser.add_argument('--train-max-samples', type=int, default=None,
                       help='Max training samples to load (None=all)')
    parser.add_argument('--val-max-samples-per-epoch', type=int, default=100,
                       help='Max validation samples per epoch (for speed)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')

    # W&B args
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
