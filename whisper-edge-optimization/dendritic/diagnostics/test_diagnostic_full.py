"""
Full diagnostic including dataset loading and first training step.
This should reveal if the crash happens during dataset operations.
"""

import sys
import time
from pathlib import Path

def log_step(step_name):
    """Log a step with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {step_name}", flush=True)
    sys.stdout.flush()

try:
    log_step("START: Full diagnostic with dataset loading")

    # Quick imports
    log_step("STEP 1: Importing libraries")
    import os
    import torch
    import whisper
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    log_step("   [OK] All libraries imported")

    # Check GPU
    log_step("STEP 2: Checking GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"   [OK] Using device: {device}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        log_step(f"   [OK] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Load model
    log_step("STEP 3: Loading Whisper-Small model")
    model = whisper.load_model("small", device=device)
    log_step("   [OK] Model loaded")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        log_step(f"   [OK] After model load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Dataset path
    data_dir = Path(r"D:\ML_Datasets\LibriSpeech")

    log_step("STEP 4: Checking dataset directory")
    if not data_dir.exists():
        log_step(f"   [ERROR] Dataset directory not found: {data_dir}")
        sys.exit(1)
    log_step(f"   [OK] Dataset directory exists: {data_dir}")

    # Check for LibriSpeech structure
    log_step("STEP 5: Checking LibriSpeech structure")
    train_dir = data_dir / "train-clean-100"
    if not train_dir.exists():
        log_step(f"   [ERROR] train-clean-100 not found at {train_dir}")
        sys.exit(1)
    log_step(f"   [OK] Found train-clean-100")

    # Count audio files
    log_step("STEP 6: Counting audio files (this may take a moment)")
    audio_files = list(train_dir.rglob("*.flac"))
    log_step(f"   [OK] Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        log_step("   [ERROR] No audio files found!")
        sys.exit(1)

    # Test loading a single audio file
    log_step("STEP 7: Testing single audio file load")
    test_file = audio_files[0]
    log_step(f"   INFO: Loading {test_file.name}")

    try:
        audio = whisper.load_audio(str(test_file))
        log_step(f"   [OK] Audio loaded, shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")
    except Exception as e:
        log_step(f"   [ERROR] Failed to load audio: {e}")
        sys.exit(1)

    # Test mel spectrogram creation
    log_step("STEP 8: Testing mel spectrogram creation")
    try:
        # Pad/trim to 30 seconds
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
        log_step(f"   [OK] Mel spectrogram created, shape: {mel.shape}")
    except Exception as e:
        log_step(f"   [ERROR] Failed to create mel spectrogram: {e}")
        sys.exit(1)

    # Test model forward pass with mel
    log_step("STEP 9: Testing model forward pass with single sample")
    try:
        mel = mel.to(device)
        with torch.no_grad():
            audio_features = model.encoder(mel)
        log_step(f"   [OK] Encoder forward pass successful, output shape: {audio_features.shape}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            log_step(f"   [OK] After forward pass - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except Exception as e:
        log_step(f"   [ERROR] Forward pass failed: {e}")
        sys.exit(1)

    # Clean up
    log_step("STEP 10: Cleaning up")
    del mel, audio_features, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_step("   [OK] Cleanup complete")

    # Now test creating a small dataset
    log_step("STEP 11: Creating minimal dataset (5 samples)")

    class SimpleWhisperDataset(Dataset):
        def __init__(self, audio_files, max_samples=5):
            self.audio_files = audio_files[:max_samples]
            log_step(f"   INFO: Dataset initialized with {len(self.audio_files)} samples")

        def __len__(self):
            return len(self.audio_files)

        def __getitem__(self, idx):
            audio_path = self.audio_files[idx]

            # Load and process audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio)

            # Dummy text (we're just testing loading)
            text = "test transcription"

            return {
                'mel': mel,
                'text': text
            }

    try:
        test_dataset = SimpleWhisperDataset(audio_files, max_samples=5)
        log_step(f"   [OK] Dataset created with {len(test_dataset)} samples")
    except Exception as e:
        log_step(f"   [ERROR] Dataset creation failed: {e}")
        sys.exit(1)

    # Test DataLoader with num_workers=0 (safe for Windows)
    log_step("STEP 12: Creating DataLoader (num_workers=0, batch_size=2)")
    try:
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Safe for Windows
            pin_memory=True
        )
        log_step("   [OK] DataLoader created")
    except Exception as e:
        log_step(f"   [ERROR] DataLoader creation failed: {e}")
        sys.exit(1)

    # Test loading a batch
    log_step("STEP 13: Loading first batch from DataLoader")
    try:
        batch = next(iter(test_loader))
        mel_batch = batch['mel']
        log_step(f"   [OK] Batch loaded, mel shape: {mel_batch.shape}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            log_step(f"   [OK] After batch load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except Exception as e:
        log_step(f"   [ERROR] Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test forward pass with batch
    log_step("STEP 14: Testing forward pass with batch")
    try:
        mel_batch = mel_batch.to(device)
        with torch.no_grad():
            audio_features = model.encoder(mel_batch)
        log_step(f"   [OK] Batch forward pass successful, output shape: {audio_features.shape}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            log_step(f"   [OK] After batch forward - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except Exception as e:
        log_step(f"   [ERROR] Batch forward pass failed: {e}")
        sys.exit(1)

    # Test with BF16 (same as training)
    log_step("STEP 15: Testing forward pass with BF16 autocast")
    try:
        mel_batch = batch['mel'].to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                audio_features = model.encoder(mel_batch)
        log_step(f"   [OK] BF16 forward pass successful, output dtype: {audio_features.dtype}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            log_step(f"   [OK] After BF16 forward - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except Exception as e:
        log_step(f"   [ERROR] BF16 forward pass failed: {e}")
        sys.exit(1)

    # Final cleanup
    log_step("STEP 16: Final cleanup")
    del model, test_dataset, test_loader, batch, mel_batch, audio_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        log_step(f"   [OK] Final memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    log_step("SUCCESS: Full diagnostic completed!")
    log_step("All components working: model, dataset, dataloader, forward pass, BF16")
    log_step("Your system should be ready for the full training run.")

except Exception as e:
    log_step(f"ERROR: Diagnostic failed with exception:")
    log_step(f"   {type(e).__name__}: {e}")
    import traceback
    log_step("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

log_step("END: Full diagnostic complete")
