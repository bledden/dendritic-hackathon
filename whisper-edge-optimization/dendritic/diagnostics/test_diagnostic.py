"""
Diagnostic script to identify where crashes occur during startup.
Logs each step to help pinpoint the failure point.
"""

import sys
import time

def log_step(step_name):
    """Log a step with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {step_name}", flush=True)
    sys.stdout.flush()

try:
    log_step("START: Diagnostic script beginning")

    # Step 1: Basic imports
    log_step("STEP 1: Importing os and pathlib")
    import os
    from pathlib import Path
    log_step("   [OK] os and pathlib imported")

    # Step 2: PyTorch import
    log_step("STEP 2: Importing PyTorch")
    import torch
    log_step(f"   [OK] PyTorch {torch.__version__} imported")

    # Step 3: CUDA availability
    log_step("STEP 3: Checking CUDA availability")
    cuda_available = torch.cuda.is_available()
    log_step(f"   [OK] CUDA available: {cuda_available}")

    if cuda_available:
        log_step("STEP 4: Getting CUDA device properties")
        device_name = torch.cuda.get_device_name(0)
        log_step(f"   [OK] GPU: {device_name}")

        log_step("STEP 5: Checking GPU memory")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log_step(f"   [OK] Total GPU memory: {total_memory:.2f} GB")

        log_step("STEP 6: Creating small test tensor on GPU")
        test_tensor = torch.randn(100, 100).cuda()
        log_step(f"   [OK] Test tensor created: {test_tensor.shape}")

        log_step("STEP 7: Checking allocated GPU memory")
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        log_step(f"   [OK] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

        log_step("STEP 8: Clearing GPU memory")
        del test_tensor
        torch.cuda.empty_cache()
        log_step("   [OK] GPU memory cleared")

    # Step 9: Import numpy and other ML libraries
    log_step("STEP 9: Importing numpy and tqdm")
    import numpy as np
    from tqdm import tqdm
    log_step("   [OK] numpy and tqdm imported")

    # Step 10: Import Whisper
    log_step("STEP 10: Importing Whisper (this may take a moment)")
    import whisper
    log_step(f"   [OK] Whisper imported")

    # Step 11: Load Whisper model
    log_step("STEP 11: Loading Whisper-Small model (this will take 1-2 minutes)")
    log_step("   INFO: This downloads ~500MB if not cached")
    model = whisper.load_model("small", device="cpu")  # Load to CPU first
    log_step(f"   [OK] Whisper model loaded to CPU")

    log_step("STEP 12: Getting model parameter count")
    total_params = sum(p.numel() for p in model.parameters())
    log_step(f"   [OK] Model has {total_params:,} parameters")

    if cuda_available:
        log_step("STEP 13: Moving model to GPU")
        model = model.cuda()
        log_step("   [OK] Model moved to GPU")

        log_step("STEP 14: Checking GPU memory after model load")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        log_step(f"   [OK] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Step 15: Import PerforatedAI
    log_step("STEP 15: Importing PerforatedAI")
    try:
        import perforatedai as pai
        log_step(f"   [OK] PerforatedAI {pai.__version__} imported")
    except Exception as e:
        log_step(f"   [WARNING] PerforatedAI import failed: {e}")
        log_step("   [INFO] This is expected if PAI not installed")

    # Step 16: Test BF16
    if cuda_available:
        log_step("STEP 16: Testing BF16 support")
        if torch.cuda.is_bf16_supported():
            test_bf16 = torch.randn(100, 100, dtype=torch.bfloat16).cuda()
            log_step(f"   [OK] BF16 supported and working")
            del test_bf16
        else:
            log_step(f"   [WARNING] BF16 not supported on this GPU")

    # Step 17: Clean up
    log_step("STEP 17: Cleaning up model")
    del model
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log_step("   [OK] Model cleaned up")

    # Step 18: Final memory check
    if cuda_available:
        log_step("STEP 18: Final GPU memory check")
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        log_step(f"   [OK] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    log_step("SUCCESS: All diagnostic steps completed!")
    log_step("Your system appears ready to run training.")

except Exception as e:
    log_step(f"ERROR: Diagnostic failed with exception:")
    log_step(f"   {type(e).__name__}: {e}")
    import traceback
    log_step("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

log_step("END: Diagnostic script complete")
