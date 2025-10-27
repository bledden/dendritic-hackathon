#!/usr/bin/env python3
"""
Diagnostic script to understand PAI checkpoint save/load issue.

This script helps us understand:
1. What gets saved in a PAI checkpoint
2. What PAI modules exist before/after save
3. Why load_net_from_dict() can't find PAI modules
"""

import torch
import whisper
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PerforatedAI"))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

def inspect_model_modules(model, label):
    """Inspect what PAI modules exist in a model."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    # Check for PAI modules
    pai_modules = UPA.get_pai_modules(model, 0)
    print(f"Number of PAI modules found: {len(pai_modules)}")

    if pai_modules:
        print("\nPAI modules:")
        for i, module in enumerate(pai_modules[:5]):  # Show first 5
            print(f"  {i+1}. {module.name} (type: {type(module).__name__})")
        if len(pai_modules) > 5:
            print(f"  ... and {len(pai_modules) - 5} more")
    else:
        print("  [WARNING] NO PAI MODULES FOUND!")

    # Check state dict keys
    state_dict = model.state_dict()
    pai_keys = [k for k in state_dict.keys() if 'dendrite' in k.lower() or 'pai' in k.lower()]
    print(f"\nPAI-related keys in state_dict: {len(pai_keys)}")
    if pai_keys:
        for key in pai_keys[:5]:
            print(f"  - {key}")
        if len(pai_keys) > 5:
            print(f"  ... and {len(pai_keys) - 5} more")

    # Check for tracker_string
    if hasattr(model, 'tracker_string'):
        print(f"\n[OK] Model has 'tracker_string' attribute")
    else:
        print(f"\n[WARNING] Model does NOT have 'tracker_string' attribute")

def main():
    print("="*70)
    print("PAI CHECKPOINT DIAGNOSTIC TOOL")
    print("="*70)

    # Step 1: Load fresh Whisper model
    print("\n[1/6] Loading fresh Whisper Small model...")
    model = whisper.load_model("small", device="cpu")
    inspect_model_modules(model, "BEFORE PAI INITIALIZATION")

    # Step 2: Initialize PAI
    print("\n[2/6] Initializing PAI...")
    from whisper.model import ResidualAttentionBlock, AudioEncoder, TextDecoder

    GPA.pc.append_modules_to_convert([ResidualAttentionBlock])
    GPA.pc.append_modules_to_track([AudioEncoder, TextDecoder])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_max_dendrites(5)

    model = UPA.initialize_pai(
        model,
        save_name="diagnostic_test",
        maximizing_score=True,
        making_graphs=False
    )

    inspect_model_modules(model, "AFTER PAI INITIALIZATION")

    # Step 3: Save checkpoint
    print("\n[3/6] Saving checkpoint with UPA.save_system()...")
    save_dir = Path("../results/diagnostic_checkpoint")
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        UPA.save_system(model, "diagnostic_test", "test_checkpoint")
        print("[OK] Checkpoint saved successfully")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
        return

    # Step 4: Load fresh model (simulates what PAI does when adding dendrites)
    print("\n[4/6] Loading FRESH Whisper model (no PAI init)...")
    fresh_model = whisper.load_model("small", device="cpu")
    inspect_model_modules(fresh_model, "FRESH MODEL (Before Load)")

    # Step 5: Try to load checkpoint into fresh model
    print("\n[5/6] Attempting to load checkpoint into fresh model...")
    try:
        loaded_model = UPA.load_system(
            fresh_model,
            "diagnostic_test",
            "test_checkpoint",
            load_from_manual_save=True
        )
        print("[OK] Load succeeded!")
        inspect_model_modules(loaded_model, "AFTER LOADING CHECKPOINT")
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Try alternative - initialize PAI first, THEN load
    print("\n[6/6] Alternative approach: Initialize PAI on fresh model FIRST...")
    fresh_model2 = whisper.load_model("small", device="cpu")

    # Initialize PAI
    GPA.pc.append_modules_to_convert([ResidualAttentionBlock])
    GPA.pc.append_modules_to_track([AudioEncoder, TextDecoder])
    GPA.pc.set_unwrapped_modules_confirmed(True)

    fresh_model2 = UPA.initialize_pai(
        fresh_model2,
        save_name="diagnostic_test2",
        maximizing_score=True,
        making_graphs=False
    )

    inspect_model_modules(fresh_model2, "FRESH MODEL WITH PAI INIT (Before Load)")

    try:
        loaded_model2 = UPA.load_system(
            fresh_model2,
            "diagnostic_test",
            "test_checkpoint",
            load_from_manual_save=True
        )
        print("[OK] Load succeeded with pre-initialized PAI!")
        inspect_model_modules(loaded_model2, "AFTER LOADING INTO PAI-INITIALIZED MODEL")
    except Exception as e:
        print(f"[ERROR] Load failed even with pre-init: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
