"""
Diagnostic Script: Investigate why get_pai_modules() returns empty list

This script creates a Whisper model, initializes PAI, and inspects the module
structure to understand why get_pai_modules() can't find PAI modules.
"""

import sys
import torch
import whisper
from pathlib import Path

# Add PerforatedAI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PerforatedAI"))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

print("=" * 70)
print("PAI MODULE STRUCTURE DIAGNOSTIC")
print("=" * 70)

# Load Whisper Small model
print("\n[1/6] Loading Whisper Small model...")
model = whisper.load_model("small")
print(f"      Model loaded: {model.__class__.__name__}")
print(f"      Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize PAI (matching train_dendritic_full.py with FIX applied)
print("\n[2/6] Initializing PAI...")

# CRITICAL FIX: Whisper uses custom whisper.model.Linear (subclass of nn.Linear)
# PAI uses type() not isinstance(), so we must explicitly add Whisper's Linear class!
from whisper.model import Linear as WhisperLinear
GPA.pc.append_modules_to_convert([WhisperLinear])

GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_max_dendrites(5)
GPA.pc.set_improvement_threshold(0.0001)

model = UPA.initialize_pai(model)
print("      PAI initialized")

# Test get_pai_modules()
print("\n[3/6] Calling get_pai_modules()...")
pai_modules = UPA.get_pai_modules(model, 0)
print(f"      Found {len(pai_modules)} PAI modules")

if len(pai_modules) == 0:
    print("      [WARNING] NO PAI MODULES FOUND!")
else:
    print("      [OK] PAI modules found:")
    for i, module in enumerate(pai_modules):
        print(f"         {i+1}. {module.name}")

# Inspect model structure
print("\n[4/6] Inspecting model structure...")
print("\n      Top-level attributes:")
for attr_name in dir(model):
    if not attr_name.startswith('_'):
        attr = getattr(model, attr_name)
        if isinstance(attr, torch.nn.Module):
            print(f"         - {attr_name}: {attr.__class__.__name__}")

# Check for .main_module wrapping
print("\n      Checking for .main_module wrapping...")
if hasattr(model, 'encoder'):
    print(f"         model.encoder: {model.encoder.__class__.__name__}")
    if hasattr(model.encoder, 'main_module'):
        print(f"         model.encoder.main_module: {model.encoder.main_module.__class__.__name__}")
        print("         [FOUND] Encoder is wrapped with main_module!")
    else:
        print("         [NOT FOUND] Encoder has no main_module attribute")

if hasattr(model, 'decoder'):
    print(f"         model.decoder: {model.decoder.__class__.__name__}")
    if hasattr(model.decoder, 'main_module'):
        print(f"         model.decoder.main_module: {model.decoder.main_module.__class__.__name__}")
        print("         [FOUND] Decoder is wrapped with main_module!")
    else:
        print("         [NOT FOUND] Decoder has no main_module attribute")

# Recursively search for PAINeuronModule
print("\n[5/6] Manual recursive search for PAINeuronModule...")

def find_pai_modules_manual(module, prefix='', depth=0, max_depth=10):
    """Manually search for PAINeuronModule instances"""
    found = []
    if depth > max_depth:
        return found

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        class_name = child.__class__.__name__

        # Print ALL module types we encounter
        if 'PAI' in class_name or 'Tracked' in class_name:
            print(f"      {'  ' * depth}[{class_name}] {full_name}")

        if 'PAINeuronModule' in class_name:
            found.append((full_name, child))

        found.extend(find_pai_modules_manual(child, full_name, depth + 1, max_depth))
    return found

pai_modules_manual = find_pai_modules_manual(model)
print(f"\n      Manual search found {len(pai_modules_manual)} PAINeuronModule instances")

# Compare with get_pai_modules()
print("\n[6/6] Comparison and diagnosis...")
print(f"      get_pai_modules() found: {len(pai_modules)}")
print(f"      Manual search found:     {len(pai_modules_manual)}")

if len(pai_modules) == 0 and len(pai_modules_manual) > 0:
    print("\n      [DIAGNOSIS] get_pai_modules() FAILS to find modules that exist!")
    print("      This confirms the module structure prevents proper detection.")
    print("\n      Example of found module structure:")
    if pai_modules_manual:
        example_name, example_module = pai_modules_manual[0]
        print(f"         Name: {example_name}")
        print(f"         Type: {example_module.__class__.__name__}")
        if hasattr(example_module, 'name'):
            print(f"         Module.name: {example_module.name}")
elif len(pai_modules) == 0 and len(pai_modules_manual) == 0:
    print("\n      [DIAGNOSIS] NO PAI modules created during initialization!")
    print("      PAI may not be converting modules as expected.")
else:
    print("\n      [OK] get_pai_modules() correctly found all modules")

# Check if this is why checkpoint loading fails
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if len(pai_modules) == 0:
    print("""
This explains why dendrite loading fails during restructuring:

1. initialize_pai() is called and creates PAI modules
2. get_pai_modules() cannot find the created modules
3. During restructuring, load_net_from_dict() calls get_pai_modules()
4. Returns empty list -> dendrite loading loop skips
5. Model continues with no dendrites -> 0% compression

ROOT CAUSE: Module structure incompatibility between PAI's wrapping and
get_pai_modules() search algorithm.
""")
else:
    print("""
get_pai_modules() successfully found PAI modules. The dendritic compression
failure may have a different root cause. Further investigation needed.
""")

print("=" * 70)
