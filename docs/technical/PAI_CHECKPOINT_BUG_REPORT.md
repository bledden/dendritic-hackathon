# PerforatedAI Checkpoint Bug Report

**Date:** October 25, 2025
**Discoverer:** Blake Ledden
**Context:** Dendritic Hackathon - Whisper Small Compression
**Severity:** HIGH - Blocks dendrite addition functionality

---

## Executive Summary

PerforatedAI's `save_system()` and `load_system()` have a critical bug that prevents proper checkpoint loading when PAI is used with complex models like OpenAI Whisper. The issue causes **silent failure** - checkpoints load without errors but PAI structure is not restored.

---

## Bug Description

### Observed Behavior

When PAI tries to load a "best model" checkpoint to add dendrites:

1. Training runs normally for 18+ epochs
2. PAI detects validation plateau
3. Attempts to load best checkpoint: `load_system(model, folder, "best_model")`
4. **Hits breakpoint** at `utils_perforatedai.py:940`
5. Shows warning: "PAI load_net and load_system uses a state_dict so it must be called with a net after initialize_pai has been called"
6. If breakpoint bypassed, load appears to succeed
7. **BUT:** No PAI modules exist after loading (silent failure)

### Root Cause

**State Dict Key Mismatch:**

When PAI initializes a model, it wraps modules with `.main_module`:
```python
# Original Whisper structure:
encoder.blocks.0.attn.query.weight

# After PAI initialization:
encoder.main_module.blocks.0.attn.query.weight
```

When `save_system()` saves the checkpoint:
- ✅ Saves with `.main_module` keys
- ✅ Saves tracker_string correctly

When `load_system()` loads the checkpoint:
- ❌ Expects model to already have PAI structure initialized
- ❌ Tries to load `.main_module` keys into non-wrapped model
- ❌ Keys don't match → loads nothing
- ❌ No error raised (silent failure)

---

## Reproduction Steps

### Minimal Reproduction

```python
import whisper
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from whisper.model import ResidualAttentionBlock, AudioEncoder, TextDecoder

# Step 1: Load and initialize PAI
model = whisper.load_model("small", device="cpu")
GPA.pc.append_modules_to_convert([ResidualAttentionBlock])
GPA.pc.append_modules_to_track([AudioEncoder, TextDecoder])
GPA.pc.set_unwrapped_modules_confirmed(True)

model = UPA.initialize_pai(model, save_name="test", maximizing_score=True)

# Step 2: Save checkpoint
UPA.save_system(model, "test", "checkpoint")

# Step 3: Try to load into fresh model (THIS FAILS SILENTLY)
fresh_model = whisper.load_model("small", device="cpu")
loaded_model = UPA.load_system(fresh_model, "test", "checkpoint", load_from_manual_save=True)

# Step 4: Check if PAI modules exist
pai_modules = UPA.get_pai_modules(loaded_model, 0)
print(f"PAI modules: {len(pai_modules)}")  # Returns 0 (WRONG!)
```

###Expected: 200+ PAI modules
### Actual: 0 PAI modules (silent failure)

---

## Impact

**Blocks Core Functionality:**
- ❌ Dendrite addition requires loading best checkpoint
- ❌ Model restructuring cannot proceed
- ❌ Compression is impossible
- ❌ Entire dendritic workflow broken for complex models

**Affected Models:**
- ✅ Whisper (confirmed)
- Likely: Any model with nested module structure
- Likely: Transformers, Vision Transformers, etc.

---

## Current Workarounds

### Workaround 1: Comment Out Breakpoints (Temporary)
```python
# In utils_perforatedai.py line 940:
# pdb.set_trace()  # COMMENTED OUT
# sys.exit(-1)  # COMMENTED OUT
```

**Status:** Allows training to continue but dendrites still don't work properly

### Workaround 2: Pre-initialize PAI Before Loading (Untested)
```python
fresh_model = whisper.load_model("small", device="cpu")
# Initialize PAI on fresh model BEFORE loading
fresh_model = UPA.initialize_pai(fresh_model, save_name="temp", maximizing_score=True)
# Now load checkpoint
loaded_model = UPA.load_system(fresh_model, "test", "checkpoint")
```

**Status:** Theoretically should work, but may cause other issues

---

## Proposed Fix

### Option A: Smart Key Remapping in load_net_from_dict()

Modify `load_net_from_dict()` to detect and handle key mismatches:

```python
def load_net_from_dict(net, state_dict):
    pai_modules = get_pai_modules(net, 0)

    # Check if state_dict has .main_module but net doesn't (or vice versa)
    if pai_modules == []:
        # Check if state_dict has PAI structure
        has_main_module_in_dict = any('.main_module.' in k for k in state_dict.keys())

        if has_main_module_in_dict:
            print("WARNING: State dict has PAI structure but model doesn't")
            print("         Remapping keys to match non-PAI model structure...")

            # Remap keys: remove .main_module. from state_dict keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('.main_module.', '.')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

    # Continue with normal loading
    ...
```

### Option B: Auto-initialize PAI Before Loading

Modify `load_system()` to automatically initialize PAI if needed:

```python
def load_system(net, folder, name, ...):
    # Check if net has PAI initialized
    pai_modules = get_pai_modules(net, 0)

    if pai_modules == []:
        # Load tracker_string from checkpoint to get PAI config
        state_dict = torch.load(...)
        if 'tracker_string' in state_dict:
            print("Auto-initializing PAI before loading checkpoint...")
            # Initialize PAI on net using saved config
            net = initialize_pai(net, ...)  # Need to extract config from tracker

    # Continue with normal loading
    net = load_net(net, folder, name)
    ...
```

### Option C: Save Both Wrapped and Unwrapped State Dicts

Modify `save_system()` to save two versions:

```python
def save_system(net, folder, name):
    # Save PAI version (with .main_module)
    save_net(net, folder, name + "_pai")

    # Save unwrapped version (without .main_module)
    unwrapped_state = {}
    for key, value in net.state_dict().items():
        unwrapped_key = key.replace('.main_module.', '.')
        unwrapped_state[unwrapped_key] = value
    torch.save(unwrapped_state, folder + "/" + name + "_unwrapped.pt")
```

---

## Recommended Solution

**Hybrid Approach:**

1. **Immediate:** Option A (key remapping) - Fixes the issue without breaking existing code
2. **Long-term:** Option B (auto-initialization) - Makes load_system() truly self-contained
3. **Documentation:** Update docs to explain checkpoint compatibility requirements

---

## Testing Strategy

### Test Cases

1. **Load PAI checkpoint into PAI-initialized model** (should work)
2. **Load PAI checkpoint into fresh model** (currently fails, should work after fix)
3. **Load non-PAI checkpoint into PAI model** (edge case)
4. **Load checkpoint saved before dendrites into model during dendrite addition** (your use case)

### Validation

```python
# After fix, this should work:
model1 = initialize_and_train_model()
save_system(model1, "test", "checkpoint")

model2 = load_fresh_model()  # No PAI init
loaded = load_system(model2, "test", "checkpoint")

assert len(get_pai_modules(loaded, 0)) > 0  # Should pass!
```

---

## Files Affected

- `PerforatedAI/perforatedai/utils_perforatedai.py`
  - Line 914: `load_net_from_dict()` - where fix should go
  - Line 753: `load_system()` - may need updates
  - Line 711: `save_system()` - may need to save additional info

---

## Additional Notes

### Debug Breakpoints Found

While investigating, found 7 `pdb.set_trace()` calls in production code:
- Lines: 387, 463, 560, 658, 940, 995, 1026
- **These should be removed or wrapped in debug flags**
- They break automation and cause user confusion

### Related Issues

This may affect:
- Model restarts after crashes
- Distributed training checkpointing
- Transfer learning scenarios
- Any use case requiring checkpoint portability

---

## Contact

**Reporter:** Blake Ledden
**Project:** Dendritic Compression for Whisper Small
**Repo:** Private hackathon project
**Willing to contribute fix:** Yes

---

## Appendix: Diagnostic Output

See `diagnose_pai_checkpoint.py` for full reproduction script.

**Key finding:**
```
AFTER PAI INITIALIZATION
Number of PAI modules found: 0  ← WRONG! Should be 200+
```

This indicates PAI's module wrapping isn't working as expected on Whisper, OR the diagnostic method is wrong. Need to investigate `get_pai_modules()` as well.
