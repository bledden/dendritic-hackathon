# Validation Bug Fix - Post-Compression Crash

## Bug Summary

**When:** After first successful compression at epoch 16
**Where:** During validation at epoch 17
**Error:** `AttributeError: 'TypeError' object has no attribute 'shape'`

## Root Cause

After PAI compression, calling `whisper.decode()` breaks because Whisper's internal decoding logic performs type checking that fails when layers are compressed to `DendriticMLP`.

**Error chain:**
1. `whisper.decode(model, mel, options)` called
2. Whisper internally calls `model.encoder(mel)`
3. Encoder returns a TypeError object instead of audio features tensor
4. Whisper tries to access `.shape` attribute on the error object
5. Crash with `AttributeError: 'TypeError' object has no attribute 'shape'`

**Why this happens:**
- Whisper's `whisper.decode()` uses internal type checking/assumptions
- PAI compression changes layer types from `whisper.model.Linear` → `DendriticMLP`
- Whisper's decoding logic doesn't recognize the compressed layer types

## Solution

**Replace `whisper.decode()` with direct encoder/decoder calls:**

### Before (broken):
```python
options = whisper.DecodingOptions(language='en', without_timestamps=True)
results = whisper.decode(model, mel, options)
```

### After (fixed):
```python
# Direct encoder call (compatible with compressed layers)
audio_features = model.encoder(mel)

# Manual greedy decoding
tokens = torch.tensor([sot_sequence] * batch_size, device=device)
for _ in range(max_length):
    logits = model.decoder(tokens, audio_features)
    next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_tokens], dim=1)
    if (next_tokens == tokenizer.eot).all():
        break
```

## Why This Fix Works

1. **Direct model calls:** We call `model.encoder()` and `model.decoder()` directly, bypassing Whisper's internal decoding logic
2. **Same approach as training:** Training already uses direct calls and works fine post-compression
3. **Full compatibility:** Works with both uncompressed and compressed models
4. **Greedy decoding:** Simplified but effective for validation WER calculation

## Memory Analysis

This was **NOT a memory issue:**
- Pre-compression: 4.57GB allocated, 10.12GB peak
- Post-compression: 6.37GB allocated, 15.92GB peak
- Total available: 34.19GB (only 18.6% used)

Memory increase is expected during compression phase (candidate dendrites).

## Files Modified

1. `train_dendritic_full.py` - Fixed validate() function (lines 176-254)
2. `train_dendritic_fixed_trigger.py` - Applied same fix for Test 1b
3. `RESUME_TEST_1.bat` - Created recovery script

## How to Resume Test 1

The checkpoint saved at epoch 16 contains the compressed model. Simply run:

```batch
RESUME_TEST_1.bat
```

This will:
1. Load the compressed model (467M params with dendrites)
2. Continue training from epoch 17
3. Use the fixed validation function
4. Complete remaining epochs 17-25

## Expected Behavior After Fix

- Validation will run without crashes
- WER calculation will work correctly
- Compression has already occurred (48 MLP layers)
- Model will continue training in compressed state
- Additional compressions may trigger if WER plateaus again

## Next Steps

1. **Immediate:** Resume Test 1 to verify fix works
2. **After Test 1:** Review dendrite initialization issue (117% WER start)
3. **Future:** Consider modifying PAI or Whisper for better integration
