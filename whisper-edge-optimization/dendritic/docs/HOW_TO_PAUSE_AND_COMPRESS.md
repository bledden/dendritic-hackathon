# How to Use PAUSE and FORCE_COMPRESS - Step by Step

## Important: NO .txt Extension!

The files must be named exactly `PAUSE` or `FORCE_COMPRESS` with **NO extension**.

## Method 1: PowerShell (Easiest)

### To PAUSE training:

1. Open PowerShell
2. Copy/paste this command:
   ```powershell
   echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"
   ```
3. Press Enter
4. Done! Training will pause after current epoch finishes

### To FORCE COMPRESS:

1. Open PowerShell
2. Copy/paste this command:
   ```powershell
   echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"
   ```
3. Press Enter
4. Done! Compression will trigger at next epoch

## Method 2: File Explorer (Manual)

### Step-by-Step with Screenshots:

**Step 1:** Navigate to results folder
```
D:\ML_Results\dendritic_whisper\test_35_hybrid\
```

**Step 2:** Right-click in the folder → New → Text Document

**Step 3:** You'll see "New Text Document.txt"

**Step 4:** Rename it to exactly: `PAUSE` (or `FORCE_COMPRESS`)
- **DELETE the .txt extension!**
- Final name should be just: `PAUSE` or `FORCE_COMPRESS`

**Step 5:** Windows will warn: "If you change a file name extension, the file might become unusable. Are you sure you want to change it?"
- Click **YES**

**Step 6:** Verify the file has NO extension
- In File Explorer, enable "File name extensions" view
- View → Show → File name extensions (checkbox)
- File should show as `PAUSE` not `PAUSE.txt`

## Verification

### Check if file is correct:

**In PowerShell:**
```powershell
# Check if PAUSE file exists (should show file info)
Get-Item "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"

# Check if FORCE_COMPRESS file exists
Get-Item "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"
```

**If correct:** Shows file info
**If wrong:** Shows error "cannot find path"

## What Happens When You Create These Files?

### PAUSE File:

```
Epoch 15: 100%|████████████████| 3568/3568 [10:13<00:00, 5.82it/s]
Train loss: 0.0126

[Checking for PAUSE file...]

======================================================================
PAUSE REQUESTED
======================================================================
Training paused after epoch 15
Last checkpoint saved at epoch 15
To resume: Run the same command again
Removing pause file: D:\ML_Results\...\PAUSE

Cleaning up...
[OK] GPU resources released
[OK] Memory cleanup complete
```

### FORCE_COMPRESS File:

```
Epoch 15: 100%|████████████████| 3568/3568 [10:13<00:00, 5.82it/s]
Train loss: 0.0126

Validation WER: 18.40%

[Checking for FORCE_COMPRESS file...]

======================================================================
MANUAL COMPRESSION TRIGGERED
======================================================================
User requested compression at epoch 15
Forcing PAI to compress...

*** MODEL RESTRUCTURED! Dendrites added/incorporated. ***
   New parameters: 467,351,808
   ...
```

## Common Mistakes

### ❌ WRONG:
- File named: `PAUSE.txt` (has extension)
- File named: `pause` (lowercase)
- File named: `PAUSE.doc` (wrong extension)
- File in wrong folder

### ✅ CORRECT:
- File named: `PAUSE` (no extension)
- File named: `FORCE_COMPRESS` (no extension)
- In folder: `D:\ML_Results\dendritic_whisper\test_35_hybrid\`

## Quick Reference

| Action | PowerShell Command |
|--------|-------------------|
| Pause training | `echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"` |
| Force compression | `echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"` |
| Resume training | `.\RUN_TEST_35_HYBRID.bat` (same command as before) |

## File Contents Don't Matter

The files can be empty or have any content - the script only checks if they **exist**. The `echo $null >` command creates an empty file, but you could also create a file with text in it - doesn't matter!

## Files Are Auto-Deleted

Once the script detects the file, it **automatically deletes it**. So:
- Create `PAUSE` → Script pauses and deletes the file
- Create `FORCE_COMPRESS` → Script compresses and deletes the file
- You won't see leftover files cluttering the directory

## Still Confused?

**Easiest method - just use PowerShell:**

1. Open PowerShell
2. Copy this exact command:
   ```powershell
   echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"
   ```
3. Press Enter
4. Done!

That's it! No need to worry about extensions or file explorer tricks.
