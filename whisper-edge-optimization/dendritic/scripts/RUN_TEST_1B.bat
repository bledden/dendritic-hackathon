@echo off
REM ============================================================================
REM Test 1b: Fixed-Epoch Compression (Guaranteed Triggers)
REM Target: Same as Test 1 but with GUARANTEED compression at epochs 8, 16
REM Expected: 2-3 compressions, 10-20%% parameter reduction, ~3.5 hours runtime
REM ============================================================================

echo.
echo ========================================================================
echo Test 1b: Fixed-Epoch MLP Compression with BF16
echo ========================================================================
echo.
echo Configuration:
echo   - Layers: MLP only (48 layers total)
echo   - Precision: BF16 mixed precision
echo   - Compression trigger: FIXED at epochs 8, 16, 24
echo   - Max dendrite cycles: 3
echo   - Epochs: 25
echo   - Batch size: 8
echo   - GPU: RTX 5090 (32GB)
echo.
echo Expected compressions: 3 (epochs 8, 16, 24)
echo Expected memory usage: 5.5-10 GB (safe)
echo Expected runtime: ~3.5 hours
echo.
echo Press Ctrl+C to cancel, or
pause

cd /d "%~dp0"

python train_dendritic_fixed_trigger.py ^
  --device cuda ^
  --save-name test_1b_fixed_trigger_mlp_only ^
  --val-max-samples 100 ^
  --max-epochs 25 ^
  --batch-size 8 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --fixed-switch-num 8 ^
  --max-dendrites 3 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

echo.
echo ========================================================================
echo Test 1b completed! Check results in:
echo   D:\ML_Results\dendritic_whisper\test_1b_fixed_trigger_mlp_only\
echo ========================================================================
echo.
pause
