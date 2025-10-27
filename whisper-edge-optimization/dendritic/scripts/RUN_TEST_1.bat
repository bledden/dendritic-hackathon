@echo off
REM ============================================================================
REM Test 1: Conservative Full-Model Compression
REM Target: Compress MLP layers only, use BF16, n=3 trigger, 25 epochs
REM Expected: 5-10%% parameter reduction, ~9.5 hours runtime
REM ============================================================================

echo.
echo ========================================================================
echo Test 1: Conservative MLP-Only Compression with BF16
echo ========================================================================
echo.
echo Configuration:
echo   - Layers: MLP only (48 layers total)
echo   - Precision: BF16 mixed precision
echo   - Compression trigger: 3 epochs without improvement
echo   - Max dendrite cycles: 3
echo   - Epochs: 25
echo   - Batch size: 8
echo   - GPU: RTX 5090 (32GB)
echo.
echo Expected memory usage: 5.5-6 GB (plenty of headroom)
echo Expected runtime: ~9.5 hours
echo.
echo Press Ctrl+C to cancel, or
pause

cd /d "%~dp0"

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_1_conservative_mlp_only ^
  --val-max-samples 100 ^
  --max-epochs 25 ^
  --batch-size 8 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --n-epochs-to-switch 3 ^
  --max-dendrites 3 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

echo.
echo ========================================================================
echo Test completed! Check results in:
echo   D:\ML_Results\dendritic_whisper\test_1_conservative_mlp_only\
echo ========================================================================
echo.
pause
