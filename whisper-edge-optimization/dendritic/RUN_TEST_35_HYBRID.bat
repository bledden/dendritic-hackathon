@echo off
REM 35-Epoch Test with Hybrid Compression Mode
REM
REM Features:
REM - Hybrid compression: Force at epochs 10, 20, 30 OR natural plateau (n=3)
REM - PAUSE support: Create "PAUSE" file in results dir to pause after current epoch
REM - FORCE_COMPRESS support: Create "FORCE_COMPRESS" file to manually trigger compression
REM - Safe termination: Proper GPU cleanup on exit
REM - Memory clearing: Prevents fragmentation after restructuring
REM - pin_memory: Faster CPU->GPU transfer
REM
REM Expected timeline (~7.5 hours total):
REM - Epochs 1-10: Pre-compression training (~100 min)
REM - Epoch 10: Add dendrites (240M -> 467M)
REM - Epochs 11-20: Learn dendrite importance (~165 min)
REM - Epoch 20: PRUNE (467M -> ~150M) - THE BIG MOMENT!
REM - Epochs 21-35: Fine-tune compressed model (~165 min)
REM
REM To pause training:
REM   echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\PAUSE"
REM
REM To force compression manually:
REM   echo $null > "D:\ML_Results\dendritic_whisper\test_35_hybrid\FORCE_COMPRESS"

echo ======================================================================
echo 35-EPOCH HYBRID COMPRESSION TEST
echo ======================================================================
echo.
echo This test will:
echo - Run for 35 epochs (~7.5 hours)
echo - Force compression at epochs 10, 20, 30
echo - Or trigger naturally if plateau detected (3 epochs without improvement)
echo.
echo You can pause anytime by creating PAUSE file in results directory
echo You can force compression by creating FORCE_COMPRESS file
echo.
echo Press Ctrl+C now to cancel, or
pause

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_35_hybrid ^
  --val-max-samples 100 ^
  --max-epochs 35 ^
  --batch-size 8 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --compression-mode hybrid ^
  --force-trigger-interval 10 ^
  --n-epochs-to-switch 3 ^
  --max-dendrites 3 ^
  --num-workers 0 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo TRAINING COMPLETED SUCCESSFULLY!
    echo ======================================================================
    echo Check results in: D:\ML_Results\dendritic_whisper\test_35_hybrid
    echo.
) else (
    echo.
    echo ======================================================================
    echo TRAINING FAILED
    echo ======================================================================
    echo Exit code: %ERRORLEVEL%
    echo Check logs above for errors
    echo.
)

REM Auto-exit after 5 seconds
timeout /t 5 /nobreak > nul
exit /b %ERRORLEVEL%
