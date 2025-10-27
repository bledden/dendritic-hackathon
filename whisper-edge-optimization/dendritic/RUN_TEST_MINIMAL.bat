@echo off
echo ======================================================================
echo MINIMAL TEST - Quick startup test with actual training script
echo ======================================================================
echo.
echo This runs the REAL training script but with minimal settings:
echo - Only 1 epoch
echo - Only 10 validation samples
echo - Only 20 training samples
echo - Batch size 2
echo.
echo This should complete in under 5 minutes if working properly.
echo.
echo ======================================================================
echo.

cd /d "%~dp0"

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_minimal ^
  --val-max-samples 10 ^
  --train-max-samples 20 ^
  --max-epochs 1 ^
  --batch-size 2 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --compression-mode history ^
  --n-epochs-to-switch 3 ^
  --max-dendrites 3 ^
  --num-workers 0 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

echo.
if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo SUCCESS! Minimal test completed.
    echo ======================================================================
    echo.
    echo The full 35-epoch test should work now.
) else (
    echo ======================================================================
    echo FAILED with exit code: %ERRORLEVEL%
    echo ======================================================================
    echo.
    echo Check error messages above to identify the issue.
)

REM Auto-exit after 10 seconds
timeout /t 10 /nobreak > nul
exit /b %ERRORLEVEL%
