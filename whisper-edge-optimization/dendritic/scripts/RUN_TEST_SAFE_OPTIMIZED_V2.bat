@echo off
REM Safe optimized version for Windows - V2 with safe termination
REM - Memory clearing after restructuring (SAFE, big impact)
REM - pin_memory=True (usually safe, modest impact)
REM - num_workers=0 (avoid Windows multiprocessing issues)
REM - Proper cleanup on exit
REM - 15 epochs for faster iteration

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_safe_optimized_v2 ^
  --val-max-samples 100 ^
  --max-epochs 15 ^
  --batch-size 8 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --n-epochs-to-switch 3 ^
  --max-dendrites 3 ^
  --num-workers 0 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training completed successfully!
    echo Exit code: %ERRORLEVEL%
) else (
    echo.
    echo Training failed with error code: %ERRORLEVEL%
)

REM Give brief pause to see final message, then auto-exit
timeout /t 3 /nobreak > nul
echo Exiting...
exit /b %ERRORLEVEL%
