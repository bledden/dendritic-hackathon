@echo off
REM Test 1 with validation fix - Fresh start
REM This version has the whisper.decode() bug fixed

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_1_fixed_validation ^
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

pause
