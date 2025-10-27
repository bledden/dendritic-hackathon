@echo off
REM Resume Test 1 from checkpoint after validation fix
REM This will load the compressed model from epoch 16 and continue

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

pause
