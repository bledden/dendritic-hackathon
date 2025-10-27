@echo off
REM Test with all optimizations + larger batch size
REM - Memory clearing after restructuring
REM - DataLoader workers (4) + pin_memory
REM - Batch size 16 (instead of 8) for faster throughput
REM - Reduced to 15 epochs for faster iteration

python train_dendritic_full.py ^
  --device cuda ^
  --save-name test_optimized_bs16 ^
  --val-max-samples 100 ^
  --max-epochs 15 ^
  --batch-size 16 ^
  --do-training ^
  --use-amp ^
  --amp-dtype bfloat16 ^
  --n-epochs-to-switch 3 ^
  --max-dendrites 3 ^
  --num-workers 4 ^
  --data-dir "D:\ML_Datasets\LibriSpeech" ^
  --results-dir "D:\ML_Results\dendritic_whisper"

pause
