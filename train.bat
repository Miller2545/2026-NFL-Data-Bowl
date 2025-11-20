@echo off
title NFL Data Bowl - Train + Ensemble Evaluate

echo ============================================
echo   NFL Data Bowl — Training Configuration
echo ============================================
echo.

REM === Ask user for batch size ===
set /p BATCH_SIZE="Enter batch size (e.g., 4096): "

REM === Ask user for n_frames ===
set /p N_FRAMES="Enter number of frames to use (e.g., 10): "

REM === Ask user for epochs ===
set /p EPOCHS="Enter number of epochs (e.g., 50): "

echo.
echo ============================================
echo Training with:
echo   Batch Size : %BATCH_SIZE%
echo   n_frames   : %N_FRAMES%
echo   Epochs     : %EPOCHS%
echo ============================================
echo.
pause

echo --------------------------------------------
echo TRAINING LSTM MODEL
echo --------------------------------------------
python train_torch.py ^
  --arch lstm ^
  --input_dir data/train ^
  --output_dir data/train ^
  --weeks 1-14 ^
  --n_frames %N_FRAMES% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --val_split 0.1 ^
  --model_dir models/model_lstm

echo.
echo --------------------------------------------
echo TRAINING GRU MODEL
echo --------------------------------------------
python train_torch.py ^
  --arch gru ^
  --input_dir data/train ^
  --output_dir data/train ^
  --weeks 1-14 ^
  --n_frames %N_FRAMES% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --val_split 0.1 ^
  --model_dir models/model_gru

echo.
echo --------------------------------------------
echo TRAINING TRANSFORMER MODEL
echo --------------------------------------------
python train_torch.py ^
  --arch transformer ^
  --input_dir data/train ^
  --output_dir data/train ^
  --weeks 1-14 ^
  --n_frames %N_FRAMES% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --val_split 0.1 ^
  --model_dir models/model_transformer

echo.
echo ---------------------------------------------
echo NFL Big Data Bowl - Weighted Ensemble Evaluate
echo ---------------------------------------------

REM Evaluate on weeks 15-18
python evaluate_torch_weighted.py ^
  --input_dir data/train ^
  --output_dir data/train ^
  --weeks 15-18 ^
  --n_frames %N_FRAMES% ^
  --model_lstm models/model_lstm/best_model.pt ^
  --model_gru models/model_gru/best_model.pt ^
  --model_trans models/model_transformer/best_model.pt ^
  --out_dir ensemble_results

echo ---------------------------------------------
echo Evaluation complete!
echo Outputs created in: ensemble_results
echo   - rmse_by_model_and_horizon.csv
echo   - ensemble_weights_by_t.csv
echo   - rmse_curve.png
echo ---------------------------------------------
pause
