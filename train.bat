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
echo ============================================
echo Training Finished! Now Evaluating Ensemble
echo ============================================
echo.

python ensemble_evaluate_torch.py ^
  --lstm_model_path models/model_lstm/best_model.pt ^
  --gru_model_path models/model_gru/best_model.pt ^
  --transformer_model_path models/model_transformer/best_model.pt ^
  --input_dir data/train ^
  --output_dir data/train ^
  --weeks 15-18 ^
  --n_frames %N_FRAMES% ^
  --batch_size %BATCH_SIZE% ^
  --save_csv

echo.
echo ============================================
echo      Ensemble Evaluation Complete!
echo ============================================
echo Results written to:
echo   - models/model_lstm/ensemble_competition_rmse.txt
echo   - models/model_lstm/ensemble_rmse_by_t.csv
echo   - artifacts/ensemble_preds_torch.csv
echo.
pause
