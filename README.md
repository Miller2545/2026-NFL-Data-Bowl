# 2026 NFL Data Bowl Prediction Competition

## 📁 Project Structure & File-by-File Overview

This repository contains a full PyTorch-based solution for the NFL Big Data Bowl competition.  
It includes preprocessing pipelines, three complementary deep learning architectures (LSTM, GRU, Transformer), an ensemble system, and evaluation tooling.

Below is a detailed explanation of what each file does and how they interact.

---

### 🔧 Core Files

#### preprocess.py

Purpose:  
Handles all feature engineering, normalization, frame padding, categorical encoding, and physics-based feature generation.

Key responsibilities:

- Loads raw player tracking frames per week.
- Normalizes coordinates so play direction is consistent.
- Extracts release-frame context for each player.
- Builds padded input sequences of length `n_frames`.
- Computes physics features:
  - velocity components (vx, vy)
  - acceleration components (ax, ay)
  - kinetic energy
  - momentum
  - angular change, turning rate
- Encodes categorical variables (player_position, player_role, player_side).
- Outputs:
  - X_array → [samples, T, F]
  - t_input → [samples, 1]
  - meta_rows containing game_id/play_id/nfl_id/frame info

This file ensures that training and test data are processed identically.

---

### 🧠 Model Definitions

#### model_torch.py

Purpose:  
Defines three neural network architectures used for training and ensembling.

##### 1. LSTMFrameConditionedModel

- Two-layer LSTM sequence encoder.
- Time-conditioning branch (dense network with scalar time).
- MLP head predicts (x, y).
- Strong at short-term temporal patterns.

##### 2. GRUFrameConditionedModel

- Similar to LSTM model but with GRU layers.
- Lower memory footprint.
- Different inductive biases → improves ensemble diversity.

##### 3. TransformerFrameConditionedModel

- Conv1D pre-processing layer for local motion extraction.
- Learnable CLS token for sequence summarization.
- Transformer encoder stack (multi-head attention).
- Time-conditioning branch.
- Strongest long-range spatiotemporal learner.

Why multiple models?  
Ensembles greatly outperform any single architecture.

---

### 📘 Training

#### train_torch.py

Purpose:  
Full training pipeline for a single model (LSTM, GRU, Transformer).

Functionality includes:

- Loads weekly tracking data.
- Performs preprocessing using preprocess.py.
- Builds PyTorch datasets and dataloaders.
- Uses AMP (automatic mixed precision) for faster GPU training.
- Optimizer: AdamW
- Schedulers:
  - Warmup (LinearLR)
  - Reduce-on-plateau (ReduceLROnPlateau)
- Logs training + validation loss each epoch.
- Saves the best model weights to best_model.pt.
- Exports loss curves as PNG plots.

Example usage:
python train_torch.py --model lstm --weeks 1-14 --epochs 150 --batch_size 4096

---

### 📊 Evaluation & Ensemble

#### evaluate_torch_weighted.py

Purpose:  
Evaluates LSTM, GRU, and Transformer and learns optimal ensemble weights per prediction horizon.

Features:

- Predicts validation set with all models.
- Computes RMSE for each model for each horizon.
- Learns weights through:
  - L1-constrained least squares
  - Softmax weighting
  - Ridge regression
- Outputs:
  - ensemble_weights_by_t.csv
  - validation_curve.png

Example output table:
t_int, rmse_lstm, rmse_gru, rmse_trans, w_lstm, w_gru, w_trans, rmse_ensemble

---

#### inference_ensemble.py

Purpose:  
Runs inference using all three models, weighted by per-horizon weights.

Pipeline:

- Loads LSTM, GRU, Transformer
- Loads ensemble_weights_by_t.csv
- For each row with horizon `t_int`:
  y = w_lstm*y_lstm + w_gru*y_gru + w_trans*y_trans
- Produces Kaggle submission predictions for (x, y)

This file is used in the Kaggle notebook environment.

---

## 🗂️ Folder Structure

2026-NFL-Data-Bowl/
├─ preprocess.py
├─ model_torch.py
├─ train_torch.py
├─ evaluate_torch_weighted.py
├─ inference_ensemble.py
│
├─ models/
│   ├─ model_lstm/best_model.pt
│   ├─ model_gru/best_model.pt
│   └─ model_transformer/best_model.pt
│
├─ data/
│   ├─ train/
│   └─ processed/
│
└─ plots/
    ├─ loss_curves/
    └─ rmse_per_horizon/

---

## 🎯 Summary

This codebase provides:

- Complete NFL tracking → deep learning pipeline
- Three advanced models (LSTM/GRU/Transformer)
- Physics-informed feature engineering
- Horizon-weighted ensemble system
- AMP-accelerated GPU training
- Kaggle-ready inference

## Git Tutorial

This quick guide will teach you how to:

- Clone the main repository  
- Create your own branch  
- Save (commit) your changes  
- Push your branch to the remote repository  

No prior Git knowledge required.

---

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Miller2545/2026-NFL-Data-Bowl.git

cd 2026-NFL-Data-Bowl
```

(Note: The above is if you are working in linux, for windows it will just create the folder 2026-NFL-Data-Bowl and you can access the work from there!)

### 2. Creating Your Working Branch

```bash
git checkout main
git pull
```

The above ensures you are on our main branch and pulls any changes from github.

```bash
git checkout -b <gmuid>
```

This will create your own branch so that we have a main branch that we will always have something working code-wise, and our own branches that we change ourselves. With us all working in the same project this allows us to be able to work within the same files and not step on eachothers toes.

### 3. Gitting Committed

```bash
git status
```

This will show you if there were any changes made on your current branch that need to be committed and pushed. It should look something like:

```bash
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

If you haven't saved a file with changes yet.

When you save changes it will look like the following:

```bash
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md

no changes added to commit (use "git add" and/or "git commit -a")
```

Now since we have changes, we can either use the UI in either RStudio or VScode, or go to the command line and add, commit, and push our changes.

```bash
git add .
git commit -m "Added: git commit to tutorial section of README.md"
git push
```

#### 4. Profit

The following above will add all files to your commit, commit the changes with the message "Added: git commit to tutorial section of README.md", and then push the changes to github.

This will push it to your working branch. When it comes to merging all of our work into the main working branch, I can handle that to make is easier!
