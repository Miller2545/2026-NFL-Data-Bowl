# evaluate_torch_weighted.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from preprocess import preprocess_inputs, build_training_rows
from model_torch import (
    LSTMFrameConditionedModel,
    GRUFrameConditionedModel,
    TransformerFrameConditionedModel,
)

# ------------------------------
# RMSE helper
# ------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


# ------------------------------
# Solve non-negative weights
# ------------------------------
def solve_softmax_weights(loss_lstm, loss_gru, loss_trans):
    losses = np.array([loss_lstm, loss_gru, loss_trans])
    weights = np.exp(-losses)
    weights /= weights.sum()
    return weights


def solve_ridge_weights(y_true, preds_list, alpha=1e-3):
    # preds_list = [pred_lstm, pred_gru, pred_trans] (each shape [N,2])
    X = np.stack(preds_list, axis=2)  # [N,2,3]
    X = X.reshape(-1, 3)              # flatten coords: [2N,3]
    Y = y_true.reshape(-1)            # [2N]

    A = X.T @ X + alpha * np.eye(3)
    b = X.T @ Y

    w = np.linalg.solve(A, b)
    w = np.maximum(w, 0)
    w /= w.sum()
    return w


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--model_lstm", type=Path, required=True)
    parser.add_argument("--model_gru", type=Path, required=True)
    parser.add_argument("--model_trans", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("ensemble"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Parse weeks
    # ------------------------------
    def parse_weeks(s):
        s = s.strip()
        if "-" in s:
            a, b = s.split("-", 1)
            return [f"{w:02d}" for w in range(int(a), int(b) + 1)]
        return [f"{int(x):02d}" for x in s.split(",")]

    weeks = parse_weeks(args.weeks)
    input_files = [args.input_dir / f"input_2023_w{w}.csv" for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    print("Loading + preprocessing validation data...")
    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    X_array, t_input, Y_array, rows = build_training_rows(
        output_files,
        release_pois=prep["release_pois"],
        padded_sequences=prep["padded_sequences"],
    )

    print("Shapes:")
    print("X_array:", X_array.shape)
    print("t_input:", t_input.shape)
    print("Y_array:", Y_array.shape)

    # ------------------------------
    # Device
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------------
    # Load Models
    # ------------------------------
    N, T, F = X_array.shape

    print("Loading LSTM...")
    lstm = LSTMFrameConditionedModel(T, F).to(device)
    lstm.load_state_dict(torch.load(args.model_lstm, map_location=device))
    lstm.eval()

    print("Loading GRU...")
    gru = GRUFrameConditionedModel(T, F).to(device)
    gru.load_state_dict(torch.load(args.model_gru, map_location=device))
    gru.eval()

    print("Loading Transformer...")
    trans = TransformerFrameConditionedModel(
        n_in_steps=T,
        n_features=F,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)
    trans.load_state_dict(torch.load(args.model_trans, map_location=device))
    trans.eval()

    # ------------------------------
    # Predict all samples
    # ------------------------------
    print("Running predictions...")

    X = torch.from_numpy(X_array).float().to(device)
    t = torch.from_numpy(t_input).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_lstm = lstm(X, t).cpu().numpy()
            pred_gru = gru(X, t).cpu().numpy()
            pred_trans = trans(X, t).cpu().numpy()

    print("Predictions complete.")

    # ------------------------------
    # Compute RMSE per horizon
    # ------------------------------
    rows["t_int"] = rows["t_int"].astype(int)
    horizons = sorted(rows["t_int"].unique())

    rmse_rows = []
    weight_rows = []

    print("Computing RMSE & optimal weights by horizon...")

    for h in horizons:
        idx = (rows["t_int"] == h).values
        y_true = Y_array[idx]

        p_lstm = pred_lstm[idx]
        p_gru = pred_gru[idx]
        p_trans = pred_trans[idx]

        rmse_lstm = rmse(y_true, p_lstm)
        rmse_gru = rmse(y_true, p_gru)
        rmse_trans = rmse(y_true, p_trans)

        # weight methods
        w_soft = solve_softmax_weights(rmse_lstm, rmse_gru, rmse_trans)
        w_ridge = solve_ridge_weights(y_true, [p_lstm, p_gru, p_trans], alpha=1e-3)

        # Evaluate both
        ensemble_soft = (w_soft[0] * p_lstm +
                          w_soft[1] * p_gru +
                          w_soft[2] * p_trans)
        ensemble_ridge = (w_ridge[0] * p_lstm +
                           w_ridge[1] * p_gru +
                           w_ridge[2] * p_trans)

        rmse_soft = rmse(y_true, ensemble_soft)
        rmse_ridge = rmse(y_true, ensemble_ridge)

        # choose best
        if rmse_soft <= rmse_ridge:
            best_rmse = rmse_soft
            w_best = w_soft
            method = "softmax"
        else:
            best_rmse = rmse_ridge
            w_best = w_ridge
            method = "ridge"

        rmse_rows.append([
            h,
            rmse_lstm, rmse_gru, rmse_trans,
            best_rmse,
            method
        ])

        weight_rows.append([
            h,
            w_best[0], w_best[1], w_best[2],
            method
        ])

    # ------------------------------
    # Save tables
    # ------------------------------
    rmse_df = pd.DataFrame(
        rmse_rows,
        columns=[
            "t_int",
            "rmse_lstm",
            "rmse_gru",
            "rmse_trans",
            "rmse_ensemble",
            "method",
        ]
    )

    weights_df = pd.DataFrame(
        weight_rows,
        columns=[
            "t_int",
            "w_lstm",
            "w_gru",
            "w_trans",
            "method",
        ]
    )

    rmse_path = args.out_dir / "rmse_by_model_and_horizon.csv"
    weights_path = args.out_dir / "ensemble_weights_by_t.csv"

    rmse_df.to_csv(rmse_path, index=False)
    weights_df.to_csv(weights_path, index=False)

    print("[Saved]", rmse_path)
    print("[Saved]", weights_path)

    # ------------------------------
    # Plot RMSE
    # ------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(rmse_df["t_int"], rmse_df["rmse_lstm"], label="LSTM")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_gru"], label="GRU")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_trans"], label="Transformer")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_ensemble"], label="Ensemble", linewidth=3)
    plt.xlabel("t_int (future frame)")
    plt.ylabel("RMSE")
    plt.title("RMSE by Model and Horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out_dir / "rmse_curve.png")
    print("[Saved]", args.out_dir / "rmse_curve.png")


if __name__ == "__main__":
    main()
