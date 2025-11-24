# evaluate_torch_weighted.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from preprocess import preprocess_inputs, build_training_rows
from model_torch import (
    LSTMFrameConditionedModel,
    GRUFrameConditionedModel,
    TransformerFrameConditionedModel,
)


# ------------------------------
# Helpers
# ------------------------------
def rmse(y_true, y_pred):
    """Root Mean Squared Error over (x,y) pairs."""
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def solve_softmax_weights(loss_lstm, loss_gru, loss_trans):
    """Softmax over negative losses → nonnegative weights that sum to 1."""
    losses = np.array([loss_lstm, loss_gru, loss_trans], dtype=np.float64)
    weights = np.exp(-losses)
    weights /= weights.sum()
    return weights


def solve_ridge_weights(y_true, preds_list, alpha=1e-3):
    """
    Ridge regression to learn per-horizon ensemble weights.

    y_true:   [N,2]
    preds_list: list of [N,2] arrays (LSTM, GRU, Transformer)
    """
    # Stack predictions: [N,2,3] -> [2N,3]
    X = np.stack(preds_list, axis=2).astype(np.float64)  # force float64
    X = X.reshape(-1, 3)                                  # [2N,3]
    Y = y_true.astype(np.float64).reshape(-1)             # [2N]

    # Rescale to avoid huge values in X^T X
    max_abs = max(np.abs(X).max(), np.abs(Y).max(), 1.0)
    Xs = X / max_abs
    Ys = Y / max_abs

    # Ridge: (X^T X + αI) w = X^T Y
    A = Xs.T @ Xs + alpha * np.eye(3, dtype=np.float64)
    b = Xs.T @ Ys

    w = np.linalg.solve(A, b)
    # Enforce nonnegative + normalize
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=np.float64)
    w /= s
    return w


def parse_weeks(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return [f"{w:02d}" for w in range(int(a), int(b) + 1)]
    return [f"{int(x):02d}" for x in s.split(",") if x.strip()]


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM/GRU/Transformer ensemble with per-horizon weights."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True, help="e.g. '15-18'")
    parser.add_argument("--n_frames", type=int, default=10)

    parser.add_argument("--model_lstm", type=Path, required=True)
    parser.add_argument("--model_gru", type=Path, required=True)
    parser.add_argument("--model_trans", type=Path, required=True)

    parser.add_argument("--out_dir", type=Path, default=Path("ensemble_results"))
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size to use during evaluation.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Weeks / file lists
    # ------------------------------
    weeks = parse_weeks(args.weeks)
    input_files = [args.input_dir / f"input_2023_w{w}.csv" for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    print(f"Evaluating on weeks: {weeks}")
    print("Loading + preprocessing validation data...")

    # ------------------------------
    # Preprocessing
    # ------------------------------
    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    X_array, t_input, Y_array, rows = build_training_rows(
        output_files,
        release_pois=prep["release_pois"],
        padded_sequences=prep["padded_sequences"],
    )

    print("Shapes:")
    print("  X_array:", X_array.shape, "(N, time, features)")
    print("  t_input:", t_input.shape, "(N, 1)")
    print("  Y_array:", Y_array.shape, "(N, 2)")

    N, T, F = X_array.shape

    # ------------------------------
    # Device
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------
    # Load models
    # ------------------------------
    print("Loading LSTM model from:", args.model_lstm)
    lstm = LSTMFrameConditionedModel(T, F, hidden_units=64).to(device)
    lstm.load_state_dict(torch.load(args.model_lstm, map_location=device))
    lstm.eval()

    print("Loading GRU model from:", args.model_gru)
    gru = GRUFrameConditionedModel(T, F, hidden_units=64).to(device)
    gru.load_state_dict(torch.load(args.model_gru, map_location=device))
    gru.eval()

    print("Loading Transformer model from:", args.model_trans)
    trans = TransformerFrameConditionedModel(
        n_in_steps=T,
        n_features=F,
        d_model=128,
        n_heads=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        conv_kernel_size=3
    ).to(device)
    trans.load_state_dict(torch.load(args.model_trans, map_location=device))
    trans.eval()

    # ------------------------------
    # Predict all samples (batched)
    # ------------------------------
    print("Running predictions (batched)...")

    X = torch.from_numpy(X_array).float()
    t = torch.from_numpy(t_input).float()

    batch_size = args.batch_size
    use_cuda_amp = (device.type == "cuda")

    pred_lstm_list = []
    pred_gru_list = []
    pred_trans_list = []

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Batches"):
            end = min(start + batch_size, N)
            xb = X[start:end].to(device)
            tb = t[start:end].to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                out_lstm = lstm(xb, tb)       # [B,2]
                out_gru = gru(xb, tb)         # [B,2]
                out_trans = trans(xb, tb)     # [B,2]

            pred_lstm_list.append(out_lstm.cpu().numpy())
            pred_gru_list.append(out_gru.cpu().numpy())
            pred_trans_list.append(out_trans.cpu().numpy())

    pred_lstm = np.vstack(pred_lstm_list)   # [N,2]
    pred_gru = np.vstack(pred_gru_list)     # [N,2]
    pred_trans = np.vstack(pred_trans_list) # [N,2]

    print("Predictions complete.")

    # ------------------------------
    # Horizon (t_int) handling
    # ------------------------------
    if "t_int" in rows.columns:
        rows["t_int"] = rows["t_int"].astype(int)
    else:
        if "t_input" in rows.columns:
            rows["t_int"] = rows["t_input"].round().astype(int)
        else:
            rows["t_int"] = np.arange(len(rows), dtype=int)

    horizons = sorted(rows["t_int"].unique())

    # ------------------------------
    # Compute RMSE & weights per horizon
    # ------------------------------
    rmse_rows = []
    weight_rows = []

    print("Computing RMSE & optimal weights by horizon...")
    for h in horizons:
        idx = (rows["t_int"] == h).values
        if not idx.any():
            continue

        y_true = Y_array[idx]
        p_lstm = pred_lstm[idx]
        p_gru = pred_gru[idx]
        p_trans = pred_trans[idx]

        rmse_lstm = rmse(y_true, p_lstm)
        rmse_gru = rmse(y_true, p_gru)
        rmse_trans = rmse(y_true, p_trans)

        # Candidate weighting methods
        w_soft = solve_softmax_weights(rmse_lstm, rmse_gru, rmse_trans)
        w_ridge = solve_ridge_weights(y_true, [p_lstm, p_gru, p_trans], alpha=1e-3)

        ensemble_soft = (w_soft[0] * p_lstm +
                         w_soft[1] * p_gru +
                         w_soft[2] * p_trans)
        ensemble_ridge = (w_ridge[0] * p_lstm +
                          w_ridge[1] * p_gru +
                          w_ridge[2] * p_trans)

        rmse_soft = rmse(y_true, ensemble_soft)
        rmse_ridge = rmse(y_true, ensemble_ridge)

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
            rmse_lstm,
            rmse_gru,
            rmse_trans,
            best_rmse,
            method,
            int(idx.sum()),
        ])

        weight_rows.append([
            h,
            w_best[0],
            w_best[1],
            w_best[2],
            method,
        ])

    # ------------------------------
    # Save per-model/horizon RMSE + weights
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
            "n_samples",
        ],
    )

    weights_df = pd.DataFrame(
        weight_rows,
        columns=[
            "t_int",
            "w_lstm",
            "w_gru",
            "w_trans",
            "method",
        ],
    )

    rmse_path = args.out_dir / "rmse_by_model_and_horizon.csv"
    weights_path = args.out_dir / "ensemble_weights_by_t.csv"

    rmse_df.to_csv(rmse_path, index=False)
    weights_df.to_csv(weights_path, index=False)

    print("[Saved]", rmse_path)
    print("[Saved]", weights_path)

    # ------------------------------
    # Build full ensemble predictions using weights_by_h
    # ------------------------------
    weights_by_h = {
        int(row["t_int"]): (row["w_lstm"], row["w_gru"], row["w_trans"])
        for _, row in weights_df.iterrows()
    }

    preds_ens = np.zeros_like(pred_lstm, dtype=np.float64)
    t_ints = rows["t_int"].to_numpy()

    for i in range(N):
        h = int(t_ints[i])
        if h in weights_by_h:
            w_lstm, w_gru, w_trans = weights_by_h[h]
        else:
            w_lstm = w_gru = w_trans = 1.0 / 3.0

        preds_ens[i, :] = (
            w_lstm * pred_lstm[i]
            + w_gru * pred_gru[i]
            + w_trans * pred_trans[i]
        )

    # ------------------------------
    # Overall RMSE summary (like before)
    # ------------------------------
    diff = preds_ens - Y_array
    mse_vec = np.mean(diff[:, 0]**2 + diff[:, 1]**2)
    rmse_overall_vec = np.sqrt(mse_vec)
    rmse_x = rmse(Y_array[:, 0], preds_ens[:, 0])
    rmse_y = rmse(Y_array[:, 1], preds_ens[:, 1])

    # Competition-style RMSE:
    # sqrt( sum((dx^2 + dy^2)) / (2N) )
    comp_rmse = np.sqrt(
        np.sum(diff[:, 0]**2 + diff[:, 1]**2) / (2.0 * float(N))
    )

    print("\n==== Weighted Ensemble RMSE (validation) ====")
    print(f"RMSE_overall_vec : {rmse_overall_vec:.6f}")
    print(f"RMSE_x           : {rmse_x:.6f}")
    print(f"RMSE_y           : {rmse_y:.6f}")
    print(f"Competition RMSE : {comp_rmse:.6f}")
    print("=============================================")

    # Save competition RMSE to txt (like before)
    comp_path = args.out_dir / "ensemble_competition_rmse.txt"
    with open(comp_path, "w") as f:
        f.write(f"{comp_rmse:.6f}\n")
    print("[Saved]", comp_path)

    # ------------------------------
    # Per-horizon ensemble RMSE table (overall/x/y/n)
    # ------------------------------
    by_t_records = []
    for h in horizons:
        idx = (rows["t_int"] == h).values
        if not idx.any():
            continue

        px = preds_ens[idx, 0]
        py = preds_ens[idx, 1]
        tx = Y_array[idx, 0]
        ty = Y_array[idx, 1]

        rmse_overall_h = np.sqrt(np.mean((px - tx)**2 + (py - ty)**2))
        rmse_x_h = rmse(tx, px)
        rmse_y_h = rmse(ty, py)

        by_t_records.append({
            "t_int": int(h),
            "rmse_overall": rmse_overall_h,
            "rmse_x": rmse_x_h,
            "rmse_y": rmse_y_h,
            "n": int(idx.sum()),
        })

    by_t_df = pd.DataFrame(by_t_records).sort_values("t_int")

    print("\nEnsemble RMSE by t_int:")
    print(by_t_df.to_string(index=False))

    by_t_path = args.out_dir / "ensemble_rmse_by_t.csv"
    by_t_df.to_csv(by_t_path, index=False)
    print("[Saved]", by_t_path)

    # ------------------------------
    # Plot RMSE curves (per model & ensemble)
    # ------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_df["t_int"], rmse_df["rmse_lstm"], label="LSTM")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_gru"], label="GRU")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_trans"], label="Transformer")
    plt.plot(rmse_df["t_int"], rmse_df["rmse_ensemble"], label="Ensemble (per-horizon best)", linewidth=3)
    plt.xlabel("t_int (future frame)")
    plt.ylabel("RMSE")
    plt.title("RMSE by Model and Horizon")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path = args.out_dir / "rmse_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print("[Saved]", plot_path)


if __name__ == "__main__":
    main()
