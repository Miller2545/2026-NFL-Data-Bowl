import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from preprocess import preprocess_inputs, build_training_rows
from model_torch import (
    LSTMFrameConditionedModel,
    GRUFrameConditionedModel,
    TransformerFrameConditionedModel,
)


def parse_weeks(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return [f"{w:02d}" for w in range(a, b + 1)]
    return [f"{int(x):02d}" for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation: LSTM + GRU (+ optional Transformer)"
    )
    parser.add_argument("--lstm_model_path", type=Path, required=True,
                        help="Path to best LSTM model .pt")
    parser.add_argument("--gru_model_path", type=Path, required=True,
                        help="Path to best GRU model .pt")
    parser.add_argument("--transformer_model_path", type=Path, default=None,
                        help="Optional path to Transformer model .pt")
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True,
                        help="Holdout weeks to evaluate on, e.g. '15-18'")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_path", type=Path,
                        default=Path("artifacts/ensemble_preds_torch.csv"))
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    input_files = [args.input_dir / f"input_2023_w{w}.csv" for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    print(f"Evaluating ensemble on weeks: {weeks}")

    # ---------- Build evaluation arrays ----------
    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    X_array, t_input, Y_array, rows = build_training_rows(
        output_files,
        release_pois=prep["release_pois"],
        padded_sequences=prep["padded_sequences"],
    )

    N, n_in_steps, n_features = X_array.shape
    print(f"X_array shape: {X_array.shape}")
    print(f"t_input shape: {t_input.shape}")
    print(f"Y_array shape: {Y_array.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Build & load models ----------
    models = []

    # LSTM
    lstm_model = LSTMFrameConditionedModel(
        n_in_steps=n_in_steps,
        n_features=n_features,
        hidden_units=64,
    ).to(device)
    lstm_state = torch.load(args.lstm_model_path, map_location=device)
    lstm_model.load_state_dict(lstm_state)
    lstm_model.eval()
    models.append(("lstm", lstm_model))

    # GRU
    gru_model = GRUFrameConditionedModel(
        n_in_steps=n_in_steps,
        n_features=n_features,
        hidden_units=64,
    ).to(device)
    gru_state = torch.load(args.gru_model_path, map_location=device)
    gru_model.load_state_dict(gru_state)
    gru_model.eval()
    models.append(("gru", gru_model))

    # Optional Transformer
    if args.transformer_model_path is not None:
        trans_model = TransformerFrameConditionedModel(
            n_in_steps=n_in_steps,
            n_features=n_features,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)
        trans_state = torch.load(args.transformer_model_path, map_location=device)
        trans_model.load_state_dict(trans_state)
        trans_model.eval()
        models.append(("transformer", trans_model))

    print("Ensemble members:", [name for name, _ in models])

    # ---------- Run ensemble predictions ----------
    X = torch.from_numpy(X_array).float().to(device)
    t = torch.from_numpy(t_input).float().to(device)

    preds_chunks = []
    bs = args.batch_size
    use_cuda_amp = (device.type == "cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda_amp):
        for i in range(0, N, bs):
            xb = X[i:i+bs]
            tb = t[i:i+bs]

            member_preds = []
            for name, m in models:
                p = m(xb, tb)     # [batch_i, 2]
                member_preds.append(p)

            # Average across models
            stacked = torch.stack(member_preds, dim=0)      # [n_models, batch_i, 2]
            p_ens = torch.mean(stacked, dim=0)              # [batch_i, 2]
            preds_chunks.append(p_ens.cpu().numpy())

    preds = np.vstack(preds_chunks)  # [N, 2]
    Y = Y_array                      # [N, 2]

    # ---------- Overall RMSE ----------
    diff = preds - Y
    mse_overall = np.mean(diff[:, 0]**2 + diff[:, 1]**2)
    rmse_overall_vec = np.sqrt(mse_overall)
    rmse_x = np.sqrt(np.mean((preds[:, 0] - Y[:, 0])**2))
    rmse_y = np.sqrt(np.mean((preds[:, 1] - Y[:, 1])**2))

    print("\n==== Ensemble RMSE Results (vector Euclidean) ====")
    print(f"RMSE_overall : {rmse_overall_vec:.4f}")
    print(f"RMSE_x       : {rmse_x:.4f}")
    print(f"RMSE_y       : {rmse_y:.4f}")

    # ---------- Competition-style RMSE (same formula as before) ----------
    N_float = float(N)
    comp_rmse = np.sqrt(np.sum(diff[:, 0]**2 + diff[:, 1]**2) / (2.0 * N_float))
    print("\n===============================")
    print(f"🏈 Ensemble Competition RMSE: {comp_rmse:.6f}")
    print("===============================")

    # Save competition metric next to LSTM model dir (arbitrary choice)
    out_dir = args.lstm_model_path.parent
    comp_path = out_dir / "ensemble_competition_rmse.txt"
    with open(comp_path, "w") as f:
        f.write(f"{comp_rmse:.6f}\n")
    print(f"[Saved] ensemble competition RMSE → {comp_path}")

    # ---------- RMSE by t_input (frame index) ----------
    df = rows.copy()
    df["pred_x_ensemble"] = preds[:, 0]
    df["pred_y_ensemble"] = preds[:, 1]
    df["t_int"] = df["t_input"].round().astype(int)

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    by_t_records = []
    for t_int, grp in df.groupby("t_int"):
        px = grp["pred_x_ensemble"].to_numpy()
        py = grp["pred_y_ensemble"].to_numpy()
        tx = grp["x"].to_numpy()
        ty = grp["y"].to_numpy()

        rmse_overall_t = np.sqrt(np.mean((px - tx)**2 + (py - ty)**2))
        rmse_x_t = rmse(px, tx)
        rmse_y_t = rmse(py, ty)

        by_t_records.append({
            "t_int": t_int,
            "rmse_overall": rmse_overall_t,
            "rmse_x": rmse_x_t,
            "rmse_y": rmse_y_t,
            "n": float(len(grp)),
        })

    by_t = pd.DataFrame(by_t_records).sort_values("t_int")
    print("\nRMSE by t_input (frames after release):")
    print(by_t.to_string(index=False))

    rmse_table_path = out_dir / "ensemble_rmse_by_t.csv"
    by_t.to_csv(rmse_table_path, index=False)
    print(f"\n[Saved] ensemble RMSE by t_input → {rmse_table_path}")

    # ---------- Optional: per-row CSV ----------
    if args.save_csv:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "week", "game_id", "play_id", "nfl_id", "frame_id", "t_input",
            "x", "y", "pred_x_ensemble", "pred_y_ensemble",
        ]
        if "week" not in df.columns:
            df["week"] = np.nan
        df[cols].to_csv(args.csv_path, index=False)
        print(f"[Saved] per-row ensemble predictions → {args.csv_path}")


if __name__ == "__main__":
    main()
