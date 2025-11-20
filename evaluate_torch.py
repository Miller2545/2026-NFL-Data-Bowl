# evaluate_torch.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from preprocess import preprocess_inputs, build_training_rows
from model_torch import LSTMFrameConditionedModel


def parse_weeks(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return [f"{w:02d}" for w in range(a, b + 1)]
    return [f"{int(x):02d}" for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch model RMSE on holdout weeks")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True, help="e.g. '15-18'")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_path", type=Path, default=Path("artifacts/eval_preds_torch.csv"))
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    input_files = [args.input_dir / f"input_2023_w{w}.csv" for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    print(f"Evaluating on weeks: {weeks}")

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

    # Build model and load weights
    model = LSTMFrameConditionedModel(
        n_in_steps=n_in_steps,
        n_features=n_features,
        hidden_units=64,
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    X = torch.from_numpy(X_array).float().to(device)
    t = torch.from_numpy(t_input).float().to(device)

    preds_list = []
    batch_size = 4096

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            tb = t[i:i+batch_size]
            pb = model(xb, tb)  # [B, 2]
            preds_list.append(pb.cpu().numpy())

    preds = np.vstack(preds_list)  # [N, 2]
    Y = Y_array  # [N, 2]

    # ----- RMSE overall, per-component -----
    diff = preds - Y
    mse_overall = np.mean(diff[:, 0]**2 + diff[:, 1]**2)
    rmse_overall_vec = np.sqrt(mse_overall)
    rmse_x = np.sqrt(np.mean((preds[:, 0] - Y[:, 0])**2))
    rmse_y = np.sqrt(np.mean((preds[:, 1] - Y[:, 1])**2))

    print("\n==== RMSE Results (vector Euclidean) ====")
    print(f"RMSE_overall (sqrt(E[dx^2+dy^2])) : {rmse_overall_vec:.4f}")
    print(f"RMSE_x                          : {rmse_x:.4f}")
    print(f"RMSE_y                          : {rmse_y:.4f}")

    # ----- Competition-style RMSE -----
    N_float = float(N)
    comp_rmse = np.sqrt(np.sum(diff[:, 0]**2 + diff[:, 1]**2) / (2.0 * N_float))
    print("\n===============================")
    print(f"🏈 Competition RMSE (Leaderboard metric): {comp_rmse:.6f}")
    print("===============================")

    comp_path = args.model_path.parent / "competition_rmse_torch.txt"
    with open(comp_path, "w") as f:
        f.write(f"{comp_rmse:.6f}\n")
    print(f"[Saved] competition RMSE → {comp_path}")

    # ----- RMSE by t_input (like your table) -----
    df = rows.copy()
    df["pred_x"] = preds[:, 0]
    df["pred_y"] = preds[:, 1]
    df["t_int"] = df["t_input"].round().astype(int)

    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    by_t_records = []
    for t_int, grp in df.groupby("t_int"):
        px = grp["pred_x"].to_numpy()
        py = grp["pred_y"].to_numpy()
        tx = grp["x"].to_numpy()
        ty = grp["y"].to_numpy()

        rmse_overall = np.sqrt(np.mean((px - tx)**2 + (py - ty)**2))
        rmse_x_t = rmse(px, tx)
        rmse_y_t = rmse(py, ty)

        by_t_records.append({
            "t_int": t_int,
            "rmse_overall": rmse_overall,
            "rmse_x": rmse_x_t,
            "rmse_y": rmse_y_t,
            "n": float(len(grp)),
        })

    by_t = pd.DataFrame(by_t_records).sort_values("t_int")
    print("\nRMSE by t_input (frames after release):")
    print(by_t.to_string(index=False))

    rmse_table_path = args.model_path.parent / "rmse_by_t_torch.csv"
    by_t.to_csv(rmse_table_path, index=False)
    print(f"\n[Saved] RMSE by t_input → {rmse_table_path}")

    # Optional: save per-row preds
    if args.save_csv:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["week", "game_id", "play_id", "nfl_id", "frame_id", "t_input",
                "x", "y", "pred_x", "pred_y"]
        for col in ["week"]:
            if col not in df.columns:
                df[col] = np.nan
        df[cols].to_csv(args.csv_path, index=False)
        print(f"[Saved] per-row predictions → {args.csv_path}")


if __name__ == "__main__":
    main()
