import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_squared_error

from preprocess import preprocess_inputs, build_training_rows

def parse_weeks(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return [f"{w:02d}" for w in range(a, b + 1)]
    return [f"{int(x):02d}" for x in s.split(",") if x.strip()]

def rmse(a, b):
    """Root mean squared error across all samples (vectorized)."""
    return np.sqrt(np.mean((a - b) ** 2))

def main():
    parser = argparse.ArgumentParser(description="Evaluate model RMSE on a (holdout) set.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to .keras model (e.g., models/.../best_model.keras)")
    parser.add_argument("--input_dir",  type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True, help="Weeks to evaluate on, e.g. '14-18' or '10,12,13'")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--save_csv", action="store_true", help="Save per-row predictions CSV")
    parser.add_argument("--csv_path", type=Path, default=Path("artifacts/eval_preds.csv"))
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    input_files  = [args.input_dir  / f"input_2023_w{w}.csv"  for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    # 1) Build arrays like the training set
    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    X, t_in, Y, rows = build_training_rows(
        output_files,
        release_pois=prep["release_pois"],
        padded_sequences=prep["padded_sequences"]
    )

    # 2) Load model
    model = keras.models.load_model(args.model_path)

    # 3) Predict
    preds = model.predict({"seq_input": X, "time_input": t_in}, batch_size=args.batch_size, verbose=1)

    # 4) RMSE metrics
    overall_rmse = rmse(preds, Y)
    rmse_x = rmse(preds[:, 0], Y[:, 0])
    rmse_y = rmse(preds[:, 1], Y[:, 1])

    print("\n==== RMSE Results ====")
    print(f"Overall RMSE : {overall_rmse:.4f}")
    print(f"RMSE_x       : {rmse_x:.4f}")
    print(f"RMSE_y       : {rmse_y:.4f}")

    # 5) RMSE by horizon (t_input rounded to int frames)
    df = rows.copy()
    df["pred_x"] = preds[:, 0]
    df["pred_y"] = preds[:, 1]
    df["err2"]   = (df["pred_x"] - df["x"])**2 + (df["pred_y"] - df["y"])**2
    df["t_int"]  = df["t_input"].round().astype(int)

    by_t = (
        df.groupby("t_int", as_index=False)
          .apply(lambda g: pd.Series({
              "rmse_overall": np.sqrt(((g["pred_x"] - g["x"])**2 + (g["pred_y"] - g["y"])**2).mean()),
              "rmse_x": np.sqrt(((g["pred_x"] - g["x"])**2).mean()),
              "rmse_y": np.sqrt(((g["pred_y"] - g["y"])**2).mean()),
              "n": len(g)
          }))
    )

    print("\nRMSE by t_input (frames after release):")
    print(by_t.sort_values("t_int").to_string(index=False))

    # Save RMSE-by-horizon table
    rmse_table_path = args.model_path.parent / "rmse_by_t.csv"
    by_t.to_csv(rmse_table_path, index=False)
    print(f"\n[Saved] RMSE by t_input → {rmse_table_path}")

    # Compute competition-style overall RMSE
    N = len(Y)
    diff_x = Y[:, 0] - preds[:, 0]
    diff_y = Y[:, 1] - preds[:, 1]

    comp_rmse = np.sqrt(np.sum(diff_x**2 + diff_y**2) / (2 * N))
    print("\n===============================")
    print(f"Competition RMSE (Leaderboard metric): {comp_rmse:.6f}")
    print("===============================")

    # Save it to a text file for easy reference
    rmse_path = args.model_path.parent / "competition_rmse.txt"
    with open(rmse_path, "w") as f:
        f.write(f"{comp_rmse:.6f}\n")
    print(f"[Saved] Single-value RMSE → {rmse_path}")


    # 6) CSV output (for debugging/plots)
    if args.save_csv:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_cols = [
            "week","game_id","play_id","nfl_id","frame_id","t_input",
            "x","y","pred_x","pred_y"
        ]
        # rows may not contain 'week' if not present -> handle gracefully
        for col in ["week"]:
            if col not in df.columns:
                df[col] = np.nan

        df[out_cols].to_csv(args.csv_path, index=False)
        print(f"\n[Saved] per-row predictions → {args.csv_path}")

if __name__ == "__main__":
    main()
