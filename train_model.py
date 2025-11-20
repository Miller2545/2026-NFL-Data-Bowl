# train.py
# Train the LSTM frame-conditioned model using the new preprocessing utilities.

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import time
from preprocess import preprocess_inputs, build_training_rows
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

def build_model(n_in_steps: int, n_features: int, hidden_units: int = 64) -> keras.Model:
    """Constructs the Keras model: Masking -> LSTM stack + Dense(time) -> concat -> MLP -> xy"""

    # Sequence input: [batch, time, features]
    seq_input = keras.Input(shape=(n_in_steps, n_features), name="seq_input")
    # Scalar time input: [batch, 1]
    time_in   = keras.Input(shape=(1,), name="time_input")

    # Sequence encoder
    x = layers.Masking(mask_value=0.0)(seq_input)

    # First LSTM layer (returns sequences so a second LSTM can refine)
    x = layers.LSTM(units=hidden_units, return_sequences=True)(x)

    # Second LSTM layer (encodes the whole history into a single vector)
    x = layers.LSTM(units=hidden_units, return_sequences=False)(x)

    # Regularization
    x = layers.Dropout(0.2)(x)

    # Time branch
    t = layers.Dense(16, activation="relu")(time_in)

    # Merge sequence + time encoding
    merged = layers.Concatenate()([x, t])

    # MLP head
    z = layers.Dense(128, activation="relu")(merged)
    z = layers.Dense(64, activation="relu")(z)

    # Final (x, y) prediction
    out = layers.Dense(2, name="xy_output")(z)

    model = keras.Model(inputs=[seq_input, time_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")

    return model

def parse_weeks(s: str):
    """Parse a weeks string like '1-18' or '1,2,3,5' into a sorted list of zero-padded week strings."""
    s = s.strip()
    weeks = []
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        weeks = [f"{w:02d}" for w in range(a, b + 1)]
    else:
        weeks = [f"{int(x):02d}" for x in s.split(",") if x.strip()]
    return sorted(set(weeks))

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train LSTM model on NFL tracking data.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"), help="Directory with input_*.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"), help="Directory with output_*.csv")
    parser.add_argument("--weeks", type=str, default="01-18", help="Weeks to include, e.g. '1-18' or '1,2,5'")
    parser.add_argument("--n_frames", type=int, default=10, help="Sequence length (frames) before release")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--hidden_units", type=int, default=64, help="LSTM hidden units")
    parser.add_argument("--model_dir", type=Path, default=Path("models/lstm_frame_conditioned_tf"), help="Output model directory")
    parser.add_argument("--save_arrays", action="store_true", help="Save X/Y/time arrays to disk for debugging")
    args = parser.parse_args()

    # Resolve files from weeks pattern
    weeks = parse_weeks(args.weeks)
    input_files  = [args.input_dir  / f"input_2023_w{w}.csv"  for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    # --- Preprocess (train/test-agnostic part)
    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    release_pois     = prep["release_pois"]
    padded_sequences = prep["padded_sequences"]

    # --- Build training rows (needs outputs with ground truth)
    X_array, time_input, Y_array, train_rows = build_training_rows(
        output_files,
        release_pois=release_pois,
        padded_sequences=padded_sequences
    )

    # --- Optionally save arrays for inspection
    if args.save_arrays:
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_data.npy", X_array)
        np.save(out_dir / "Y_data.npy", Y_array)
        np.save(out_dir / "time_input.npy", time_input)
        train_rows.to_parquet(out_dir / "train_rows.parquet")
        print(f"[Artifacts] Saved arrays and train_rows to {out_dir}")

    # --- Build & train model
    n_in_steps = X_array.shape[1]
    n_features = X_array.shape[2]
    model = build_model(n_in_steps, n_features, hidden_units=args.hidden_units)
    model.summary()

    # A couple of handy callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,       # halve LR when val_loss stops improving
            patience=2,       # wait 2 epochs before reducing
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,       # stop if no improvement for 4 epochs
            min_delta=1e-3,   # require at least 0.001 improvement
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=args.model_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # Recompile with LR scheduler tracking
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    history = model.fit(
        x={"seq_input": X_array, "time_input": time_input},
        y=Y_array,
        batch_size=args.batch_size,
        epochs=50,            # give room; EarlyStopping will stop earlier
        validation_split=args.val_split,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )

    # --- Save model + manifest
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / "best_model.keras")

    # --- Calculate and log total time ---
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    print(f"\n⏱️  Total training time: {elapsed:.2f} seconds ({elapsed_min:.2f} minutes)\n")

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "weeks": weeks,
        "n_frames": args.n_frames,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_split": args.val_split,
        "hidden_units": args.hidden_units,
        "model_dir": str(args.model_dir),
        "n_samples": int(X_array.shape[0]),
        "n_in_steps": int(n_in_steps),
        "n_features": int(n_features),
        "history_keys": list(history.history.keys()),
    }
    with open(args.model_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Model] Saved to {args.model_dir}")
    print(f"[Manifest] {args.model_dir / 'training_manifest.json'}")

    # Convert history.history to a DataFrame for plotting and saving
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = args.model_dir / "history.csv"
    hist_df.to_csv(hist_csv_file, index=False)

    # Plot the loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Training Curve", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the figure
    plot_path = args.model_dir / "loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[Plot] Saved training curve to {plot_path}")
    print(f"[CSV]  Saved history to {hist_csv_file}")

if __name__ == "__main__":
    # Make TF logs quieter if desired
    tf.get_logger().setLevel("ERROR")
    main()
