# train_torch.py
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import tqdm

from preprocess import preprocess_inputs, build_training_rows
from model_torch import LSTMFrameConditionedModel, TransformerFrameConditionedModel, GRUFrameConditionedModel

warmup_epochs = 5

def parse_weeks(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return [f"{w:02d}" for w in range(a, b + 1)]
    return [f"{int(x):02d}" for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train LSTM frame-conditioned model (PyTorch + CUDA + AMP)")
    parser.add_argument("--input_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/train"))
    parser.add_argument("--weeks", type=str, required=True, help="e.g. '1-14' or '1,2,3'")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=Path, default=Path("models/lstm_frame_conditioned_torch"))
    parser.add_argument("--arch", type=str, default="lstm", choices=["lstm", "gru", "transformer"], help="Which architecture to train: 'lstm', 'gru', or 'transformer'",
)

    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    input_files = [args.input_dir / f"input_2023_w{w}.csv" for w in weeks]
    output_files = [args.output_dir / f"output_2023_w{w}.csv" for w in weeks]

    print(f"Weeks: {weeks}")
    print("Building training arrays using preprocess.py ...")

    prep = preprocess_inputs(input_files, n_frames=args.n_frames)
    X_array, t_input, Y_array, rows = build_training_rows(
        output_files,
        release_pois=prep["release_pois"],
        padded_sequences=prep["padded_sequences"],
    )

    print(f"X_array shape: {X_array.shape}  (N, time, features)")
    print(f"t_input shape: {t_input.shape}  (N, 1)")
    print(f"Y_array shape: {Y_array.shape}  (N, 2)")

    N, n_in_steps, n_features = X_array.shape

    # ---------- Device + AMP setup ----------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # can help with performance

    # Convert data to tensors
    X = torch.from_numpy(X_array).float()
    t = torch.from_numpy(t_input).float()
    Y = torch.from_numpy(Y_array).float()

    dataset = TensorDataset(X, t, Y)

    # Train/val split
    n_val = int(args.val_split * N)
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,          # try 4, 8, or 16 depending on CPU
        pin_memory=True,        # MUCH faster host→GPU transfers
        persistent_workers=True # keeps workers alive
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Build model
    if args.arch == "lstm":
        model = LSTMFrameConditionedModel(
            n_in_steps=n_in_steps,
            n_features=n_features,
            hidden_units=args.hidden_units,
        ).to(device)
    elif args.arch == "gru":
        model = GRUFrameConditionedModel(
            n_in_steps=n_in_steps,
            n_features=n_features,
            hidden_units=args.hidden_units,
        ).to(device)
    else:  # transformer
        model = TransformerFrameConditionedModel(
            n_in_steps=n_in_steps,
            n_features=n_features,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)

    print(f"Using architecture: {args.arch}")
    print(model)


    criterion = nn.MSELoss()
    if args.arch == "transformer":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-2,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-1,   # 10% of lr to start (1e-3 is *very* tiny)
        total_iters=warmup_epochs
    )

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=True,
    )

    # ---------- Mixed precision scaler ----------
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print(model)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for xb, tb, yb in prog_bar:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # ----- AMP forward + backward -----
            with torch.amp.autocast("cuda", enabled=use_cuda):
                preds = model(xb, tb)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

            current_train_loss = train_loss_sum / train_count
            prog_bar.set_postfix(loss=current_train_loss)

        train_loss = train_loss_sum / train_count

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, tb, yb in val_loader:
                xb = xb.to(device)
                tb = tb.to(device)
                yb = yb.to(device)

                # Use autocast in eval too (no GradScaler needed here)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    preds = model(xb, tb)
                    loss = criterion(preds, yb)

                bs = xb.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        val_loss = val_loss_sum / val_count
        # Step the right scheduler
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        # Log history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            torch.save(best_state, args.model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best_val_loss={best_val_loss:.4f}")
                break

    # Save final model
    torch.save(model.state_dict(), args.model_dir / "final_model.pt")

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(args.model_dir / "history_torch.csv", index=False)
    print(f"[Saved] history_torch.csv in {args.model_dir}")


if __name__ == "__main__":
    main()
