import os

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf

import kaggle_evaluation.nfl_inference_server

MODEL_PATH = "/kaggle/input/nfl-r-lstm-model/lstm_frame_conditioned_tf"

# If you used HDF5 instead, use e.g.:
# MODEL_PATH = "/kaggle/input/nfl-r-lstm-model/lstm_frame_conditioned.h5"

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3

FEATURE_COLS = [
    "x", "y",
    "s", "a",
    "o", "dir",
    "ball_land_x", "ball_land_y",
    "absolute_yardline_number",
]

N_FRAMES = 10

_model = None  # global cache so we only load once


def load_model_once():
    """Load the Keras model from disk only once."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def normalize_xy_py(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the R normalize_xy(): make all plays go 'right'."""
    df = df.copy()
    right = df["play_direction"] == "right"

    # x, y
    df.loc[~right, "x"] = FIELD_LENGTH - df.loc[~right, "x"]
    df.loc[~right, "y"] = FIELD_WIDTH  - df.loc[~right, "y"]

    # ball landing
    df.loc[~right, "ball_land_x"] = FIELD_LENGTH - df.loc[~right, "ball_land_x"]
    df.loc[~right, "ball_land_y"] = FIELD_WIDTH  - df.loc[~right, "ball_land_y"]

    return df


def build_sequence_for_player(df_input: pd.DataFrame,
                              game_id: float,
                              play_id: float,
                              nfl_id: float,
                              n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Build a (n_frames, len(FEATURE_COLS)) sequence for one (game, play, player):
    - use all frames seen so far (df_input accumulates over time)
    - sort by frame_id
    - take the last up to n_frames
    - pad with zeros at the TOP if fewer than n_frames
    """
    mask = (
        (df_input["game_id"] == game_id) &
        (df_input["play_id"] == play_id) &
        (df_input["nfl_id"] == nfl_id)
    )
    sub = df_input.loc[mask].sort_values("frame_id")

    # Select the same feature columns as in R
    sub = sub[FEATURE_COLS]

    if len(sub) > n_frames:
        sub = sub.iloc[-n_frames:, :]
    real_len = len(sub)

    if real_len < n_frames:
        pad = np.zeros((n_frames - real_len, len(FEATURE_COLS)), dtype=float)
        seq = np.vstack([pad, sub.to_numpy(dtype=float)])
    else:
        seq = sub.to_numpy(dtype=float)

    # Sanity check
    assert seq.shape == (n_frames, len(FEATURE_COLS))
    return seq


def build_batch_inputs(test_df: pd.DataFrame,
                       test_input_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    For all rows in test_df, build:
      X_batch: (N, 10, 9)
      t_batch: (N, 1)
    using the same logic as training:
      - sequence from last 10 frames in test_input
      - t_input = frame_id (raw frame index)
    """
    X_seqs = []
    t_vals = []

    for _, row in test_df.iterrows():
        game_id = row["game_id"]
        play_id = row["play_id"]
        nfl_id  = row["nfl_id"]
        frame_id = row["frame_id"]  # time index to predict at

        seq = build_sequence_for_player(
            test_input_df, game_id, play_id, nfl_id, n_frames=N_FRAMES
        )
        X_seqs.append(seq)
        t_vals.append([float(frame_id)])  # shape (1,)

    X_batch = np.stack(X_seqs, axis=0)          # (N, 10, 9)
    t_batch = np.array(t_vals, dtype=float)     # (N, 1)
    return X_batch, t_batch


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Kaggle evaluation entrypoint.
    test: rows we must predict (game_id, play_id, nfl_id, frame_id).
    test_input: tracking data for the same plays at current timestep (all players/frames seen so far).
    Must return a DataFrame with columns ['x', 'y'] of length == len(test).
    """
    model = load_model_once()

    # If no rows to predict, return empty
    if len(test) == 0:
        return pl.DataFrame({"x": [], "y": []})

    # Convert Polars -> Pandas because we love R logic we love R logic
    test_df = test.to_pandas()
    test_input_df = test_input.to_pandas()

    # Normalize coordinates to "right" direction like in R
    test_input_df = normalize_xy_py(test_input_df)

    # Build batch inputs: (N, 10, 9) and (N, 1)
    X_batch, t_batch = build_batch_inputs(test_df, test_input_df)

    # Run model:
    preds = model.predict(
        {"seq_input": X_batch, "time_input": t_batch},
        verbose=0
    )  # (N, 2)

    # Build predictions DataFrame
    pred_x = preds[:, 0]
    pred_y = preds[:, 1]

    predictions = pl.DataFrame(
        {
            "x": pred_x,
            "y": pred_y,
        }
    )

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == len(test)
    return predictions


# When your notebook is run on the hidden test set, inference_server.serve must be called within 10 minutes
# or the gateway will throw an error.
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
