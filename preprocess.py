from pathlib import Path
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ----------------------------
# Shared helpers/readers
# ----------------------------
INPUT_DTYPES = {
    "game_id": "float64",
    "play_id": "float64",
    "player_to_predict": "boolean",
    "nfl_id": "float64",
    "frame_id": "float64",
    "play_direction": "string",
    "absolute_yardline_number": "float64",
    "player_name": "string",
    "player_height": "string",
    "player_weight": "float64",
    "player_birth_date": "string",
    "player_position": "string",
    "player_side": "string",
    "player_role": "string",
    "x": "float64",
    "y": "float64",
    "s": "float64",
    "a": "float64",
    "o": "float64",
    "dir": "float64",
    "num_frames_output": "float64",
    "ball_land_x": "float64",
    "ball_land_y": "float64",
    "week": "float64",
}

OUTPUT_DTYPES = {
    "game_id": "float64",
    "play_id": "float64",
    "nfl_id": "float64",
    "frame_id": "float64",
    "x": "float64",
    "y": "float64",
    "week": "float64",
}

INPUT_FEATURE_COLS = [
    "x",
    "y",
    "s",
    "a",
    "o",
    "dir",
    "ball_land_x",
    "ball_land_y",
    "absolute_yardline_number",
    "dx",
    "dy",
    "ds",
    "rel_x_ball",
    "rel_y_ball",
    "dist_to_ball_land",
]

def _add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple physics/geometry-based features:
      - dx, dy: frame-to-frame deltas in x,y
      - ds: frame-to-frame delta in speed
      - rel_x_ball, rel_y_ball: position relative to ball landing
      - dist_to_ball_land: distance to ball landing
    Assumes x,y,ball_land_x,ball_land_y are already normalized.
    """
    # Ensure sorted per player over time
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"], kind="mergesort")

    g = df.groupby(["game_id", "play_id", "nfl_id"], sort=False)

    # frame-to-frame deltas
    df["dx"] = g["x"].diff().fillna(0.0)
    df["dy"] = g["y"].diff().fillna(0.0)
    df["ds"] = g["s"].diff().fillna(0.0)

    # distance & relative position to ball landing
    # (if ball_land_x/y are missing or NaN, we just get NaNs; that’s fine)
    df["rel_x_ball"] = df["ball_land_x"] - df["x"]
    df["rel_y_ball"] = df["ball_land_y"] - df["y"]
    df["dist_to_ball_land"] = np.sqrt(
        (df["rel_x_ball"] ** 2) + (df["rel_y_ball"] ** 2)
    )

    return df

def _extract_week_from_name(name: str) -> Optional[int]:
    m = re.search(r"w([0-9]{2})", name)
    return int(m.group(1)) if m else None

def _read_input_week(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=INPUT_DTYPES)
    if "week" not in df.columns:
        wk = _extract_week_from_name(path.name)
        if wk is not None:
            df["week"] = wk
    return df

def _read_output_week(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=OUTPUT_DTYPES)
    if "week" not in df.columns:
        wk = _extract_week_from_name(path.name)
        if wk is not None:
            df["week"] = wk
    return df

def _normalize_xy(df: pd.DataFrame) -> pd.DataFrame:
    right = df["play_direction"].astype("string").str.lower().eq("right")
    # x/y
    df.loc[~right, "x"] = 120.0 - df.loc[~right, "x"]
    df.loc[~right, "y"] = 53.3  - df.loc[~right, "y"]
    # ball landing
    if "ball_land_x" in df.columns:
        df.loc[~right, "ball_land_x"] = 120.0 - df.loc[~right, "ball_land_x"]
    if "ball_land_y" in df.columns:
        df.loc[~right, "ball_land_y"] = 53.3  - df.loc[~right, "ball_land_y"]
    return df

def _last_frame_idx(df: pd.DataFrame, keys: List[str]) -> pd.Index:
    return (
        df.groupby(keys, dropna=False)["frame_id"]
        .idxmax()
        .dropna()
        .astype(int)
    )

def _build_padded_for_group(df: pd.DataFrame, n_frames: int) -> Tuple[int, np.ndarray]:
    # ensure columns exist
    for c in INPUT_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.sort_values("frame_id", kind="mergesort")
    take = df[INPUT_FEATURE_COLS].tail(n_frames).to_numpy(dtype=float)
    real_len = take.shape[0]
    if real_len < n_frames:
        pad = np.zeros((n_frames - real_len, take.shape[1]), dtype=float)
        take = np.vstack([pad, take])  # pad on top
    return real_len, take  # shape (n_frames, 9)


# ============================================================
# 1) TRAIN/TEST-AGNOSTIC INPUT PREPROCESSOR
#    Loads inputs, normalizes, finds release context, and
#    returns padded sequences per (game, play, nfl_id).
# ============================================================
def preprocess_inputs(
    input_files: List[Path],
    n_frames: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    input_files : list of Paths to input CSVs (train or test)
    n_frames    : number of frames for padded sequences

    Returns
    -------
    dict with:
        - input_all: normalized input dataframe
        - release_ctx: last frame per (game_id, play_id, nfl_id)
        - release_pois: subset where player_to_predict == True
        - padded_sequences: DataFrame with columns:
            [game_id, play_id, nfl_id, release_frame, real_len, padded_mat]
            where padded_mat is a (n_frames, 9) numpy array
    """
    # Load and concat
    input_all = pd.concat([_read_input_week(Path(p)) for p in input_files], ignore_index=True)

    # Normalize coordinates
    input_all = _normalize_xy(input_all)

    # Add physics features
    input_all = _add_physics_features(input_all)

    # Release context (max frame per player in play)
    idx = _last_frame_idx(input_all, ["game_id","play_id","nfl_id"])
    release_ctx = input_all.loc[idx].copy().reset_index(drop=True)

    # Players to predict
    release_pois = release_ctx.loc[release_ctx["player_to_predict"].fillna(False)].copy()

    # Attach release_frame to every row
    rf_map = release_ctx[["game_id","play_id","nfl_id","frame_id"]].rename(columns={"frame_id":"release_frame"})
    input_joined = input_all.merge(rf_map, on=["game_id","play_id","nfl_id"], how="left", validate="m:1")

    # Build padded sequences per (game, play, player, release_frame)
    gj = input_joined[~input_joined["release_frame"].isna()].copy()
    group_keys = ["game_id","play_id","nfl_id","release_frame"]

    records = []
    for keys, g in gj.groupby(group_keys, sort=False):
        real_len, mat = _build_padded_for_group(g, n_frames=n_frames)
        records.append({
            "game_id": keys[0],
            "play_id": keys[1],
            "nfl_id":  keys[2],
            "release_frame": keys[3],
            "real_len": real_len,
            "padded_mat": mat,  # (n_frames, 9)
        })
    padded_sequences = pd.DataFrame(records)

    return {
        "input_all": input_all,
        "release_ctx": release_ctx,
        "release_pois": release_pois,
        "padded_sequences": padded_sequences,
    }


# ============================================================
# 2) TRAINING ROWS (requires outputs with ground truth)
# ============================================================
def build_training_rows(
    output_files: List[Path],
    release_pois: pd.DataFrame,
    padded_sequences: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Expands POI sequences across *future frames from outputs*
    and returns model-ready arrays.

    Returns
    -------
    X_array    : (N, n_frames, 9)
    time_input : (N, 1) from frame_id
    Y_array    : (N, 2) from output x,y
    rows_df    : the merged dataframe (keys & any extra columns)
    """
    output_all = pd.concat([_read_output_week(Path(p)) for p in output_files], ignore_index=True)

    # filter output to only POIs
    output_pois = output_all.merge(
        release_pois[["game_id","play_id","nfl_id","num_frames_output"]],
        on=["game_id","play_id","nfl_id"],
        how="inner",
    )

    # Join POIs with padded sequences
    pois_with_seq = release_pois[["game_id","play_id","nfl_id","num_frames_output"]].merge(
        padded_sequences[["game_id","play_id","nfl_id","padded_mat"]],
        on=["game_id","play_id","nfl_id"],
        how="inner",
    )

    # Build training rows
    train_frames = output_pois.merge(
        pois_with_seq[["game_id","play_id","nfl_id","num_frames_output","padded_mat"]],
        on=["game_id","play_id","nfl_id"],
        how="inner",
    )

    # time input (use frame_id as in R)
    train_frames["t_input"] = train_frames["frame_id"].astype(float)

    # tensors
    X_list = train_frames["padded_mat"].tolist()
    n_frames = X_list[0].shape[0]
    n_feats  = X_list[0].shape[1]
    assert all(x.shape == (n_frames, n_feats) for x in X_list)

    X_array    = np.stack(X_list, axis=0)                                # (N, T, F)
    time_input = train_frames["t_input"].to_numpy(float).reshape(-1, 1)  # (N, 1)
    Y_array    = train_frames[["x","y"]].to_numpy(float)                 # (N, 2)

    return X_array, time_input, Y_array, train_frames


# ============================================================
# 3) INFERENCE ROWS (no ground truth; provide frames to predict)
# ============================================================
def build_inference_rows(
    future_frames_df: pd.DataFrame,
    release_pois: pd.DataFrame,
    padded_sequences: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Parameters
    ----------
    future_frames_df : DataFrame with at least
        ['week','game_id','play_id','nfl_id','frame_id'] for frames to predict
    release_pois     : from preprocess_inputs (subset of release players)
    padded_sequences : from preprocess_inputs

    Returns
    -------
    X_array    : (N, n_frames, 9)
    time_input : (N, 1) using provided future frame_id
    rows_df    : merged rows with keys (useful to reattach predictions)
    """
    # limit frames to POIs
    to_pred = future_frames_df.merge(
        release_pois[["game_id","play_id","nfl_id"]],
        on=["game_id","play_id","nfl_id"],
        how="inner",
    )

    # attach sequences
    to_pred = to_pred.merge(
        padded_sequences[["game_id","play_id","nfl_id","padded_mat"]],
        on=["game_id","play_id","nfl_id"],
        how="inner",
    )

    to_pred["t_input"] = to_pred["frame_id"].astype(float)

    X_list = to_pred["padded_mat"].tolist()
    n_frames = X_list[0].shape[0]
    n_feats  = X_list[0].shape[1]
    assert all(x.shape == (n_frames, n_feats) for x in X_list)

    X_array    = np.stack(X_list, axis=0)                               # (N, T, F)
    time_input = to_pred["t_input"].to_numpy(float).reshape(-1, 1)      # (N, 1)

    return X_array, time_input, to_pred
