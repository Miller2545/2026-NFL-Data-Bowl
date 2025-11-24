from pathlib import Path
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

FIELD_WIDTH = 53.3
FIELD_CENTER_Y = FIELD_WIDTH / 2.0
FIELD_LENGTH = 120.0
YARD_TO_METER = 0.9144

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
        "x", "y", "s", "a", "o", "dir",
        "ball_land_x", "ball_land_y",
        "absolute_yardline_number",
        "player_weight_kg",
        "s_mps", "a_mps2",
        "vx_mps", "vy_mps",
        "ax_mps2", "ay_mps2",
        "px", "py",
        "ke",
        "force_x", "force_y",
        "pos_id_norm",
        "role_id_norm",
        "side_id_norm",
        "dist_sideline_left",
        "dist_sideline_right",
        "dist_sideline_nearest",
        "dist_field_center",
        "dist_endzone",
        "dist_nearest_teammate",
        "dist_nearest_opponent",
        "opp_within_3",
        "opp_within_5",
        "opp_within_10",
    ]

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic physics features using speed (s), acceleration (a), direction (dir),
    and player_weight. Assumes:
      - s is in yards/second
      - a is in yards/second^2
      - dir is in degrees, 0 = pointing towards opponent's endzone, increasing clockwise
      - player_weight is in pounds (NFL data)
    """

    df = df.copy()

    # --- unit conversions ---
    # weight: pounds -> kg
    df["player_weight_kg"] = df["player_weight"] * 0.453592

    df["s_mps"] = df["s"] * YARD_TO_METER
    df["a_mps2"] = df["a"] * YARD_TO_METER

    # --- velocity components from speed + direction ---
    # NFL tracking: dir is usually degrees from x-axis
    # We'll treat 0 degrees as "toward +x" (right), and standard math orientation.
    # If the competition docs define differently, adjust here.
    theta = np.deg2rad(df["dir"].fillna(0.0))
    df["vx_mps"] = df["s_mps"] * np.cos(theta)
    df["vy_mps"] = df["s_mps"] * np.sin(theta)

    # --- acceleration components ---
    df["ax_mps2"] = df["a_mps2"] * np.cos(theta)
    df["ay_mps2"] = df["a_mps2"] * np.sin(theta)

    # --- momentum components: p = m * v ---
    df["px"] = df["player_weight_kg"] * df["vx_mps"]
    df["py"] = df["player_weight_kg"] * df["vy_mps"]

    # --- kinetic energy: KE = 0.5 * m * v^2 ---
    df["ke"] = 0.5 * df["player_weight_kg"] * (df["s_mps"] ** 2)

    # --- "force-like" proxy: F = m * a ---
    df["force_x"] = df["player_weight_kg"] * df["ax_mps2"]
    df["force_y"] = df["player_weight_kg"] * df["ay_mps2"]

    return df

def add_categorical_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode player_position, player_side, player_role as numeric features.
    This uses simple integer encodings scaled to [0, 1] so that models
    can still exploit them without explicit embeddings.
    """

    df = df.copy()

    # position
    df["player_position_cat"] = df["player_position"].astype("category")
    df["pos_id"] = df["player_position_cat"].cat.codes

    # role
    df["player_role_cat"] = df["player_role"].astype("category")
    df["role_id"] = df["player_role_cat"].cat.codes

    # side
    df["player_side_cat"] = df["player_side"].astype("category")
    df["side_id"] = df["player_side_cat"].cat.codes

    # Normalize IDs to [0, 1] (rough but effective)
    for col in ["pos_id", "role_id", "side_id"]:
        max_val = df[col].max()
        if max_val > 0:
            df[col + "_norm"] = df[col] / max_val
        else:
            df[col + "_norm"] = 0.0

    return df

def add_field_geometry_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-player field geometry features and multi-agent interaction features.

    Expects columns:
      - game_id, play_id, frame_id
      - x, y  (already normalized so offense goes to the right)
      - player_side (same value => teammate, different => opponent)
    """

    # ---------- 1) Field geometry features ----------
    df = df.copy()

    # Distance to each sideline
    df["dist_sideline_left"] = df["y"]
    df["dist_sideline_right"] = FIELD_WIDTH - df["y"]

    # Nearest sideline
    df["dist_sideline_nearest"] = np.minimum(
        df["dist_sideline_left"], df["dist_sideline_right"]
    )

    # Distance to center stripe
    df["dist_field_center"] = (df["y"] - FIELD_CENTER_Y).abs()

    # Distance to far endzone (assuming offense always to the right)
    df["dist_endzone"] = FIELD_LENGTH - df["x"]

    # We'll compute these per (game, play, frame)
    def per_frame_features(g: pd.DataFrame) -> pd.DataFrame:
        """
        g: rows for a single (game_id, play_id, frame_id)
        """
        n = len(g)
        if n <= 1:
            # Edge case: only one player? Just fill NaNs / zeros.
            return g.assign(
                dist_nearest_teammate=np.nan,
                dist_nearest_opponent=np.nan,
                opp_within_3=0,
                opp_within_5=0,
                opp_within_10=0,
            )

        # Positions [n,2]
        pos = g[["x", "y"]].to_numpy(dtype=np.float32)

        # Pairwise distances [n,n]
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        dmat = np.sqrt(dx * dx + dy * dy)  # Euclidean distance

        # Prevent self-distance from being counted
        np.fill_diagonal(dmat, np.inf)

        # Team masks: same side vs opponent
        side = g["player_side"].to_numpy()
        same_team = side[:, None] == side[None, :]   # [n,n] bool
        opp_team = ~same_team

        # Distances to teammates/opponents (inf where not applicable)
        d_team = np.where(same_team, dmat, np.inf)
        d_opp = np.where(opp_team, dmat, np.inf)

        # Nearest teammate / opponent distance
        dist_nearest_teammate = np.min(d_team, axis=1)
        dist_nearest_opponent = np.min(d_opp, axis=1)

        # If a player has no valid teammate/opponent (e.g., special cases), set NaN
        dist_nearest_teammate[~np.isfinite(dist_nearest_teammate)] = np.nan
        dist_nearest_opponent[~np.isfinite(dist_nearest_opponent)] = np.nan

        # Counts of opponents within certain radii
        opp_within_3 = np.sum(d_opp < 3.0, axis=1)
        opp_within_5 = np.sum(d_opp < 5.0, axis=1)
        opp_within_10 = np.sum(d_opp < 10.0, axis=1)

        return g.assign(
            dist_nearest_teammate=dist_nearest_teammate,
            dist_nearest_opponent=dist_nearest_opponent,
            opp_within_3=opp_within_3,
            opp_within_5=opp_within_5,
            opp_within_10=opp_within_10,
        )

    df = (
        df.groupby(["game_id", "play_id", "frame_id"], group_keys=False)
          .apply(per_frame_features)
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
# 1) INPUT PREPROCESSOR
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

    input_all = _normalize_xy(input_all)

    input_all = add_physics_features(input_all)

    input_all = add_categorical_numeric_features(input_all)

    input_all = add_field_geometry_and_interaction_features(input_all)

    # Only clean columns we actually feed into the model
    feat_cols = [c for c in INPUT_FEATURE_COLS if c in input_all.columns]

    # Replace inf/-inf -> NaN, then NaN -> 0
    input_all[feat_cols] = (
        input_all[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

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
