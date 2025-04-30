# play_loader.py
"""Utility for fetching a single play from the big parquet file in the exact
format expected by the Transformer pipeline.

Example
-------
>>> from play_loader import load_play
>>> ctx, gt_future = load_play(
...     parquet_path="../../processed_data/full_plays.parquet",
...     game_id=2021090900,
...     play_id=75,
...     ctx_len=100,
...     n_future=40,
... )
>>> print(ctx.shape, gt_future.shape)  # (1,100,46) (1,40,46)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Sequence, Optional, Union

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# columns order identical to the dataset builder

_XY_COLS: Sequence[str] = [
    f"{axis}_{i}" for i in range(1, 24) for axis in ("x", "y")
]

_SNAP_COLS: Sequence[str] = [
    "event_ball_snap",
    "event_snap_direct",
]


def _read_play(
    parquet_path: Union[str, Path],
    game_id: int,
    play_id: int,
) -> pd.DataFrame:
    
    print(os.getcwd())
    """Stream‑read a single play; returns sorted DataFrame."""
    df = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
        filters=[("gameId", "==", game_id), ("playId", "==", play_id)],
    )
    if df.empty:
        raise ValueError(f"Play (gameId={game_id}, playId={play_id}) not found in parquet.")
    return df.sort_values("frameId").reset_index(drop=True)


def _pad_left(arr: np.ndarray, max_len: int) -> np.ndarray:
    """Pad/truncate to *max_len* rows (same logic as dataset builder)."""
    out = np.zeros((max_len, arr.shape[1]), dtype=np.float32)
    if len(arr) >= max_len:
        out[:] = arr[-max_len:]
    else:
        out[-len(arr):] = arr
    return out


def load_play(
    parquet_path: Union[str, Path],
    game_id: int,
    play_id: int,
    ctx_len: int = 100,
    n_future: int = 40,
    snap_cols: Optional[Sequence[str]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return `(context_seq, ground_truth_future)` tensors for a single play.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the *full_plays.parquet* file.
    game_id, play_id : int
        Identifiers for the play.
    ctx_len : int, default 100
        Number of frames provided as model context (≤ *ctx_len* after padding).
    n_future : int, default 40
        Number of frames to keep for ground‑truth comparison.
    snap_cols : list[str] | None
        Columns used to detect the first snap frame.  If *None*, defaults to
        `["event_ball_snap", "event_autoevent_ballsnap"]`.

    Returns
    -------
    context_seq : tf.Tensor
        Shape `(1, ctx_len, 46)` – left‑padded if necessary.
    gt_future_seq : tf.Tensor
        Shape `(1, n_future, 46)` – may be shorter if the play ends earlier.
    """
    snap_cols = list(snap_cols or _SNAP_COLS)

    df = _read_play(parquet_path, game_id, play_id)

    # find first snap row
    snap_idx = (df[snap_cols].sum(axis=1) >= 0.5).idxmax()

    # slice context & future indices
    start_ctx = max(0, snap_idx - ctx_len)
    end_future = snap_idx + n_future

    arr = df[_XY_COLS].to_numpy("float32")
    context = arr[start_ctx:snap_idx]
    future  = arr[snap_idx:end_future]

    context = _pad_left(context, ctx_len)               # (ctx_len,46)

    # add batch dims to match model & visualiser expectations
    context_b = tf.expand_dims(context, 0)
    future_b  = tf.expand_dims(future, 0)
    return context_b, future_b


__all__ = ["load_play"]
