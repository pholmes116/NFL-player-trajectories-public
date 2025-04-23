#!/usr/bin/env python
"""
Memory‑efficient builder for a Transformer‑ready dataset from
`model_input.parquet` (output of *polars_pipeline.py*).

1. **Streaming I/O – never loads the full parquet into RAM.**
   Plays are fetched one at a time with Polars `scan_parquet` + filters.
2. **Generator‑based dataset creation** – examples are yielded on‑the‑fly;
   no huge Python lists or `np.stack` calls.
3. **Constant‑size padding** implemented with a lightweight NumPy helper
   (faster & cheaper than `tf.keras.preprocessing.sequence.pad_sequences`).
4. **Direct TFRecord writing** – we stream examples into a `tf.data.Dataset`
   and call `Dataset.save()`.  Disk is the only bottleneck; peak RAM stays
   low, regardless of dataset size."""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple
from tqdm import tqdm

import numpy as np
import polars as pl
import tensorflow as tf


# ════════════════════════════════════════════════════════════════════════════════
# Feature / sequence constants
# ════════════════════════════════════════════════════════════════════════════════
NUM_ENTITIES: int = 23
COORDS: list[str] = ["x", "y"]
NUM_FEATS: int = NUM_ENTITIES * len(COORDS)  # 46
SEQ_MIN_LEN: int = 2
MAX_SEQ_LEN: int = 100  # default – overridable via CLI

POSITIONAL_FEATURES = [f"{c}_{i}"
                  for i in range(1, NUM_ENTITIES + 1)
                  for c in COORDS]

INPUT_FEATURES = POSITIONAL_FEATURES.copy()  # Change as needeed
TARGET_FEATURES = POSITIONAL_FEATURES.copy()  # Change as needeed

DATA_FOLDER = "processed_data/"
INPUT_DATA_NAME = "model_input.parquet"
OUTPUT_DATA_NAME = "transformer_dataset"

# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def pad_left(seq: np.ndarray, max_len: int) -> np.ndarray:
    """Left‑pad / truncate 2‑D array to `max_len` rows, constant‑time."""
    out = np.zeros((max_len, NUM_FEATS), dtype=np.float32)
    seq_len = len(seq)
    if seq_len >= max_len:
        out[:] = seq[-max_len:]
    else:
        out[-seq_len:] = seq
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Data streaming
# ════════════════════════════════════════════════════════════════════════════════

def _iter_play_keys(path: str | Path, limit: int | None) -> Iterable[Tuple[int, int]]:
    """Yield (gameId, playId) pairs, optionally limited to the first *limit*."""
    lazy_keys = pl.scan_parquet(path).select(["gameId", "playId"]).unique()
    if limit:
        lazy_keys = lazy_keys.limit(limit)
    for gid, pid in lazy_keys.collect(streaming=True).iter_rows():
        yield int(gid), int(pid)


def _load_play(path: str | Path, gid: int, pid: int) -> pl.DataFrame:
    """Load a single play’s frames, sorted by frameId (streaming)."""
    return (
        pl.scan_parquet(path)
        .filter((pl.col("gameId") == gid) & (pl.col("playId") == pid))
        .select(["frameId"] + INPUT_FEATURES)
        .sort("frameId")
        .collect(streaming=True)
    )


def example_generator(path: str | Path, max_len: int, limit_plays: int | None):
    """Yield `(X_padded, y_vec)` pairs one at a time — constant memory."""
    play_keys = list(_iter_play_keys(path, limit=limit_plays))
    for gid, pid in tqdm(play_keys, desc="Generating examples", unit="play"):
        play_df = _load_play(path, gid, pid)
        if play_df.height < SEQ_MIN_LEN:
            continue

        arr = play_df.select(INPUT_FEATURES).to_numpy()
        for t in range(arr.shape[0] - 1):
            x_seq = arr[: t + 1]
            y_vec = arr[t + 1]
            yield pad_left(x_seq, max_len), y_vec.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# Main routine
# ════════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("nfl_transformer_dataset (streaming)")
    p.add_argument("--parquet", type=str, default= DATA_FOLDER + INPUT_DATA_NAME, help="Path to model_input.parquet")
    p.add_argument("--save-dir", type=str, default= DATA_FOLDER + OUTPUT_DATA_NAME, help="Output directory")
    p.add_argument("--max-len", type=int, default=MAX_SEQ_LEN, help="Pad/truncation length (frames)")
    p.add_argument("--n-plays", type=int, default=None, help="Sample first N plays (omit ⇒ all plays)")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n ⏳ Streaming examples … this may take a while but uses little RAM. \n ")
    ds = tf.data.Dataset.from_generator(
        lambda: example_generator(args.parquet, args.max_len, args.n_plays),
        output_signature=(
            tf.TensorSpec(shape=(args.max_len, NUM_FEATS), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_FEATS,), dtype=tf.float32),
        ),
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf.get_logger().setLevel("ERROR")
    ds.save(out_dir.as_posix())
    print(f"✅ Finished — dataset written to ‘{out_dir}/’ in TFRecord shards.")


if __name__ == "__main__":
    main()
