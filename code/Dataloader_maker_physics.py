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
# COORDS: list[str] = ["x", "y"]
COORDS: list[str] = ["x", "y", "s", "a", "dir_sin", "dir_cos"]
NUM_FEATS: int = NUM_ENTITIES * len(COORDS)          # 23 * 6 = 138
META_LEN: int = 4                                    # gameId, playId, split, firstFrameId, 
SEQ_MIN_LEN: int = 2
MAX_SEQ_LEN: int = 100

POSITIONAL_FEATURES = [f"{c}_{i}"
                  for i in range(1, NUM_ENTITIES + 1)
                  for c in COORDS]

INPUT_FEATURES = POSITIONAL_FEATURES.copy()  # Change as needeed
TARGET_FEATURES = POSITIONAL_FEATURES.copy()  # Change as needeed (don't think this line of code is needed??)

DATA_FOLDER = "processed_data/"
INPUT_DATA_NAME = "model_input_2.parquet"
OUTPUT_DATA_NAME = "transformer_dataset"

# Lookup Table for train test val split
SPLIT_PATH = DATA_FOLDER + "train_test_val.parquet"

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


def example_generator(split_map: dict, path: str | Path, max_len: int, limit_plays: int | None):
    play_keys = list(_iter_play_keys(path, limit=limit_plays))

    for gid, pid in tqdm(play_keys, desc="Generating examples", unit="play"):
        split_id = split_map.get((gid, pid))
        if split_id is None:
            continue

        play_df = _load_play(path, gid, pid)
        if play_df.height < SEQ_MIN_LEN:
            continue

        frame_ids = play_df["frameId"].to_numpy()
        arr = play_df.select(INPUT_FEATURES).to_numpy()

        for t in range(SEQ_MIN_LEN-1, arr.shape[0]-1):
            # Frame t (current) and frame t+1 (next)
            raw_seq = arr[:t+1]
            next_frame = arr[t+1]

            # Calculate first frame index AFTER truncation
            start_idx = max(0, (t + 1) - max_len)
            first_frame_id = int(frame_ids[start_idx])

            # Pad/truncate sequence
            x_seq = pad_left(raw_seq, max_len)

            # Now unpack the NEXT frame:
            # (instead of using all 46 features blindly as target)
            next_frame_reshaped = next_frame.reshape((23, 6))

            next_x = next_frame_reshaped[:, 0] # x
            next_y = next_frame_reshaped[:, 1] # y
            next_s = next_frame_reshaped[:, 2] # s
            next_a = next_frame_reshaped[:, 3] # a
            next_dir_sin = next_frame_reshaped[:, 4] # dir_sin
            next_dir_cos = next_frame_reshaped[:, 5] # dir_cos

            # Stack needed features (example: x,y,s,a,dir_sin,dir_cos)
            y_vec = np.concatenate([next_x, next_y, next_s, next_a, next_dir_sin, next_dir_cos], axis = 0)

            meta_vec = np.array(
                [gid, pid, split_id, first_frame_id],
                dtype=np.int32
            )

            yield meta_vec, x_seq, y_vec.astype(np.float32)



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
    split_tbl = (
        pl.read_parquet(SPLIT_PATH, columns=["gameId", "playId", "split"])
        .unique([ "gameId", "playId" ])               # ◀ safety
        .with_columns(
            pl.when(pl.col("split") == "train").then(0)
                .when(pl.col("split") == "val").then(1)
                .otherwise(2)
                .alias("split_id")
        )
        .select(["gameId", "playId", "split_id"])
        .to_dict(as_series=False)
    )

    split_map = {
        (g, p): s  for g, p, s in zip(split_tbl["gameId"],
                                    split_tbl["playId"],
                                    split_tbl["split_id"])
    }

    args = parse_args()

    print("\n ⏳ Streaming examples … this may take a while but uses little RAM. \n ")
    ds = tf.data.Dataset.from_generator(
        lambda: example_generator(split_map, args.parquet, args.max_len, args.n_plays),
        output_signature=(
            tf.TensorSpec(shape=(META_LEN,),            dtype=tf.int32),   # meta_vec
            tf.TensorSpec(shape=(args.max_len, NUM_FEATS), dtype=tf.float32),
            tf.TensorSpec(shape=(138,),           dtype=tf.float32),
        ),
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf.get_logger().setLevel("ERROR")
    ds.save(out_dir.as_posix())
    print(f"✅ Finished — dataset written to ‘{out_dir}/’ in TFRecord shards.")


if __name__ == "__main__":
    main()
