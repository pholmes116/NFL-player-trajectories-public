#!/usr/bin/env python
"""
Create a train/val/test split file for NFL tracking data
-------------------------------------------------------

Reads one or more `tracking_week_*.csv` files, grabs a single
row per (gameId, playId), does a reproducible game-level split,
and writes `train_test_val.parquet` to disk.

Usage
-----
$ python make_split.py \
      --raw-dir        raw_data \
      --out-dir        processed_data \
      --n-weeks        9 \
      --train-ratio    0.7 \
      --val-ratio      0.1 \
      --seed           42
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import polars as pl


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("NFL train/val/test splitter")
    p.add_argument("--raw-dir",   type=str, default="raw_data",
                   help="Folder containing tracking_week_*.csv files")
    p.add_argument("--out-dir",   type=str, default="processed_data",
                   help="Where to write train_test_val.parquet")
    p.add_argument("--n-weeks",   type=int, default=2,
                   help="Number of tracking weeks present")
    p.add_argument("--train-ratio", type=float, default=0.70, help="Train proportion")
    p.add_argument("--val-ratio",   type=float, default=0.10, help="Val proportion")
    p.add_argument("--seed",        type=int,   default=42,   help="RNG seed")
    return p.parse_args()


def build_split_table(raw_dir: Path,
                      n_weeks: int,
                      train_ratio: float,
                      val_ratio: float,
                      seed: int) -> pl.DataFrame:
    """
    Return a Polars DataFrame with columns [gameId, playId, split].
    Exactly one row per (gameId, playId).
    """
    # ---- load only gameId / playId once per play ------------------------------
    tracking_files = [
        raw_dir / f"tracking_week_{i}.csv" for i in range(1, n_weeks + 1)
    ]
    df = pl.concat(
        [pl.read_csv(f, columns=["gameId", "playId"]).unique()
         for f in tracking_files]
    ).unique()                     # one row per play

    # ---- reproducible game-level split ---------------------------------------
    unique_games = df.select("gameId").unique().to_series().to_list()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)

    n_games = len(unique_games)
    n_train = int(n_games * train_ratio)
    n_val   = int(n_games * val_ratio)

    train_gids = set(unique_games[:n_train])
    val_gids   = set(unique_games[n_train:n_train + n_val])
    test_gids  = set(unique_games[n_train + n_val:])

    split_col = (
        pl.when(pl.col("gameId").is_in(train_gids)).then(pl.lit("train"))
          .when(pl.col("gameId").is_in(val_gids)).then(pl.lit("val"))
          .otherwise(pl.lit("test"))
    )

    return df.with_columns(split_col.alias("split"))


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tbl = build_split_table(
        raw_dir=raw_dir,
        n_weeks=args.n_weeks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_path = out_dir / "train_test_val.parquet"
    tbl.write_parquet(out_path)
    print(f"✅ Saved split file → {out_path}  ({len(tbl)} plays)")


if __name__ == "__main__":
    main()
