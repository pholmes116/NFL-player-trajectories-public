# nfl_tracking_polars_pipeline.py
"""
End‑to‑end feature‑engineering pipeline rewritten in **Polars** for maximum
speed and low‑memory footprint.

Folder layout (same as original):
└── NFL_data/
    ├── games.csv
    ├── plays.csv
    ├── player_play.csv
    ├── players.csv
    └── tracking_week_1.csv
"""

from __future__ import annotations

import numpy as np
import polars as pl
import os
import json

FOLDER = "raw_data/"
SAVE_FOLDER = "processed_data/"
FILE_NAME = "model_input_2.parquet"
N_TRACKING_FILES = 2

TRACKING_FILES = [FOLDER + f"tracking_week_{i}.csv" for i in range(1, N_TRACKING_FILES + 1)]

os.makedirs(FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility expressions / helpers (unchanged)
# ---------------------------------------------------------------------------

_DEG2RAD = np.pi / 180.0

def _height_to_cm() -> pl.Expr:
    return (
        pl.col("height").str.split("-").list.get(0).cast(pl.Int32()) * 30.48
        + pl.col("height").str.split("-").list.get(1).cast(pl.Int32()) * 2.54
    )

def _weight_to_kg() -> pl.Expr:
    return pl.col("weight") * 0.453_592

def _standard_scale(expr: pl.Expr, mean: float, std: float) -> pl.Expr:
    return (expr - mean) / std

def _minmax_scale(expr: pl.Expr, vmin: float, vmax: float) -> pl.Expr:
    return (expr - vmin) / (vmax - vmin)

def _sin_deg(expr: pl.Expr) -> pl.Expr:
    return (expr * _DEG2RAD).sin()

def _cos_deg(expr: pl.Expr) -> pl.Expr:
    return (expr * _DEG2RAD).cos()

def _normalise_clock() -> pl.Expr:
    mins = pl.col("gameClock").str.split(":").list.get(0).cast(pl.Int32())
    secs = pl.col("gameClock").str.split(":").list.get(1).cast(pl.Int32())
    return ((mins * 60 + secs) / 900).alias("gameClock")

def add_yards_to_score(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("yardlineSide") == pl.col("possessionTeam"))
        .then(100 - pl.col("yardlineNumber"))
        .otherwise(pl.col("yardlineNumber"))
        .alias("yardsToScore")
    )

def _in_posession(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("possessionTeam") == pl.col("club")).then(1).otherwise(0).alias("in_possession")
    )

# ---------------------------------------------------------------------------
# 1. Players table (unchanged)
# ---------------------------------------------------------------------------

players = (
    pl.read_csv(FOLDER + "players.csv")
    .with_columns([
        _height_to_cm().alias("heightMetric"),
        _weight_to_kg().alias("weightMetric"),
        pl.col("position").cast(pl.Categorical).to_physical().alias("position_token"),
    ])
    .select(["nflId", "heightMetric", "weightMetric", "position_token"])
)

stats = players.select([
    pl.mean("heightMetric").alias("hm_mean"),
    pl.std("heightMetric").alias("hm_std"),
    pl.mean("weightMetric").alias("wm_mean"),
    pl.std("weightMetric").alias("wm_std"),
]).to_dict(as_series=False)

players = players.with_columns([
    _standard_scale(pl.col("heightMetric"), stats["hm_mean"][0], stats["hm_std"][0]).alias("heightMetric"),
    _standard_scale(pl.col("weightMetric"), stats["wm_mean"][0], stats["wm_std"][0]).alias("weightMetric"),
])

# ---------------------------------------------------------------------------
# 2. Plays table (+ yardsToScore) – unchanged
# ---------------------------------------------------------------------------

plays_raw = pl.read_csv(
    FOLDER + "plays.csv",
    null_values="NA",        # treat string 'NA' in numeric cols as null
    infer_schema_length=10000,
)

plays = (
    plays_raw.select([
        "gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
        "yardlineSide", "yardlineNumber", "absoluteYardlineNumber", "gameClock", "preSnapHomeScore", "preSnapVisitorScore"
    ])
    .to_dummies(columns=["quarter", "down"], drop_first=True)
    .with_columns([
        _normalise_clock(),
        (pl.col("preSnapHomeScore") / 50).alias("preSnapHomeScore"),
        (pl.col("preSnapVisitorScore") / 50).alias("preSnapVisitorScore"),
    ])
    .pipe(add_yards_to_score)
    .drop(["defensiveTeam", "yardlineSide", "yardlineNumber", "absoluteYardlineNumber"])
)

# ---------------------------------------------------------------------------
# 3. Tracking data (weeks 1–9)
# ---------------------------------------------------------------------------

def _load_tracking_csv(path: str) -> pl.DataFrame:
    """Load a single week of tracking data with consistent schema/options."""
    return pl.read_csv(
        path,
        null_values="NA",            # treat 'NA' as null
        infer_schema_length=10000,
        schema_overrides={"nflId": pl.Int64},  # allow nulls in nflId (football)
    )

tracking_raw = pl.concat([_load_tracking_csv(p) for p in TRACKING_FILES])

tracking = (
    tracking_raw.with_columns([
        _sin_deg(pl.col("o")).alias("o_sin"),
        _cos_deg(pl.col("o")).alias("o_cos"),
        _sin_deg(pl.col("dir")).alias("dir_sin"),
        _cos_deg(pl.col("dir")).alias("dir_cos"),
    ])
    .drop(["displayName", "jerseyNumber", "time", "frameType", "playDirection", "o", "dir"])
)

# --- Normalize x and y based on domain knowledge (field size) ---
tracking = tracking.with_columns([
    (pl.col("x") / 120.0).alias("x"),
    (pl.col("y") / 120.0).alias("y"),
])

# --- Min-max scale s and a ---
SCALE_COLS = ["s", "a"]
scale_mins = tracking.select([pl.min(c).alias(f"{c}_min") for c in SCALE_COLS]).to_dict(as_series=False)
scale_maxs = tracking.select([pl.max(c).alias(f"{c}_max") for c in SCALE_COLS]).to_dict(as_series=False)

for c in SCALE_COLS:
    tracking = tracking.with_columns(_minmax_scale(pl.col(c), scale_mins[f"{c}_min"][0], scale_maxs[f"{c}_max"][0]).alias(c))

# --- Collect and save normalization/standardization info ---
scaling_info = {
    "x": {"min": 0.0, 
          "max": 120.0, 
          "method": "normalize"
    },
    "y": {"min": 0.0, 
          "max": 120.0, 
          "method": "normalize"
    },
    "s": {
        "min": scale_mins["s_min"][0],
        "max": scale_maxs["s_max"][0],
        "method": "minmax"
    },
    "a": {
        "min": scale_mins["a_min"][0],
        "max": scale_maxs["a_max"][0],
        "method": "minmax"
    },
    "heightMetric": {
        "mean": stats["hm_mean"][0],
        "std": stats["hm_std"][0],
        "method": "zscore"
    },
    "weightMetric": {
        "mean": stats["wm_mean"][0],
        "std": stats["wm_std"][0],
        "method": "zscore"
    },
    "preSnapHomeScore": {
        'divisor': 50.0,
        "method": "divisor"
    },
    "preSnapVisitorScore": {
        'divisor': 50.0,
        "method": "divisor"
    },
    "gameClock": {
        "min": 0.0,
        "max": 900.0,
        "method": "minmax"
    }

}
with open(SAVE_FOLDER + "scaling_stats.json", "w") as f:
    json.dump(scaling_info, f, indent=2)

# ---------------------------------------------------------------------------
# 3a. One‑hot encode `event`
# ---------------------------------------------------------------------------

event_df = (
    tracking
    .select(["gameId", "playId", "frameId", "event"])
    # fill nulls with the “Nothing” label
    .with_columns(pl.col("event").fill_null("Nothing"))
    # generate a dummy for every event (including “Nothing”)
    .to_dummies(
        columns=["event"],
        prefix="event",
        prefix_sep="_",
        drop_first=False,      # keep all dummies
    )
    # drop the “event_Nothing” column so that Nothing becomes the implicit/base level
    .drop("event_Nothing")
    .unique()
)

# ---------------------------------------------------------------------------
# 4. Flatten 23‑entity frames
# ---------------------------------------------------------------------------

VALUE_COLS = ["x", "y", "s", "a", "dis", "o_sin", "o_cos", "dir_sin", "dir_cos"]

complete_frames = (
    tracking.group_by(["gameId", "playId", "frameId"]).len()
    .filter(pl.col("len") == 23)
    .select(["gameId", "playId", "frameId"])
)

tracking_complete = (
    tracking.join(                     
        complete_frames,
        on=["gameId", "playId", "frameId"],
        how="inner",
    )
    .join(                             # add possessionTeam once
        plays.select(["gameId", "playId", "possessionTeam"]).unique(),
        on=["gameId", "playId"],
        how="left",
    )
)
tracking_ordered = (
    tracking_complete
    .with_columns([
        # 0 → offense, 1 → defense, 2 → ball  (lower comes first)
        (
            pl.when(pl.col("nflId").is_null())                     # ball
              .then(2)
              .when(pl.col("possessionTeam") == pl.col("club"))    # offense
              .then(0)
              .otherwise(1)                                        # defense
        ).alias("order_key")
    ])
    # secondary sort keeps ordering deterministic and stable
    .sort(["gameId", "playId", "frameId", "order_key", "nflId"], nulls_last=True)
    .with_columns(
        pl.int_range(1, 24).over(["gameId", "playId", "frameId"]).alias("entity_order")
    )
    .drop(["order_key", "possessionTeam"])  # no longer needed
)

agg_exprs = [pl.col(c).sort_by("entity_order").alias(c) for c in VALUE_COLS]
flat_lists = tracking_ordered.group_by(["gameId", "playId", "frameId"]).agg(agg_exprs)

flat_wide = flat_lists.select(["gameId", "playId", "frameId"])
for m in VALUE_COLS:
    flat_wide = flat_wide.join(
        flat_lists.select(
            ["gameId", "playId", "frameId"] + [pl.col(m).list.get(i).alias(f"{m}_{i+1}") for i in range(23)]
        ),
        on=["gameId", "playId", "frameId"],
        how="left",
    )

# ---------------------------------------------------------------------------
# 5. Merge context tables and write
# ---------------------------------------------------------------------------

model_input = (
    flat_wide
    .join(event_df, on=["gameId", "playId", "frameId"], how="left")
    .join(plays, on=["gameId", "playId"], how="left")
)

model_input = (
    model_input
    .drop(['o_sin_23', 'o_cos_23', 'dir_sin_23', 'dir_cos_23'])
    .sort(["gameId", "playId", "frameId"])
)


model_input.write_parquet(SAVE_FOLDER + FILE_NAME)
print(f"Pipeline finished — wrote model_input.parquet ({len(model_input)} rows across {N_TRACKING_FILES} weeks)")