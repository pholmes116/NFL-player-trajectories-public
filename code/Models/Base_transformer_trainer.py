#!/usr/bin/env python
"""train_transformer.py
=======================
End-to-end training script for the Transformer frame-prediction model that
was originally developed in a Jupyter notebook.

After fitting, the script automatically plots training/validation loss and
saves the figure alongside the checkpoint weights.

Example
-------
$ python train_transformer.py \
        --data ../../processed_data/transformer_dataset_9 \
        --weights_dir ../weights \
        --batch 256 --lr 3e-4 --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------
# Model import (must be on PYTHONPATH or same dir)
# -----------------------------------------------------------------------------
from Base_transformer import TransformerPredictor  # noqa: E402

# -----------------------------------------------------------------------------
# Default hyper-parameters – keep in sync with the dataset builder & notebook
# -----------------------------------------------------------------------------
NUM_FEATS: int = 46
MAX_LEN: int = 100
D_MODEL: int = 128
N_HEADS: int = 4
N_LAYERS: int = 4
D_FF: int = 512
DROPOUT: float = 0.20

# -----------------------------------------------------------------------------
# Dataset helpers (identical logic to the notebook)
# -----------------------------------------------------------------------------
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2


def _filter_split(split_num: int):
    def _f(meta, x, y):
        return tf.equal(meta[2], split_num)

    return _f


def _drop_meta(meta, x, y):
    return x, y


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------------------------
    # Data & IO
    # ------------------------------------------------------------------
    p.add_argument("--data", type=Path, default=Path("processed_data/transformer_dataset_9"),
                   help="Directory created by tf.data.Dataset.save()")
    p.add_argument("--weights_dir", type=Path, default=Path("weights"),
                   help="Folder for checkpoints, plots & final weights")

    # ------------------------------------------------------------------
    # Training hyper‑parameters
    # ------------------------------------------------------------------
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10_000)
    p.add_argument("--steps", type=int, default=1_000)
    p.add_argument("--val_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)

    # ------------------------------------------------------------------
    # **Model hyper‑parameters** (new)
    # ------------------------------------------------------------------
    p.add_argument("--num_feats", type=int, default=NUM_FEATS, help="Number of coordinate features (x,y pairs)")
    p.add_argument("--max_len", type=int, default=MAX_LEN, help="Sequence length window")
    p.add_argument("--d_model", type=int, default=D_MODEL, help="Transformer hidden size")
    p.add_argument("--n_heads", type=int, default=N_HEADS, help="Multi‑head attention heads")
    p.add_argument("--n_layers", type=int, default=N_LAYERS, help="Encoder layers")
    p.add_argument("--d_ff", type=int, default=D_FF, help="Feed‑forward inner dimension")
    p.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    p.add_argument("--mixed_precision", type= str, default= "mixed_float16" )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--evaluate", action="store_true")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Update globals so they propagate to any import‑time constants
    # ------------------------------------------------------------------
    """global NUM_FEATS, MAX_LEN, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    NUM_FEATS, MAX_LEN, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = (
        args.num_feats, args.max_len, args.d_model, args.n_heads,
        args.n_layers, args.d_ff, args.dropout,
    )"""

    cfg = dict(
        num_feats = args.num_feats,
        max_len   = args.max_len,
        d_model   = args.d_model,
        n_heads   = args.n_heads,       
        n_layers  = args.n_layers,      
        d_ff      = args.d_ff,     
        dropout   = args.dropout,    
    )

    # ------------------------------------------------------------------
    # Hardware & precision policy
    # ------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if args.mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"✅  Using GPU {gpus[0].name} with mixed-precision")
    elif gpus:
        print(f"✅  Using GPU {gpus[0].name} (mixed-precision OFF)")
    else:
        print("⚠️  No GPU found – training will be slow on CPU")

    # ------------------------------------------------------------------
    # Dataset pipeline
    # ------------------------------------------------------------------
    raw_ds = tf.data.Dataset.load(str(args.data))

    train_ds = (raw_ds
                .filter(_filter_split(SPLIT_TRAIN))
                .map(_drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(4096)
                .batch(args.batch)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (raw_ds
              .filter(_filter_split(SPLIT_VAL))
              .map(_drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(args.batch)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (raw_ds
               .filter(_filter_split(SPLIT_TEST))
               .map(_drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(4096)
               .batch(args.batch)
               .prefetch(tf.data.AUTOTUNE))

    # ------------------------------------------------------------------
    # Build / compile model
    # ------------------------------------------------------------------
    predictor = TransformerPredictor(dtype_policy = args.mixed_precision,
                                     **cfg)
    model = predictor.model

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=(args.weights_dir / "epoch_{epoch:03d}-val{val_loss:.6f}.weights.h5").as_posix(),
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=0,
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_steps=args.val_steps,
        validation_data=val_ds,
        callbacks=[ckpt_cb, early_stop_cb],
        verbose=1,
    )

    # ----------------------- evaluate -------------------------------------
    if args.evaluate:
        test_loss, test_mae = model.evaluate(test_ds, verbose=1)
        print(f"\n✅  Test MSE: {test_loss:.5f}   |   Test MAE: {test_mae:.5f}")

    final_weights = args.weights_dir / "best.weights.h5"
    model.save_weights(final_weights.as_posix())

    # safe relative‑path display ------------------------------------------------
    try:
        rel = final_weights.relative_to(Path.cwd())
    except ValueError:
        rel = final_weights
    print(f"💾  Best weights saved to: {rel}")

    (args.weights_dir / "history.json").write_text(json.dumps(history.history))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(history.history["loss"], label="train")
            plt.plot(history.history.get("val_loss", []), label="val")
            plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.tight_layout()
            fig_path = args.weights_dir / "loss_curve.png"
            plt.savefig(fig_path, dpi=120)
            print("📈  Loss curve saved to", fig_path)
        except ImportError:
            print("matplotlib missing – skip plot")


if __name__ == "__main__":
    main()

