# transformer_model.py
"""Transformer encoder-only model for sequence-to-one frame prediction
with autoregressive helpers.

Updates (2025‑04‑30)
-------------------
* **predict()** – unified API requested by the user.  Call
  `predict(seq, n_steps)` to get autoregressive rollout; `n_steps=1` gives the
  next frame.  Returns tensors in the exact format produced by the dataset
  pipeline:
    * un‑batched input  →  `(46,)` or `(n_steps, 46)`
    * batched input    →  `(B, 46)` or `(B, n_steps, 46)`
* Input may be rank‑2 `(T,F)` **or** rank‑3 `(B,T,F)`.  Wrapper will add/remove
  the batch dimension automatically, so you no longer need to “predict in
  batches” unless you want the speed‑up.
* Internal refactor – helper `_ensure_batch_dim` centralises the logic.

The underlying Keras graph (layer names, shapes, hyper‑parameters) is **still
identical to the training notebook**, so previously saved `*.weights.h5` files
load without mismatch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# Public hyper-parameters (MUST match the notebook!)
# -----------------------------------------------------------------------------
NUM_FEATS: int = 46      # x,y for 23 entities
MAX_LEN: int = 100       # sequence length used during training
D_MODEL: int = 128
N_HEADS: int = 4
N_LAYERS: int = 4
D_FF: int = 512
DROPOUT: float = 0.10

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(max_len, d_model),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:  # (B,T,D)
        return x + self.pos_emb


class PaddingMask(layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:  # x: (B,T,F)
        pad = tf.reduce_all(tf.equal(x, 0.0), axis=-1)          # (B,T)
        return pad[:, tf.newaxis, tf.newaxis, :]                # (B,1,1,T)


def _transformer_block(d_model: int, n_heads: int, d_ff: int, dropout: float) -> keras.Model:
    inputs = layers.Input(shape=(None, d_model))                   # (B,T,D)
    padding = layers.Input(shape=(1, 1, None), dtype=tf.bool)

    x = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
    )(inputs, inputs, attention_mask=padding)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)

    y = layers.Dense(d_ff, activation="relu")(x)
    y = layers.Dense(d_model)(y)
    y = layers.Dropout(dropout)(y)
    y = layers.LayerNormalization(epsilon=1e-6)(x + y)

    return keras.Model([inputs, padding], y)  # allow Keras to auto‑name


# -----------------------------------------------------------------------------
# Core model (identical to the notebook definition)
# -----------------------------------------------------------------------------

def _build_core_model(
    num_feats: int = NUM_FEATS,
    max_len: int = MAX_LEN,
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    d_ff: int = D_FF,
    dropout: float = DROPOUT,
) -> keras.Model:
    seq_in = layers.Input(shape=(max_len, num_feats), name="sequence")  # (B,T,F)

    x = layers.Dense(d_model, name="proj_xy")(seq_in)
    x = PositionalEncoding(max_len, d_model, name="pos_enc")(x)
    pad_mask = PaddingMask(name="pad_mask")(seq_in)

    for _ in range(n_layers):
        x = _transformer_block(d_model, n_heads, d_ff, dropout)([x, pad_mask])

    h_t = layers.Lambda(lambda t: t[:, -1], name="lambda_last_hidden")(x)
    out = layers.Dense(num_feats, name="pred_xy", dtype="float32")(h_t)

    return keras.Model(seq_in, out, name="NFL_Frame_Predictor")


# -----------------------------------------------------------------------------
# Wrapper class with convenience helpers
# -----------------------------------------------------------------------------
class TransformerPredictor:
    """High‑level wrapper providing single‑step and autoregressive inference.

    >>> model = TransformerPredictor()
    >>> y = model.predict(seq, n_steps=5)   # autoregressive rollout
    """

    def __init__(
        self,
        weights: Union[str, Path, None] = None,
        dtype_policy: str | None = None,
    ) -> None:
        if dtype_policy is not None:
            tf.keras.mixed_precision.set_global_policy(dtype_policy)
        self.model: keras.Model = _build_core_model()
        if weights:
            self.load_weights(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        seq: tf.Tensor | tf.Tensor,
        n_steps: int = 1,
        training: bool = False,
    ) -> tf.Tensor:
        """Return *n_steps* autoregressive predictions.

        The output format mirrors the `y` from the dataset pipeline:
        * If `seq` has rank‑2 `(T,F)` → returns `(n_steps, 46)`.
        * If `seq` has rank‑3 `(B,T,F)` → returns `(B, n_steps, 46)`.
        * When `n_steps == 1`, the `n_steps` axis is squeezed so shapes are
          `(46,)` or `(B, 46)` respectively.
        """
        is_batched, seq = self._ensure_batch_dim(seq)        # → (B,T,F)
        seq = self._pad_left(seq)

        preds: list[tf.Tensor] = []
        for _ in range(n_steps):
            y = self.model(seq, training=training)            # (B,46)
            preds.append(y)
            y_exp = tf.expand_dims(y, 1)                     # (B,1,46)
            seq = tf.concat([seq[:, 1:], y_exp], axis=1)     # slide window

        out = tf.stack(preds, axis=1) if n_steps > 1 else preds[0]
        if not is_batched:
            out = tf.squeeze(out, axis=0)                    # remove batch dim
        if n_steps == 1:
            out = tf.squeeze(out, axis=1) if len(out.shape) == 3 else out
        return out

    def predict_next(self, seq: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Alias for `predict(seq, n_steps=1)`."""
        return self.predict(seq, n_steps=1, training=training)

    def predict_autoregressive(
        self, seq: tf.Tensor, n_steps: int, training: bool = False
    ) -> tf.Tensor:
        """Alias for `predict(seq, n_steps)`."""
        return self.predict(seq, n_steps=n_steps, training=training)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def load_weights(self, path: Union[str, Path]) -> None:
        self.model.load_weights(str(path))

    def save_weights(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_batch_dim(seq: tf.Tensor) -> tuple[bool, tf.Tensor]:
        seq = tf.convert_to_tensor(seq, dtype=tf.float32)
        if seq.shape.rank == 2:          # (T,F) – add batch dim
            seq = seq[tf.newaxis, ...]
            return False, seq
        if seq.shape.rank == 3:          # already (B,T,F)
            return True, seq
        raise ValueError("Input must have shape (T,F) or (B,T,F)")

    @staticmethod
    def _pad_left(seq: tf.Tensor) -> tf.Tensor:
        """Left-pad to MAX_LEN along time dimension (rank-2 or rank-3)."""
        seq = tf.convert_to_tensor(seq, dtype=tf.float32)
        rank = seq.shape.rank
        if rank == 2:               # (T,F) – add batch dim for uniform padding
            seq = seq[tf.newaxis, ...]
        t = tf.shape(seq)[1]
        pad_len = MAX_LEN - t
        seq = seq[:, -MAX_LEN:]
        if pad_len > 0:
            paddings = [[0, 0], [pad_len, 0], [0, 0]]
            seq = tf.pad(seq, paddings)
        if rank == 2:               # remove the synthetic batch dim
            seq = tf.squeeze(seq, axis=0)
        return seq

    # ------------------------------------------------------------------
    # Proxies
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def summary(self):
        return self.model.summary


__all__: Sequence[str] = [
    "NUM_FEATS",
    "MAX_LEN",
    "D_MODEL",
    "N_HEADS",
    "N_LAYERS",
    "D_FF",
    "DROPOUT",
    "TransformerPredictor",
]
