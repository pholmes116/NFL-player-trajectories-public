import os

NUM_ENTITIES: int = 23
COORDS: list[str] = ["x", "y", "s", "a", "dir_sin", "dir_cos"]

POSITIONAL_FEATURES = [f"{c}_{i}"
                  for i in range(1, NUM_ENTITIES + 1)
                  for c in COORDS]

from pathlib import Path
import tensorflow as tf
import numpy as np
import statistics as st
    
# Function to filter based on split_id
def filter_split(split_num):
    def _filter(meta, x, y):
        return tf.equal(meta[2], split_num)
    return _filter

def drop_meta(meta, x, y):
    # x: (T, F), y: (138,)
    # Keep only last frame from x (time t) and pass to loss as part of y
    x_t = x[-1]  # shape (138,) — same as y
    return x, tf.concat([x_t, y], axis=-1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_FEATS = 138          # x,y for 23 entities
MAX_LEN  = 100          # same value you used in dataset builder
D_MODEL  = 128          # transformer hidden size
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 512
DROPOUT  = 0.1

# ╔═══════════════════╗
# ║ 2. Positional enc ║  (learnable 1‑D embedding)
# ╚═══════════════════╝
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(max_len, d_model),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_emb

# ╔═══════════════════════════╗
# ║ 3. Padding‑mask function  ║
# ╚═══════════════════════════╝
class PaddingMask(layers.Layer):
    def call(self, x):
        # x:  (B, T, F) — zero‐padded on the left
        pad = tf.reduce_all(tf.equal(x, 0.0), axis=-1)      # → (B, T)
        # reshape to (B, 1, 1, T) for MultiHeadAttention
        return pad[:, tf.newaxis, tf.newaxis, :]

# ╔════════════════════════╗
# ║ 4. Transformer encoder ║
# ╚════════════════════════╝
def transformer_block(d_model, n_heads, d_ff, dropout):
    inputs   = layers.Input(shape=(None, d_model))
    padding  = layers.Input(shape=(1,1,None), dtype=tf.bool)  # mask

    x = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout
    )(inputs, inputs, attention_mask=padding)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)

    y = layers.Dense(d_ff, activation="relu")(x)
    y = layers.Dense(d_model)(y)
    y = layers.Dropout(dropout)(y)
    y = layers.LayerNormalization(epsilon=1e-6)(x + y)

    return keras.Model([inputs, padding], y)

# ╔════════════════════════════════╗
# ║ 5. End‑to‑end prediction model ║
# ╚════════════════════════════════╝
def build_model(
    num_feats=NUM_FEATS,
    max_len=MAX_LEN,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
):
    seq_in  = layers.Input(shape=(max_len, num_feats), name="sequence")   # (B,T,F)

    # Linear projection to d_model
    x = layers.Dense(d_model)(seq_in)

    # Add learnable positional encodings
    x = PositionalEncoding(max_len, d_model)(x)

    # Build padding mask once
    pad_mask = PaddingMask()(seq_in)

    # Stack encoder layers
    for _ in range(n_layers):
        x = transformer_block(d_model, n_heads, d_ff, dropout)([x, pad_mask])

    # We need the hidden state that corresponds to *frame t* (the last row)
    # – that is always index -1 thanks to left padding.
    h_t = layers.Lambda(lambda t: t[:, -1])(x)          # (B, D)

    # Predict only (acceleration, dir_sin, dir_cos) x 22 players + ball = 69 features
    out = layers.Dense(69, name="pred_accel_dir")(h_t)

    return keras.Model(seq_in, out, name="NFL_Frame_Predictor")

# ╔═════════════════════════╗
# ║ 6. Define loss function ║
# ╚═════════════════════════╝

def physics_informed_loss_accel(y_true, y_pred, x_t, s_true, delta_t=0.1):
    """
    Physics-informed loss for NFL tracking.
    
    Inputs:
      - y_true: (batch, 23, 2) true position at t+1
      - y_pred: (batch, 23, 3) predicted (acceleration, dir_sin, dir_cos)
      - x_t:    (batch, 23, 2) starting position at t
      - s_true: (batch, 23) speed at t (unscaled)
    """

    # Unpack predictions
    a_pred = y_pred[:, :, 0]         # acceleration magnitude
    dir_sin_pred = y_pred[:, :, 1]
    dir_cos_pred = y_pred[:, :, 2]

    # Reconstruct initial velocity
    v_x_init = s_true * dir_cos_pred
    v_y_init = s_true * dir_sin_pred

    # Acceleration components
    a_x = a_pred * dir_cos_pred
    a_y = a_pred * dir_sin_pred

    # Update velocity
    v_x = v_x_init + a_x * delta_t
    v_y = v_y_init + a_y * delta_t

    # Predict new position
    x_t1_pred = x_t[:, :, 0] + v_x * delta_t
    y_t1_pred = x_t[:, :, 1] + v_y * delta_t

    position_pred = tf.stack([x_t1_pred, y_t1_pred], axis=-1)

    # Loss: MSE between predicted position and ground-truth
    loss = tf.reduce_mean(tf.square(y_true - position_pred))

    return loss

class PhysicsLossWrapper(tf.keras.losses.Loss):
    def __init__(self, delta_t=0.1):
        super().__init__()
        self.delta_t = delta_t

    def call(self, y_true, y_pred):
        # y_true: (batch, 276)
        # y_pred: (batch, 69)

        # Split x_t and y_target
        x_t_flat     = y_true[:, :138]
        y_target_flat = y_true[:, 138:]

        x_t = tf.reshape(x_t_flat, (-1, 23, 6))
        y_target = tf.reshape(y_target_flat, (-1, 23, 6))

        x_t_pos = x_t[:, :, :2]   # (x, y) at time t
        s_true  = x_t[:, :, 2]    # speed at time t
        next_pos = y_target[:, :, :2]  # true (x, y) at t+1

        y_pred = tf.reshape(y_pred, (-1, 23, 3))
        return physics_informed_loss_accel(next_pos, y_pred, x_t_pos, s_true, self.delta_t)



if __name__ == "__main__":
    
    dataset_path = "processed_data/transformer_dataset"

    # Load dataset without any transformations
    raw_ds = tf.data.Dataset.load(dataset_path)

    # Print dataset structure
    print("Dataset element specification:", raw_ds.element_spec)

    # Examine first 3 examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(raw_ds.take(3)):
        # Each example contains 3 components:
        meta_tensor = example[0]  # Metadata (gameId, playId, split_id, firstFrameId)
        x_tensor = example[1]     # Input sequence (padded frames)
        y_tensor = example[2]     # Target vector
        
        print(f"\nExample {i+1}:")
        print("Metadata tensor:", meta_tensor)
        print(f"Metadata values: {meta_tensor.numpy()}")
        print(f"Input shape: {x_tensor.shape} | dtype: {x_tensor.dtype}")
        print(f"Target shape: {y_tensor.shape} | dtype: {y_tensor.dtype}")
        
        # First 5 elements of first frame's features
        print("First frame features (first 5 values):", x_tensor[0, :5].numpy())
        print("Target values (first 5):", y_tensor[:5].numpy())

    # Examine first 3 examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(train_ds.take(3)):
        # Each example contains 3 components:
        meta_tensor = example[0]  # Metadata (gameId, playId, split_id, firstFrameId)
        x_tensor = example[1]     # Input sequence (padded frames)
        y_tensor = example[2]     # Target vector
        
        print(f"\nExample {i+1}:")
        print("Metadata tensor:", meta_tensor)
        print(f"Metadata values: {meta_tensor.numpy()}")
        print(f"Input shape: {x_tensor.shape} | dtype: {x_tensor.dtype}")
        print(f"Target shape: {y_tensor.shape} | dtype: {y_tensor.dtype}")
        
        # First 5 elements of first frame's features
        print("First frame features (first 5 values):", x_tensor[0, :-5].numpy())
        print("Target values (first 5):", y_tensor[:5].numpy())

    print("Dataset cardinality:",
        tf.data.experimental.cardinality(raw_ds).numpy())   # should now print a number
    
    train_ds = (raw_ds
            .filter(filter_split(0))
            .map(drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(4096)
            .batch(64)
            .prefetch(tf.data.AUTOTUNE))

    val_ds   = (raw_ds
                .filter(filter_split(1))
                .map(drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(64)
                .prefetch(tf.data.AUTOTUNE))

    test_ds  = (raw_ds
                .filter(filter_split(2))
                .map(drop_meta, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(64)
                .prefetch(tf.data.AUTOTUNE))

    # Take one batch from the dataset

    for x_batch, y_true_flat in train_ds.take(1):
        print("x_batch shape:", x_batch.shape)         # (64, 100, 138)
        print("y_true_flat shape:", y_true_flat.shape) # (64, 276)

    for x_batch, y_true_flat in val_ds.take(1):
        print("x_batch shape:", x_batch.shape)         # (64, 100, 138)
        print("y_true_flat shape:", y_true_flat.shape) # (64, 276)

    for x_batch, y_true_flat in test_ds.take(1):
        print("x_batch shape:", x_batch.shape)         # (64, 100, 138)
        print("y_true_flat shape:", y_true_flat.shape) # (64, 276)

    model = build_model()
    model.summary()

    # ╔════════════════════╗
    # ║ 7. Compile & train ║
    # ╚════════════════════╝

    # ── 1)  Make sure we have a place to put checkpoints ─────────────────
    WEIGHT_DIR = Path("../weights_physics")
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=(WEIGHT_DIR /
                "epoch_{epoch:03d}.weights.h5").as_posix(),
        monitor="val_loss",
        save_best_only=False,      # save every epoch → “periodic” archive
        save_weights_only=True,    # just the weights, not optimizer state
        verbose=0,
    )

    # ── 2)  Early-stopping ───────────────────────────────────────────────
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # ── 3)  Compile the model ────────────────────────────────────────────
    LR = 1e-4
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=PhysicsLossWrapper(delta_t=0.1),
        run_eagerly=True
    )

    # ── 4)  Fit – stop early, save weights each epoch ────────────────────
    EPOCHS = 5   # high ceiling; early-stop decides real count
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, ckpt_cb],
        verbose=1,
    )

    # ╔═══════════════════╗
    # ║ 8. Test Evaluation║
    # ╚═══════════════════╝
    test_loss = model.evaluate(test_ds, verbose=1)
    print(f"\n✅  Test Physics-Informed Loss: {test_loss:.5f}")

    # ╔═══════════════╗
    # ║ 9. Evaluation ║
    # ╚═══════════════╝
    # Simple end‑to‑end evaluation on a held‑out batch
    loss_fn = PhysicsLossWrapper(delta_t=0.1)
    X_batch, y_flat = next(iter(val_ds))

    # Extract x_t (last frame's x, y coordinates) from input sequence
    last_frame = X_batch[:, -1, :]
    last_frame_reshaped = tf.reshape(last_frame, (-1, 23, 6))
    x_t = last_frame_reshaped[:, :, :2]

    y_pred = model(X_batch)
    loss = loss_fn(y_flat, y_pred)

    print("Validation Physics Loss (batch):", loss.numpy())
