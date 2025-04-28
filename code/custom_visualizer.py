import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.lines import Line2D

# Default trajectory colors
DEFAULT_OFFENSE_COLOR = 'blue'
DEFAULT_DEFENSE_COLOR = 'green'
DEFAULT_BALL_COLOR = 'orange'


def plot_pitch(ax=None, field_color='#00B140', line_color='white', num_interval=10,
               endzone_left_color='#003594', endzone_right_color='#FF0000'):
    """
    Plots an NFL football field using matplotlib with yard markers from 0 up to 50 and back to 0 in specified increments.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6.7))
        created_fig = True

    # Field background and yard lines
    ax.add_patch(plt.Rectangle((0, 0), 120, 53.3, facecolor=field_color, edgecolor='none'))
    for x in range(5, 121, 5):
        ax.plot([x, x], [0, 53.3], color=line_color, linewidth=1)

    # Yard numbers: 0 -> 50 -> 0
    for x in range(10, 111, num_interval):
        label = x - num_interval if x <= 60 else 110 - x
        ax.text(x, 5, str(label), ha='center', va='center', color=line_color, fontsize=12)
        ax.text(x, 53.3 - 5, str(label), ha='center', va='center', color=line_color, fontsize=12)

    # Endzones
    ax.add_patch(plt.Rectangle((0, 0), 10, 53.3, facecolor=endzone_left_color, edgecolor='none'))
    ax.add_patch(plt.Rectangle((110, 0), 10, 53.3, facecolor=endzone_right_color, edgecolor='none'))

    # Finalize axes
    ax.set_axis_off()
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')

    return (fig, ax) if created_fig else ax


def extract_player_trajectory(sequence, team, player_index=None):
    """
    sequence: (T,46) or (1,T,46) tf.Tensor or array, with features:
      [x1,y1, x2,y2, …, x23,y23]
    team: 'offense', 'defense', or 'ball'
    player_index: 0–10 for offense/defense, ignored for ball
    """
    seq = tf.convert_to_tensor(sequence, tf.float32)
    # if you still have a batch dimension, drop it:
    if seq.ndim == 3 and seq.shape[0] == 1:
        seq = seq[0]

    # map into a global “entity index” 0–22
    if team == 'offense':
        if player_index is None or not (0 <= player_index < 11):
            raise ValueError("…")
        entity = player_index
    elif team == 'defense':
        if player_index is None or not (0 <= player_index < 11):
            raise ValueError("…")
        entity = 11 + player_index
    elif team == 'ball':
        entity = 22
    else:
        raise ValueError("`team` must be 'offense','defense',or 'ball'")

    # x and y are now just the two slots in that block
    x_idx = 2*entity
    y_idx = 2*entity + 1

    x_traj = seq[:, x_idx]
    y_traj = seq[:, y_idx]
    return tf.stack([x_traj, y_traj], axis=1)   # shape (T,2)


def descale_trajectory(coords, x_max=120.0, y_max=53.3):
    """
    Reverse min-max scaling of coordinates on the pitch.
    """
    is_tensor = isinstance(coords, tf.Tensor)
    arr = coords if is_tensor else tf.convert_to_tensor(coords, dtype=tf.float32)
    scale = tf.constant([x_max, y_max], dtype=arr.dtype)
    descaled = arr * scale
    return descaled.numpy() if not is_tensor else descaled


def plot_player_trajectory(traj_coords, ax=None,
                           traj_color=None, traj_label=None,
                           linestyle='-', show_markers=True,
                           marker_size=50,            # ← new
                           **pitch_kwargs):
    """
    Plots a single trajectory given its (T,2) coordinates on the pitch.
    """
    # Default color selection based on label hints
    if traj_color is None:
        traj_color = DEFAULT_OFFENSE_COLOR if (traj_label or '').startswith('offense') else (
                     DEFAULT_DEFENSE_COLOR if (traj_label or '').startswith('defense') else DEFAULT_BALL_COLOR)

    # Setup pitch only if no axis provided
    if ax is None:
        fig, ax = plot_pitch(**pitch_kwargs)
    else:
        fig = ax.figure

    # Convert to numpy and filter zeros
    coords = np.array(traj_coords)
    mask = ~((coords[:, 0] == 0) & (coords[:, 1] == 0))
    coords = coords[mask]

    # Plot trajectory line
    ax.plot(coords[:, 0], coords[:, 1], linestyle=linestyle, color=traj_color, label=traj_label or 'trajectory')
    # Plot start marker
    if show_markers and coords.shape[0] > 0:
        ax.scatter(coords[0, 0], coords[0, 1],
                   marker='o', color=traj_color,
                   s=marker_size)
    # Plot end marker
    if show_markers and coords.shape[0] > 1:
        ax.scatter(coords[-1, 0], coords[-1, 1],
                   marker='x', color=traj_color,
                   s=marker_size)

    ax.legend()
    return (fig, ax) if 'fig' in locals() else axs


def plot_trajectories(sequence,
                      offense=None, defense=None, include_ball=False,
                      ground_truth_seq=None, pred_seq=None,
                      gt_linestyle='-.', pred_linestyle='--',
                      gt_color='lightgreen', pred_color='darkgreen',
                      ax=None,
                      offense_color=None, defense_color=None, ball_color=None,
                      **pitch_kwargs):
    """
    Plots multiple trajectories on the pitch by extracting and plotting each with plot_player_trajectory.
    """
    offense_color = offense_color or DEFAULT_OFFENSE_COLOR
    defense_color = defense_color or DEFAULT_DEFENSE_COLOR
    ball_color = ball_color or DEFAULT_BALL_COLOR

    # Draw pitch once
    if ax is None:
        fig, ax = plot_pitch(**pitch_kwargs)
    else:
        fig = ax.figure

    def _plot(idx, team, color, label_prefix):
        scaled = extract_player_trajectory(sequence, team, idx)
        coords = descale_trajectory(scaled)
        plot_player_trajectory(coords, ax=ax, traj_color=color, 
                               traj_label=f"{label_prefix}-{idx}",
                               linestyle='-', show_markers=True)
        
        # --- ground truth (small markers) ---
        if ground_truth_seq is not None:
            gt_scaled = extract_player_trajectory(ground_truth_seq, team, idx)
            gt_coords = descale_trajectory(gt_scaled)
            plot_player_trajectory(gt_coords, ax=ax,
                                   traj_color=gt_color,
                                   traj_label=f"{label_prefix}-{idx}-gt",
                                   linestyle=gt_linestyle,
                                   show_markers=True,       # ← turn them on
                                   marker_size=20)          # ← much smaller
            
        # --- prediction (small markers) ---
        if pred_seq is not None:
            p_scaled = extract_player_trajectory(pred_seq, team, idx)
            p_coords = descale_trajectory(p_scaled)
            plot_player_trajectory(p_coords, ax=ax,
                                   traj_color=pred_color,
                                   traj_label=f"{label_prefix}-{idx}-pred",
                                   linestyle=pred_linestyle,
                                   show_markers=True,       # ← on
                                   marker_size=20)          # ← small

    # Offense
    if offense in ['all', True] or isinstance(offense, (list, tuple)):
        idxs = range(11) if offense in ['all', True] else offense
        for idx in idxs:
            _plot(idx, 'offense', offense_color, 'offense')

    # Defense
    if defense in ['all', True] or isinstance(defense, (list, tuple)):
        idxs = range(11) if defense in ['all', True] else defense
        for idx in idxs:
            _plot(idx, 'defense', defense_color, 'defense')

    # Ball
    if include_ball:
        scaled = extract_player_trajectory(sequence, 'ball')
        coords = descale_trajectory(scaled)
        plot_player_trajectory(coords, ax=ax, traj_color=ball_color, traj_label='ball')

    # Legend handles
    handles = []
    # — Input trajectories (solid)
    if include_ball:
        lab = 'ball (input)' if (ground_truth_seq is not None or pred_seq is not None) else 'ball'
        handles.append(Line2D([0], [0],
                            color=ball_color, lw=1,
                            linestyle='-',
                            label=lab))
    if offense in ['all', True] or isinstance(offense, (list, tuple)):
        lab = 'offense (input)' if (ground_truth_seq is not None or pred_seq is not None) else 'offense'
        handles.append(Line2D([0], [0],
                            color=offense_color, lw=1,
                            linestyle='-',
                            label=lab))
    if defense in ['all', True] or isinstance(defense, (list, tuple)):
        lab = 'defense (input)' if (ground_truth_seq is not None or pred_seq is not None) else 'defense'
        handles.append(Line2D([0], [0],
                            color=defense_color, lw=1,
                            linestyle='-',
                            label=lab))

    # — Ground truth
    if ground_truth_seq is not None:
        handles.append(Line2D([0], [0],
                            color=gt_color, lw=1,
                            linestyle=gt_linestyle,
                            label='ground truth'))

    # — Prediction
    if pred_seq is not None:
        handles.append(Line2D([0], [0],
                            color=pred_color, lw=1,
                            linestyle=pred_linestyle,
                            label='prediction'))

    # — Start & end markers
    handles.append(Line2D([0], [0],
                        marker='o',
                        color='grey',
                        label='start',
                        linestyle='None'))
    handles.append(Line2D([0], [0],
                        marker='x',
                        color='grey',
                        label='end',
                        linestyle='None'))

    ax.legend(handles=handles, loc='center right')


    return (fig, ax) if 'fig' in locals() else ax