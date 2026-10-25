from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.resolve()

import sys
sys.path.append(str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from configs import SpacecraftEnvConfig, ExpConfig



env_config = SpacecraftEnvConfig()


DT = env_config.dt
THETA_THRESHOLD = np.rad2deg(env_config.theta_threshold)
OMEGA_THRESHOLD = np.rad2deg(env_config.omega_threshold)
THETA_STABILITY_TOL = np.rad2deg(env_config.theta_stability_tol)
OMEGA_STABILITY_TOL = np.rad2deg(env_config.omega_stability_tol)



def compute_metrics(trajectories, 
                    dt=DT, 
                    angle_threshold=THETA_THRESHOLD,
                    omega_threshold=OMEGA_THRESHOLD,
                    angle_tol_stability=THETA_STABILITY_TOL,
                    omega_tol_stability=OMEGA_STABILITY_TOL,
                    tail_length=100,
                    label='Default', 
                    metadata=None):


    # Create directory if it doesn't exist
    # os.makedirs(save_dir, exist_ok=True)

    # obs/actions are of form (n_trajs, T, dim)
    # observations, _, __, ___, ____ = trajectories
    observations = trajectories[0]

    n_trajs = observations.shape[0]
    T = observations.shape[1]
    nx = observations.shape[2]

    q_err = observations[:, :, :4]  
    w_err = observations[:, :, 4:7]  

    q_err_scalar = q_err[:, :, 0]
    angle_errs = 2 * jnp.arccos(jnp.clip(jnp.abs(q_err_scalar), 0, 1)) * 180 / jnp.pi # Converted to degrees

    omega_err_norm = jnp.linalg.norm(w_err, axis=-1) * 180 / jnp.pi


    final_angle_errs = jnp.mean(angle_errs[:, -tail_length:], axis=1)
    final_omega_errs = jnp.mean(omega_err_norm[:, -tail_length:], axis=1)


    # Apply stability condition
    # Tail should have angle maintained b/w angle_tol_stability and omega_tol_stability

    # Condition: Is the distance from the *average* final value within the neighborhood?
    cond = ((jnp.abs(angle_errs - final_angle_errs[:, None]) < angle_tol_stability) &
            (jnp.abs(omega_err_norm - final_omega_errs[:, None]) < omega_tol_stability))

    # Backward accumulation to find where it *stays* within the neighborhood
    mask = jnp.logical_and.accumulate(cond[:, ::-1], axis=1)[:, ::-1]

    # Calculate indices and stability checks
    stability_idx = jnp.argmax(mask, axis=1)
    ever_stable = jnp.sum(mask, axis=1) >= tail_length
    
    slew_time = jnp.where(ever_stable, stability_idx * dt, jnp.nan)
    
    # Metrics
    stable_trajs = ~jnp.isnan(slew_time)
    successful = stable_trajs & (final_angle_errs < angle_threshold) & (final_omega_errs < omega_threshold)
    
    metrics_df = pd.DataFrame([{
        'Group': label,
        'Success Rate (%)': float(jnp.mean(successful) * 100),
        'Stable Rate (%)': float(jnp.mean(stable_trajs) * 100),
        'Mean Angle Error (deg)': float(jnp.mean(final_angle_errs)),
        'Mean Omega Error (deg/s)': float(jnp.mean(final_omega_errs)),
        'Mean Slew Time (s)': float(jnp.nanmean(slew_time)),
        'N Trajectories': int(n_trajs),
    }])
    
    return metrics_df


def compute_metrics_multi(trajectories_list, labels, **kwargs)-> pd.DataFrame:
    
    dfs = [compute_metrics(traj, label=lbl, **kwargs) 
           for traj, lbl in zip(trajectories_list, labels)]
    return pd.concat(dfs, ignore_index=True)


def plot_metrics_bar(metrics_df, title=None, filename=None, file_path=None, figsize=(8, 5)):
    """Create bar plots for all metrics."""
    
    metrics_config = [
        ('Success Rate (%)', (0, 100), 'steelblue'),
        ('Stable Rate (%)', (0, 100), 'seagreen'),
        ('Mean Angle Error (deg)', None, 'coral'),
        ('Mean Omega Error (deg/s)', None, 'mediumpurple'),
        ('Mean Slew Time (s)', None, 'goldenrod'),
    ]
    
    n_groups = len(metrics_df)
    
    for col, ylim, color in metrics_config:
        fig, ax = plt.subplots(figsize=figsize)
        
        if n_groups > 1:
            sns.barplot(data=metrics_df, x='Group', y=col, ax=ax, color=color)
        else:
            ax.bar(0, metrics_df[col].iloc[0], color=color, width=0.4)
            ax.set_xticks([0])
            ax.set_xticklabels([metrics_df['Group'].iloc[0]])
        
        # Value labels on bars
        for i, val in enumerate(metrics_df[col]):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}', xy=(i, val), ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(col)
        ax.set_xlabel('')
        ax.set_ylim(ylim if ylim else (0, None))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        sns.despine(ax=ax)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if filename:
            print("./Saving figure...")
            # Make figures directory inside filepath
            figure_dir = Path(file_path) / 'figures' if file_path else Path('figures')
            figure_dir.mkdir(exist_ok=True)
            suffix = col.replace(' ', '_').replace('(%)', 'pct').replace('(', '').replace(')', '').replace('/', '_')
            figure_path = figure_dir / f"{filename}_{suffix}.png"
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        
        plt.show()



def plot_metrics_comparison(metrics_list, agent_labels, title=None, filename=None,file_path=None, figsize=(10, 5)):
   
    combined = pd.concat([
        df.assign(Agent=label) for df, label in zip(metrics_list, agent_labels)
    ], ignore_index=True)
    
    metrics_config = [
        ('Success Rate (%)', (0, 100)),
        ('Stable Rate (%)', (0, 100)),
        ('Mean Angle Error (deg)', None),
        ('Mean Omega Error (deg/s)', None),
        ('Mean Slew Time (s)', None),
    ]
    
    n_groups = combined['Group'].nunique()
    
    for col, ylim in metrics_config:
        fig, ax = plt.subplots(figsize=figsize)
        
        if n_groups > 1:
            sns.barplot(data=combined, x='Group', y=col, hue='Agent', ax=ax)
            ax.legend(title='')
        else:
            sns.barplot(data=combined, x='Agent', y=col, ax=ax, palette='deep')
        
        ax.set_ylabel(col)
        ax.set_xlabel('')
        ax.set_ylim(ylim if ylim else (0, None))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        sns.despine(ax=ax)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if filename:
            figure_dir = Path(file_path) / 'figures' if file_path else Path('figures')
            figure_dir.mkdir(exist_ok=True)
            suffix = col.replace(' ', '_').replace('(%)', 'pct').replace('(', '').replace(')', '').replace('/', '_')
            plt.savefig(figure_dir / f"{filename}_{suffix}.png", dpi=150, bbox_inches='tight')
        
        plt.show()


def print_metrics(metrics_df, title=None):
    """Print formatted metrics table."""
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print('='*70)
    print(metrics_df.to_string(index=False, float_format='{:.2f}'.format))
    print()


def extract_trajectory_data(
    trajectory, dt=DT, max_torque=env_config.max_torque,
    action_low=-1.0, action_high=1.0, *, action_units="normalized",
):

    # Actions may either be normalized, or in terms of torque
    # depending on whether PPO or acmpc/mpc type policy was used
    # Note that using action_units and account later for plots
   
    if action_units not in {"normalized", "torque"}:
        raise ValueError("action_units must be 'normalized' or 'torque'")
    
    obs = np.asarray(trajectory.observations)
    single = obs.ndim == 2
    
    if single:
        obs = obs[None]
    if obs.ndim != 3 or obs.shape[-1] != 7 or obs.shape[0] == 0 or obs.shape[1] < 2:
        raise ValueError("Expected observations of form (episodes, T+1, 7) or (T+1, 7)")

    # Extract stuff
    episodes, samples, _ = obs.shape
    steps = samples - 1
    actions = np.asarray(trajectory.actions)
    rewards = np.asarray(trajectory.rewards)

    if single:
        actions, rewards = actions[None], rewards[None]
    if actions.shape == (episodes, samples, 3):
        actions = actions[:, :-1]
    if actions.shape != (episodes, steps, 3) or rewards.shape != (episodes, steps):
        raise ValueError("Expected T actions and rewards for T+1 observations")

    # To ensure actions are clipped while plotting
    # Actions are only clipped inside the environment
    low = np.broadcast_to(np.asarray(action_low, dtype=float), (3,)) * max_torque
    high = np.broadcast_to(np.asarray(action_high, dtype=float), (3,)) * max_torque

    if not np.all(np.isfinite(low) & np.isfinite(high) & (low < high)):
        raise ValueError("Action limits must be finite with low < high")
    
    requested = actions * max_torque if action_units == "normalized" else actions
    runs = {
        "time": np.arange(samples) * dt,
        "action_time": np.arange(steps) * dt,
        "reward_time": np.arange(1, samples) * dt,
        "dt": dt,
        "max_torque": max_torque,
        "angle_error": np.rad2deg(2 * np.arccos(np.clip(np.abs(obs[..., 0]), 0, 1))),
        "omega_error": np.rad2deg(np.linalg.norm(obs[..., 4:7], axis=-1)),
        "requested_torque": requested,
        "applied_torque": np.clip(requested, low, high),
        "torque_low": low,
        "torque_high": high,
        "rewards": rewards,
        "returns": rewards.sum(axis=1),
    }
    if (trajectory.Q is None) != (trajectory.R is None):
        raise ValueError("Q and R must both be present or both be None")

    # Just make sure everything in Q/R matrices is fine
    for name, matrix, dimension in (("Q", trajectory.Q, 6), ("R", trajectory.R, 3)):
        if matrix is None:
            continue
        matrix = np.asarray(matrix)
        if single:
            matrix = matrix[None]
        if matrix.ndim != 4:
            raise ValueError(f"{name} must have episode/time axes, and two matrix axes")
        if matrix.shape[0] != episodes:
            raise ValueError(f"{name} must have the same number of episodes as observations")
        if matrix.shape[1] not in (steps, samples):
            raise ValueError(f"{name} must have T or T+1 recorded samples")
        if matrix.shape[-2:] != (dimension, dimension):
            raise ValueError(f"{name} matrices must have shape ({dimension}, {dimension})")
        if matrix.ndim == 5 and matrix.shape[2] == 0:
            raise ValueError(f"{name} prediction horizon must not be empty")
        runs[f"{name}_diag"] = np.diagonal(matrix, axis1=-2, axis2=-1)


    return runs


def _finish_plot(fig, save_path):
    # Save plots
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig



def _plot_trace(ax, time, values, label, episode, percentiles):
    # Plot stuff with percentiles
    if episode is not None:
        if not 0 <= episode < values.shape[0]:
            raise ValueError(f"Invalid episode index {episode} for {label}")
        ax.plot(time, values[episode], label=label)
        return
    if len(percentiles) != 2 or not 0 <= percentiles[0] <= percentiles[1] <= 100:
        raise ValueError("percentiles must satisfy 0 <= low <= high <= 100")
    low, high = np.percentile(values, percentiles, axis=0)
    line, = ax.plot(time, np.median(values, axis=0), label=label)
    ax.fill_between(time, low, high, color=line.get_color(), alpha=0.15)


def plot_error_trajectories(
    runs, episode=None, percentiles=(10, 90),
    angle_threshold_deg=THETA_THRESHOLD, omega_threshold_deg=OMEGA_THRESHOLD,
    save_path=None,
):
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, data in runs.items():
        for ax, key in zip(axes, ("angle_error", "omega_error")):
            _plot_trace(ax, data["time"], data[key], label, episode, percentiles)
    for ax, threshold in zip(axes, (angle_threshold_deg, omega_threshold_deg)):
        ax.axhline(threshold, color="black", linestyle="--", label="Success threshold")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Attitude error (deg)")
    axes[1].set_ylabel("Angular-speed error (deg/s)")
    axes[1].set_xlabel("Time (s)")
    return _finish_plot(fig, save_path)


def plot_control_trajectories(
    runs, *, episode=0, requested=False, percentiles=(10, 90), save_path=None,
):
  
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    key = "requested_torque" if requested else "applied_torque"
    for label, data in runs.items():
        for i, ax in enumerate(axes):
            _plot_trace(ax, data["action_time"], data[key][..., i], label, episode, percentiles)
            ax.axhline(data["torque_low"][i], color="black", linestyle=":", alpha=0.4)
            ax.axhline(data["torque_high"][i], color="black", linestyle=":", alpha=0.4)
    for axis_name, ax in zip(("x", "y", "z"), axes):
        ax.set_ylabel(f"Torque {axis_name} (N m)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    return _finish_plot(fig, save_path)


def compute_control_metrics(runs):
    
    frames = []
    for label, data in runs.items():
        torque = data["applied_torque"]
        requested = data["requested_torque"]
        clipped = np.any((requested < data["torque_low"])
                         | (requested > data["torque_high"]), axis=-1)
        frames.append(pd.DataFrame({
            "Agent": label,
            "Episode": np.arange(torque.shape[0]),
            "Fixed-horizon return": data["returns"],
            "Integrated squared torque": np.sum(torque**2, axis=(1, 2)) * data["dt"],
            "Total torque variation": np.linalg.norm(np.diff(torque, axis=1), axis=-1).sum(axis=1),
            "Clipping fraction": clipped.mean(axis=1),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_rewards(runs, *, episode=None, percentiles=(10, 90), save_path=None):


    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    samples, labels = [], []
    for label, data in runs.items():
        for ax, values in zip(axes[:2], (data["rewards"], np.cumsum(data["rewards"], axis=1))):
            _plot_trace(ax, data["reward_time"], values, label, episode, percentiles)
        samples.append(data["returns"])
        labels.append(label)
    axes[0].set_ylabel("Raw reward")
    axes[1].set_ylabel("Cumulative raw reward")
    for ax in axes[:2]:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[2].boxplot(samples)
    axes[2].set_xticks(np.arange(1, len(labels) + 1), labels)
    axes[2].set_ylabel("Fixed-horizon raw return")
    axes[2].grid(axis="y", alpha=0.25)
    return _finish_plot(fig, save_path)


def plot_episode_returns(runs, *, save_path=None):

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, data in runs.items():
        ax.plot(np.arange(len(data["returns"])), data["returns"], marker=".", label=label)
    ax.set_xlabel("Evaluation rollout index")
    ax.set_ylabel("Fixed-horizon raw return")
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish_plot(fig, save_path)


def plot_cost_weights(
    runs, episode=0, timestep=0,
    normalized_control_units=True, log_scale=True, save_path=None,
):
    
 
    selected = {label: data for label, data in runs.items() if "Q_diag" in data}
    if not selected:
        raise ValueError("No Q/R recordings; generate trajectories with record_qr=True")
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.5 * len(selected)), squeeze=False)
    for row, (label, data) in enumerate(selected.items()):
        if not 0 <= episode < data["Q_diag"].shape[0]:
            raise ValueError(f"Invalid episode index {episode} for {label}")
        Q, R = data["Q_diag"][episode], data["R_diag"][episode]


        xq, xr = data["time"][:len(Q)], data["time"][:len(R)]
        xlabel = "Time (s)"

        
        if normalized_control_units:
            R = R * data["max_torque"]**2


        groups = ((Q[..., :3], xq, "Q attitude"),
                  (Q[..., 3:6], xq, "Q angular velocity"),
                  (R, xr, "R normalized control" if normalized_control_units else "R physical torque"))
        
        for ax, (values, x, title) in zip(axes[row], groups):
            for i, name in enumerate(("x", "y", "z")):
                ax.plot(x, values[:, i], label=name, marker="." if len(x) == 1 else None)
            if log_scale:
                if not np.all(np.isfinite(values) & (values > 0)):
                    raise ValueError("Weights must be finite and positive")
                ax.set_yscale("log")
            ax.set_title(f"{label}: {title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Weight")
            ax.grid(alpha=0.25)
            ax.legend()

    return _finish_plot(fig, save_path)


def plot_training_metrics(
    histories, steps_per_update, smoothing_window=1, start_steps=0, save_path=None,
):
   

    if not isinstance(smoothing_window, (int, np.integer)) or smoothing_window < 1:
        raise ValueError("smoothing_window must be a positive integer")
   
    panels = (("episode_return", "Mean completed-episode raw return"),
              ("mean_reward", "Mean training reward"), ("value_loss", "Value loss"),
              ("pg_loss", "Policy loss"), ("approx_kl", "Approximate KL"),
              ("action_std", "Action standard deviation"), ("entropy", "Gaussian entropy"),
              ("reward_scale", "Reward normalization scale"))
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for label, history in histories.items():
            if key not in history:
                continue
            values = np.asarray(history[key], dtype=float)
            if values.ndim != 1:
                raise ValueError(f"{label}/{key}: expected one scalar per update")
            spacing = steps_per_update[label] if hasattr(steps_per_update, "keys") else steps_per_update
            offset = start_steps[label] if hasattr(start_steps, "keys") else start_steps
            if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(offset) or offset < 0:
                raise ValueError("steps_per_update must be positive and start_steps nonnegative")
            steps = offset + (np.arange(len(values)) + 1) * spacing
            smoothed = pd.Series(values).rolling(smoothing_window, min_periods=1).mean().to_numpy(copy=True)
            smoothed[~np.isfinite(values)] = np.nan
            ax.plot(steps, smoothed, label=label)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend()
    return _finish_plot(fig, save_path)
