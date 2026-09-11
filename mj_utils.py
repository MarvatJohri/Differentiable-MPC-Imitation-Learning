import os
import sys

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import pandas as pd

# from simulation_env_simpler import SpacecraftEnv

from typing import Callable, Tuple, List, Dict
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sbx import PPO
from stable_baselines3.common.vec_env import VecNormalize

MAX_TORQUE = 5e-5


DT = 0.1  
THETA_THRESHOLD = 15
OMEGA_THRESHOLD = 5
THETA_TOL_STABILITY = 5
OMEGA_TOL_STABILITY = 5






def sample_state(batch_size, key, omega_min=0.0, omega_max=0.0):

    """
    Samples a batch of random states for the spacecraft environment.

    Args:
        batch_size: Number of states to sample.
        key: JAX key.
        omega_min: Minimum angular velocity (rad/s).
        omega_max: Maximum angular velocity (rad/s).
    """

    # Sample omega uniformly
    key, subkey = jax.random.split(key)
    omega = jax.random.uniform(subkey, 
                               shape=(batch_size, 3), 
                               minval=omega_min, 
                               maxval=omega_max, 
                               dtype=jnp.float64)

    # Sample quaternion
    key, subkey = jax.random.split(key)
    q = jax.random.normal(subkey, shape=(batch_size, 4), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    return jnp.concatenate([q, omega], axis=-1)


def get_expert_policy(expert_policy: PPO, vec_env: VecNormalize):


    # Extract the expert policy's actor module and parameters
    actor_module = expert_policy.policy.actor
    actor_params = expert_policy.policy.actor_state.params

    # Extract the vector normalization statistics
    obs_mean = jnp.array(vec_env.obs_rms.mean, dtype=jnp.float32)
    obs_var = jnp.array(vec_env.obs_rms.var, dtype=jnp.float32)
    obs_count = vec_env.obs_rms.count
    obs_eps = vec_env.epsilon
    obs_clip = vec_env.clip_obs

    def expert_policy_fn(obs: jnp.ndarray) -> jnp.ndarray:
        # Cast down to float32 for the actor module
        obs = obs.astype(jnp.float32)

        # Normalize the observation
        normalized_obs = (obs - obs_mean) / jnp.sqrt(obs_var + obs_eps)
        normalized_obs = jnp.clip(normalized_obs, -obs_clip, obs_clip)

        # Pass through the actor module to get the action
        dist = actor_module.apply(actor_params, normalized_obs[None, :])
        action = dist.mode()[0]  # Get the mode of the distribution and remove the batch dimension
        action = jnp.clip(action, -1.0, 1.0)  # Ensure action is within [-1, 1]
        return action.astype(jnp.float64)  # Cast back to float64 for consistency


    return expert_policy_fn


def make_dummy_expert(action_dim):

    def dummy_expert(obs):
        return jnp.zeros((action_dim,), dtype=jnp.float64)

    return dummy_expert

def make_dummy_controller(state_dim, action_dim):
    def dummy_controller(obs, goal, nom_traj, nom_cntrl):
        return jnp.zeros((action_dim,), dtype=jnp.float64), nom_traj, nom_cntrl
    return dummy_controller






def compute_metrics(trajectories: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
                    dt=DT, 
                    angle_threshold=THETA_THRESHOLD,
                    omega_threshold=OMEGA_THRESHOLD,
                    angle_tol_stability=THETA_TOL_STABILITY,
                    omega_tol_stability=OMEGA_TOL_STABILITY,
                    tail_length=100,
                    label='Default', 
                    metadata=None):


    # Create directory if it doesn't exist
    # os.makedirs(save_dir, exist_ok=True)

    # obs/actions are of form (n_trajs, T, dim)
    observations, _, __, ___, ____ = trajectories

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

    # 3. Condition: Is the distance from the *average* final value within the neighborhood?
    cond = ((jnp.abs(angle_errs - final_angle_errs[:, None]) < angle_tol_stability) &
            (jnp.abs(omega_err_norm - final_omega_errs[:, None]) < omega_tol_stability))

    # 4. Backward accumulation to find where it *stays* within the neighborhood
    mask = jnp.logical_and.accumulate(cond[:, ::-1], axis=1)[:, ::-1]

    # 5. Calculate indices and stability checks
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
    """
    Compute metrics for multiple trajectory sets.
    
    Parameters:
    -----------
    trajectories_list : list of tuples
        List of outputs from generate_n_trajectories
    labels : list of str
        Label for each trajectory set
    **kwargs : 
        Passed to compute_metrics (dt, thresholds, etc.)
    
    Returns:
    --------
    metrics_df : pd.DataFrame
    """
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
    """
    Compare metrics across multiple agents.
    
    Parameters:
    -----------
    metrics_list : list of pd.DataFrame
        List of metrics DataFrames (from compute_metrics or compute_metrics_multi)
    agent_labels : list of str
        Label for each agent
    """
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