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

MAX_TORQUE = 5e-5


DT = 0.1  
THETA_THRESHOLD = 15
OMEGA_THRESHOLD = 5
THETA_TOL_STABILITY = 5
OMEGA_TOL_STABILITY = 5



@partial(jax.jit,static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def network_output_to_QR(theta, nx, nu, decomposition_type='diagonal', 
                horizon=10, qmin=1e-1, rmin=1e-1, qmax=1e5, rmax=1e5, 
                eps=1e-3):
    """
    Convert the output of the neural network (theta) into Q and R matrices for MPC.
    theta: output of the neural network, shape (nx + nu,)
    nx: number of state variables
    nu: number of control inputs
    Returns:
        Q: state cost matrix, shape (nx, nx)
        R: control cost matrix, shape (nu, nu)
    """

    # Need to consider how to handle case where const matrix is used over the horizon


    # Split theta into Q and R part
    if decomposition_type == 'diagonal':

        # Figure out better way to do this
        # Maybe check how acmpc guys did it
        n_q = nx * horizon
        n_r = nu * horizon

        Q_flat = theta[:n_q]
        R_flat = theta[n_q:n_q + n_r]

        Q_flat = qmin + 0.5 * (qmax - qmin) * (Q_flat + 1)
        R_flat = rmin + 0.5 * (rmax - rmin) * (R_flat + 1)

        # Create sequence of Q and R matrices for each timestep in the horizon

        if horizon == 1:

            # Const matrices over time
            Q = jnp.diag(Q_flat)
            R = jnp.diag(R_flat)

        else:

            Q_diag = Q_flat.reshape(horizon, nx)
            R_diag = R_flat.reshape(horizon, nu)

            Q = jax.vmap(jnp.diag)(Q_diag)
            R = jax.vmap(jnp.diag)(R_diag)

    elif decomposition_type == 'full':
        # Decomposition is of type Q = AA^T, R = BB^T

        n_q = horizon * nx * nx
        n_r = horizon * nu * nu

        A = theta[:n_q]
        B = theta[n_q:n_q + n_r]

        if horizon == 1:
            Q = A @ A.T
            R = B @ B.T

        else:

            A  = theta[:n_q].reshape((horizon, nx, nx))
            B  = theta[n_q:n_q + n_r].reshape((horizon, nu, nu))

            # No bounding for full decomposition (yet)

            Q = jax.vmap(lambda A: A @ A.T)(A)
            R = jax.vmap(lambda B: B @ B.T)(B)

    elif decomposition_type == 'cholesky':

        # Decomposition is of type Q = LL^T, R = MM^T
        n_q = (nx * (nx + 1) // 2) * horizon
        n_r = (nu * (nu + 1) // 2) * horizon

        L_flat = theta[:n_q].reshape(horizon,-1)
        M_flat = theta[n_q:n_q + n_r].reshape(horizon,-1)

        # No bounding for cholesky decomposition (yet)

        row_indices_Q, col_indices_Q = jnp.tril_indices(nx)
        row_indices_R, col_indices_R = jnp.tril_indices(nu)

        if horizon == 1:

            L = jnp.zeros((nx,nx))
            M = jnp.zeros((nu,nu))

            L = L.at[row_indices_Q, col_indices_Q].set(L_flat)
            M = M.at[row_indices_R, col_indices_R].set(M_flat)

            Q = L @ L.T
            R = M @ M.T

        else:

            L = jnp.zeros((horizon, nx, nx))
            M = jnp.zeros((horizon, nu, nu))

            L = L.at[:, row_indices_Q, col_indices_Q].set(L_flat)
            M = M.at[:, row_indices_R, col_indices_R].set(M_flat)

            Q = jax.vmap(lambda L: L @ L.T)(L)
            R = jax.vmap(lambda M: M @ M.T)(M)

    R = R / (MAX_TORQUE ** 2)  # Scale R by max torque squared

    return Q, R



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


def make_dummy_expert(action_dim):

    def dummy_expert(obs):
        return jnp.zeros((action_dim,), dtype=jnp.float64)

    return dummy_expert

def make_dummy_controller(state_dim, action_dim):
    def dummy_controller(obs, goal, nom_traj, nom_cntrl):
        return jnp.zeros((action_dim,), dtype=jnp.float64), nom_traj, nom_cntrl
    return dummy_controller

# def save_data_to_df(trajectories, filepath, labels, dt)


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


def plot_metrics_bar(metrics_df, title=None, filename=None, figsize=(8, 5)):
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
            Path('figures').mkdir(exist_ok=True)
            suffix = col.replace(' ', '_').replace('(%)', 'pct').replace('(', '').replace(')', '').replace('/', '_')
            plt.savefig(f"figures/{filename}_{suffix}.png", dpi=150, bbox_inches='tight')
        
        plt.show()


def plot_metrics_comparison(metrics_list, agent_labels, title=None, filename=None, figsize=(10, 5)):
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
            Path('figures').mkdir(exist_ok=True)
            suffix = col.replace(' ', '_').replace('(%)', 'pct').replace('(', '').replace(')', '').replace('/', '_')
            plt.savefig(f"figures/{filename}_{suffix}.png", dpi=150, bbox_inches='tight')
        
        plt.show()


def print_metrics(metrics_df, title=None):
    """Print formatted metrics table."""
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print('='*70)
    print(metrics_df.to_string(index=False, float_format='{:.2f}'.format))
    print()