"""

Code to compute metrics and plots for evaluation

"""
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