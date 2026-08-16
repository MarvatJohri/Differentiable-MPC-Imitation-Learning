"""
Testing script for SpacecraftEnv with PPO using SBX.
"""

import os
import json
import random
import sys
from typing import Dict
import numpy as np
from datetime import datetime
from pathlib import Path

from sbx import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from matplotlib import pyplot as plt



# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BASE_SAVE_PATH = str(HERE / "results")
BASE_LOG_PATH = str(HERE / "logs")

URANUS_MPC_PATH = str((ROOT / "uranus-mpc").resolve())


sys.path.insert(0, str((ROOT / "uranus-mpc").resolve()))
sys.path.insert(0, str((ROOT / "uranus-mpc" / "utils").resolve()))

# Your environment imports - adjust as needed
from dynamics.base_dynamics import Dynamics
from dynamics.spacecraft_dynamics import SpacecraftDynamics
from dynamics.orbit_dynamics import OrbitDynamics
from simulation_env_simpler import SpacecraftEnv
from dynamics.planetary_params import Earth, Uranus
from utils.propagate import TrajectoryGenerator

from utils.learning import load_model


# Experiment identification
EXPERIMENT_NAME = "spacecraft_ppo_v1_torque_only"
EXPERIMENT_NOTES = "Initial PPO training on Earth orbit"

# Environment
PLANET = "earth"                 # "earth" or "uranus"
DT = 0.1                         # Simulation timestep
N_ORBIT = 5000                   # Orbit trajectory length
DYN_NOISE_STD = 1e-6             # Dynamics noise
MAX_EPISODE_STEPS = 1500          # Max steps per episode

# State/action limits
STATE_LIMITS = [[-1, 1]] * 4 + [[-2, 2]] * 3  # [quat, omega]
CONTROL_LIMIT_SCALE = 1        # Scales [-1, 1] control limits

# Mag field stuff
NORMALIZE_MAG = True                # Whether to normalize magnetic field vector in observations
MAG_FIELD_HORIZON = 0              # Number of future magnetic field vectors to include in observation
MAG_FIELD_SCALE = 1e-5
CONST_MAG_FIELD = False              # Whether to use a constant magnetic field throughout the episode

# Reward shaping
THETA_THRESHOLD = np.deg2rad(15.0)  # Convert to radians
OMEGA_THRESHOLD = np.deg2rad(5.0)                 # Angular velocity tolerance (rad/s)
THETA_PROXIMITY_TOL_DEG = 50.0        # Proximity tolerance for reward shaping (degrees)
THETA_PROXIMITY_TOL = np.deg2rad(THETA_PROXIMITY_TOL_DEG)         # Proximity tolerance for reward shaping (radians)
QUATERNION_PENALTY = 1.0        # Penalty weight for quaternion error
OMEGA_PENALTY = 0.1               # Penalty weight for omega error
ACTION_PENALTY = 0.01              # Penalty weight for action magnitude
PROXIMITY_PENALTY = 5.0          # Reward for being within proximity tolerance
GOAL_REWARD = 50.0              # Bonus for reaching goal



ACTION_PARALLEL_PENALTY = 0.1        # Penalty weight for magnetic dipole parallel to magnetic field
ACTION_PERPENDICULAR_PENALTY = 0.1    # Penalty weight for magnetic dipole perpendicular to magnetic field
THETA_PROGRESS_COEFF = 0.0
OMEGA_PROGRESS_COEFF = 0.0


# Evaluation stuff
NUM_EPSODES = 100



# =============================================================================
# DERIVED PATHS (don't edit)
# =============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(BASE_SAVE_PATH, EXPERIMENT_NAME)
LOG_PATH = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)

FIGURE_PATH = os.path.join(SAVE_PATH, "figures")
CHECKPOINT_PATH = os.path.join(SAVE_PATH, "checkpoints")
TENSORBOARD_PATH = os.path.join(LOG_PATH, "tensorboard")
MODEL_PATH = os.path.join(SAVE_PATH, "final_model.zip")







def make_env(dynamics_params: Dict, seed: int = None):
    """Create and wrap environment."""
    if PLANET.lower() == "earth":
        planet = Earth
    elif PLANET.lower() == "uranus":
        planet = Uranus
    else:
        raise ValueError(f"Unknown planet: {PLANET}. Choose from ['earth', 'uranus']")
    
    env = SpacecraftEnv(
        dynamics_params=dynamics_params,
        planet=planet,
        dt=DT,
        N_orbit=N_ORBIT,
        num_steps=MAX_EPISODE_STEPS,
        dyn_noise_std=DYN_NOISE_STD,
        state_limits=np.array(STATE_LIMITS),
        control_limits=CONTROL_LIMIT_SCALE * np.array([[-1, 1]] * 3),
        normalize_mag=NORMALIZE_MAG,
        mag_field_horizon=MAG_FIELD_HORIZON,
        mag_field_scale=MAG_FIELD_SCALE,
        const_mag_field=CONST_MAG_FIELD,
        theta_threshold=THETA_THRESHOLD,
        omega_threshold=OMEGA_THRESHOLD,
        theta_proximity_tol=THETA_PROXIMITY_TOL,
        quaternion_penalty=QUATERNION_PENALTY,
        omega_penalty=OMEGA_PENALTY,
        action_parallel_penalty=ACTION_PARALLEL_PENALTY,
        action_perpendicular_penalty=ACTION_PERPENDICULAR_PENALTY,
        proximity_penalty=PROXIMITY_PENALTY,
        goal_reward=GOAL_REWARD,
        theta_progress_coeff=THETA_PROGRESS_COEFF,
        omega_progress_coeff=OMEGA_PROGRESS_COEFF
    )
    
    # Wrap with Monitor for episode logging
    env = Monitor(env)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env




def evaluate_model(env: VecNormalize, model, num_episodes=100, seed=0):

    print("Num episodes for evaluation:", num_episodes)

    raw_env = env.envs[0].unwrapped  # Access the underlying SpacecraftEnv
    max_steps = raw_env.num_steps

    trajectories = np.zeros((num_episodes, max_steps+1, 7))
    target_states = np.zeros((num_episodes, 7))

    step_rewards = np.zeros((num_episodes, max_steps))  # Store rewards for each step in each episode
    step_actions = np.zeros((num_episodes, max_steps, 3))  # Store actions for each episode

    cumulative_rewards = np.zeros((num_episodes, max_steps))  # Store cumulative rewards for each episode


    # TODO: Stats to track episode performance



    for ep in range(num_episodes):
        raw_obs, _ = raw_env.reset(seed=seed + ep)
        obs = env.normalize_obs(raw_obs)  

        init_context = raw_env.episode_initial_context

        # Store initial state and target state
        trajectories[ep, 0, :] = init_context['initial_state']
        target_states[ep, :] = init_context['goal_state']

        ep_reward = 0.0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)

            raw_obs, reward, terminated, truncated, info = raw_env.step(action)
            obs = env.normalize_obs(raw_obs)  

            ep_reward += reward

            trajectories[ep, step+1, :] = raw_env.state[:7]
            step_actions[ep, step, :] = action
            step_rewards[ep, step] = reward
            cumulative_rewards[ep, step] = ep_reward

            if terminated or truncated:
                trajectories[ep, step + 2:, :] = trajectories[ep, step + 1, :]
                step_actions[ep, step + 1:, :] = np.nan
                step_rewards[ep, step + 1:] = np.nan
                cumulative_rewards[ep, step + 1:] = ep_reward
                break

        step_rewards[ep] = ep_reward

    results = {
            'trajectories': trajectories,
            'target_states': target_states,
            'step_rewards': step_rewards,
            'step_actions': step_actions,
            'cumulative_rewards': cumulative_rewards,
            'max_steps': max_steps,
        }
    
    return results

def plot_reward_evolution(results, save_path=None):
    """Plot mean reward ± std at each timestep."""
    step_rewards = results['step_rewards']
    max_steps = results['max_steps']
    
    # Compute stats ignoring NaN (episodes that ended early)
    mean_rewards = np.nanmean(step_rewards, axis=0)
    std_rewards = np.nanstd(step_rewards, axis=0)
    
    # Percentiles for bounds (more robust than std)
    p25 = np.nanpercentile(step_rewards, 25, axis=0)
    p75 = np.nanpercentile(step_rewards, 75, axis=0)
    p5 = np.nanpercentile(step_rewards, 5, axis=0)
    p95 = np.nanpercentile(step_rewards, 95, axis=0)
    
    timesteps = np.arange(max_steps)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-step reward
    ax = axes[0]
    ax.plot(timesteps, mean_rewards, 'b-', label='Mean', linewidth=2)
    ax.fill_between(timesteps, p25, p75, alpha=0.3, color='blue', label='25-75%')
    ax.fill_between(timesteps, p5, p95, alpha=0.1, color='blue', label='5-95%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.set_title('Per-Step Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative reward
    ax = axes[1]
    cum_rewards = results['cumulative_rewards']
    mean_cum = np.mean(cum_rewards, axis=0)
    p25_cum = np.percentile(cum_rewards, 25, axis=0)
    p75_cum = np.percentile(cum_rewards, 75, axis=0)
    
    ax.plot(timesteps, mean_cum, 'g-', label='Mean', linewidth=2)
    ax.fill_between(timesteps, p25_cum, p75_cum, alpha=0.3, color='green', label='25-75%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_action_evolution(results, save_path=None):
    """Plot action magnitude and per-axis actions over time."""
    step_actions = results['step_actions']
    max_steps = results['max_steps']
    timesteps = np.arange(max_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axis_labels = ['Torque X', 'Torque Y', 'Torque Z']
    colors = ['r', 'g', 'b']
    
    # Per-axis actions
    for i in range(3):
        ax = axes[i // 2, i % 2]
        actions_i = step_actions[:, :, i]
        
        mean_act = np.nanmean(actions_i, axis=0)
        p25 = np.nanpercentile(actions_i, 25, axis=0)
        p75 = np.nanpercentile(actions_i, 75, axis=0)
        
        ax.plot(timesteps, mean_act, f'{colors[i]}-', label='Mean', linewidth=2)
        ax.fill_between(timesteps, p25, p75, alpha=0.3, color=colors[i], label='25-75%')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Normalized Action')
        ax.set_title(axis_labels[i])
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Action magnitude
    ax = axes[1, 1]
    action_magnitude = np.linalg.norm(step_actions, axis=2)
    mean_mag = np.nanmean(action_magnitude, axis=0)
    p25_mag = np.nanpercentile(action_magnitude, 25, axis=0)
    p75_mag = np.nanpercentile(action_magnitude, 75, axis=0)
    
    ax.plot(timesteps, mean_mag, 'k-', label='Mean', linewidth=2)
    ax.fill_between(timesteps, p25_mag, p75_mag, alpha=0.3, color='gray', label='25-75%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('||Action||')
    ax.set_title('Action Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_theta_omega_evolution(results, save_path=None):
    """Plot theta and omega evolution over time."""
    trajectories = results['trajectories']
    target_states = results['target_states']
    max_steps = results['max_steps']
    timesteps = np.arange(max_steps + 1)
    
    # Compute theta and omega errors
    quat_traj = trajectories[:, :, :4]
    omega_traj = trajectories[:, :, 4:7]
    
    quat_target = target_states[:, :4][:, np.newaxis, :]
    omega_target = target_states[:, 4:7][:, np.newaxis, :]
    
    # Quaternion error (angle)
    dot_product = np.einsum('ijk,ijk->ij', quat_traj, quat_target)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_error = 2 * np.arccos(np.abs(dot_product)) * (180 / np.pi)  # degrees
    
    # Omega error
    omega_error = np.linalg.norm(omega_traj - omega_target, axis=2) * (180 / np.pi)  # degrees/s
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Theta error
    ax = axes[0]
    mean_theta = np.nanmean(theta_error, axis=0)
    p25_theta = np.nanpercentile(theta_error, 25, axis=0)
    p75_theta = np.nanpercentile(theta_error, 75, axis=0)
    
    ax.plot(timesteps, mean_theta, 'b-', label='Mean', linewidth=2)
    ax.fill_between(timesteps, p25_theta, p75_theta, alpha=0.3, color='blue', label='25-75%')
    ax.axhline(y=THETA_THRESHOLD * (180 / np.pi), color='r', linestyle='--', label='Theta Threshold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Theta Error (deg)')
    ax.set_title('Theta Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Omega error
    ax = axes[1]
    mean_omega = np.nanmean(omega_error, axis=0)
    p25_omega = np.nanpercentile(omega_error, 25, axis=0)
    p75_omega = np.nanpercentile(omega_error, 75, axis=0)

    ax.plot(timesteps, mean_omega, 'g-', label='Mean', linewidth=2)
    ax.fill_between(timesteps, p25_omega, p75_omega, alpha=0.3, color='green', label='25-75%')
    ax.axhline(y=OMEGA_THRESHOLD * (180 / np.pi), color='r', linestyle='--', label='Omega Threshold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Omega Error (deg/s)')
    ax.set_title('Omega Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig


def print_eval_summary(results):
    """Print summary statistics."""
    cum_rewards = results['cumulative_rewards'][:, -1]
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {len(cum_rewards)}")
    print(f"\nReward:")
    print(f"  Mean: {np.mean(cum_rewards):.2f} ± {np.std(cum_rewards):.2f}")
    print(f"  Min/Max: {np.min(cum_rewards):.2f} / {np.max(cum_rewards):.2f}")
    print(f"\nEpisode Length:")
    print("="*50)




def save_results():
    pass


def main():


    # Path to magnetic field models
    model_path = URANUS_MPC_PATH + '/models/'

    planet = Earth if PLANET.lower() == "earth" else Uranus

    # Load learned magnetic field models
    if planet is Earth:
        model_s, _ = load_model(filename=model_path + '/earth_b_4d.eqx') 
    elif planet is Uranus:
        model_s, _ = load_model(filename=model_path + '/uranus_b_4d.eqx')


    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    dynamics_params = spacecraft_dynamics.dynamics_params
    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=DT)

    rl_model_path = SAVE_PATH + '/final_model.zip'

    # Create directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # print exp info
    print("=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 60)
    print(f"Planet:       {PLANET}")
    print(f"Save path:    {SAVE_PATH}")
    print(f"Log path:     {LOG_PATH}")
    if EXPERIMENT_NOTES:
        print(f"Notes:        {EXPERIMENT_NOTES}")
    print("=" * 60)


    # make env
    eval_env = DummyVecEnv([lambda: make_env(dynamics_params=dynamics_params, seed=None)])
    eval_env = VecNormalize.load(os.path.join(SAVE_PATH, "vecnormalize_stats.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # load model
    model = PPO.load(rl_model_path, env=eval_env)

    # Run evaluations and collect trajectories
    results = evaluate_model(eval_env, model, num_episodes=NUM_EPSODES, seed=0)

    trajectories = results['trajectories']
    target_states = results['target_states']
    step_rewards = results['step_rewards']
    step_actions = results['step_actions']
    cumulative_rewards = results['cumulative_rewards']

    # Thresholds for success
    angle_threshold = 15 # phi = 2*arccos(q.T @ q_g)*180/pi
    omega_threshold = 5 # (deg/s)

    # Tolerances for stability
    angle_tol = 10
    omega_tol = 5
    tail_length = 100
    # time_hist_max = 250
    angle_hist_max = 30
    omega_hist_max = 15

    _ = system.plot_costs(trajectories, target_states, plot_stats=True)

    system.plot_violin_and_bar(trajectories, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)

    print_eval_summary(results)
    plot_reward_evolution(results, save_path=str(FIGURE_PATH + '/reward_evolution.png'))
    plot_action_evolution(results, save_path=str(FIGURE_PATH + '/action_evolution.png'))



if __name__ == "__main__":


    main()


