"""
Training script for SpacecraftEnv with PPO using SBX.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize



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

from utils.learning import load_model

# =============================================================================
# EXPERIMENT CONFIG - EDIT THIS SECTION
# =============================================================================

# Experiment params
EXPERIMENT_NAME = "spacecraft_ppo_random_test"
EXPERIMENT_NOTES = "Initial PPO training on Earth orbit"

# Environment
PLANET = "earth"                 # "earth" or "uranus"
DT = 0.1                         # Simulation timestep
N_ORBIT = 5000                   # Orbit trajectory length
DYN_NOISE_STD = 1e-6             # Dynamics noise

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
OMEGA_PENALTY = 0.5               # Penalty weight for omega error
ACTION_PENALTY = 0.1              # Penalty weight for action magnitude
PROXIMITY_PENALTY = 5.0          # Reward for being within proximity tolerance
GOAL_REWARD = 50.0              # Bonus for reaching goal



ACTION_PARALLEL_PENALTY = 0.1        # Penalty weight for magnetic dipole parallel to magnetic field
ACTION_PERPENDICULAR_PENALTY = 0.1    # Penalty weight for magnetic dipole perpendicular to magnetic field
THETA_PROGRESS_COEFF = 0.0
OMEGA_PROGRESS_COEFF = 0.0

# TODO:
# Add reward terms to heavier penalize magnetic dipole component along magnetic field
# Check training logs to check if agent is over/underfitting, saturating, 
# Not enough neurons, not enough exploration, not enough training time, not good reward shaping, not enough reward signal

# TODO: 
# Consider using torque as an action space instead of magnetic dipole moment

# PPO Hyperparameters
LEARNING_RATE = 1e-3
LEARNING_RATE_SCHEDULE = "constant"  # "constant" or "linear" or cosine
LEARNING_RATE_FINAL = 1e-5            # Minimum learning rate for linear schedule

MAX_EPISODE_STEPS = 1500          # Max steps per episode
N_ROLLOUTS = 3                      # Number of rollouts per update
N_STEPS = N_ROLLOUTS * MAX_EPISODE_STEPS  # Steps per rollout
BATCH_SIZE = 500                  # Minibatch size
N_EPOCHS = 10                    # SGD epochs per update
GAMMA = 0.99                     # Discount factor
GAE_LAMBDA = 0.95                # GAE lambda
CLIP_RANGE = 0.2                 # PPO clip range
ENT_COEF = 1e-2                  # Entropy coefficient
VF_COEF = 0.5                    # Value function coefficient
MAX_GRAD_NORM = 0.5              # Gradient clipping

# Policy network
POLICY_TYPE = "MlpPolicy"
# NET_ARCH = [256, 256]          # Uncomment to customize network size
# POLICY_NET_ARCH = [512, 512]
# VALUE_NET_ARCH = [512, 512]

NET_ARCH = [256, 256]

# Training
TOTAL_TIMESTEPS = 1_500_000
EVAL_FREQ = 5_000                # Evaluate every N timesteps
N_EVAL_EPISODES = 10             # Episodes per evaluation
CHECKPOINT_FREQ = 50_000         # Save checkpoint every N timesteps

# Reproducibility
SEED = 42


# =============================================================================
# DERIVED PATHS (don't edit)
# =============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(BASE_SAVE_PATH, EXPERIMENT_NAME)
LOG_PATH = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_PATH, "checkpoints")
TENSORBOARD_PATH = os.path.join(LOG_PATH, "tensorboard")



# Resume training
RESUME_TRAINING = False
RESUME_TIMESTEPS = 1_000_000
RESUME_MODEL_PATH = os.path.join(CHECKPOINT_PATH, f"model_{RESUME_TIMESTEPS}_steps.zip")  # Path to checkpoint to resume from





# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

class CheckpointCallbackWithVecNormalize(CheckpointCallback):
    """Checkpoint callback that also saves VecNormalize stats."""
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Save VecNormalize on same schedule as model checkpoints
        if self.n_calls % self.save_freq == 0:
            vecnorm_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
            self.training_env.save(vecnorm_path)
            if self.verbose > 0:
                print(f"Saved VecNormalize to {vecnorm_path}")
        
        return result


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_config() -> dict:
    """Return all config as a dictionary for saving."""
    return {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "notes": EXPERIMENT_NOTES,
            "timestamp": TIMESTAMP,
        },
        "environment": {
            "planet": PLANET,
            "dt": DT,
            "n_orbit": N_ORBIT,
            "num_steps": MAX_EPISODE_STEPS,
            "dyn_noise_std": DYN_NOISE_STD,
            "state_limits": STATE_LIMITS,
            "control_limit_scale": CONTROL_LIMIT_SCALE,
            "theta_tol": THETA_THRESHOLD,
            "omega_tol": OMEGA_THRESHOLD,
            "quaternion_penalty": QUATERNION_PENALTY,
            "omega_penalty": OMEGA_PENALTY,
            "goal_reward": GOAL_REWARD,
            "normalize_mag": NORMALIZE_MAG,
            "mag_field_horizon": MAG_FIELD_HORIZON,
            "const_mag_field": CONST_MAG_FIELD,
        },
        "ppo": {
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "policy_type": POLICY_TYPE,
        },
        "training": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "seed": SEED,
        },
        "resume": {
            "resumed": RESUME_TRAINING,
            "resumed_from": RESUME_MODEL_PATH,
        },
    }


def save_config(config: dict, path: str):
    """Save config to JSON file."""
    config_file = os.path.join(path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Config saved to: {config_file}")



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



def lr_scheduler(start_lr: float, schedule_type: str, end_lr: float = 0.0, total_timesteps: int = TOTAL_TIMESTEPS, initial_timesteps: int = 0):
    """
    Get learning rate schedule function for SB3/sbx.
    
    Args:
        schedule_type: "linear", "cosine", or "constant"
        start_lr: Initial learning rate
        end_lr: Final learning rate (ignored for constant)
    
    Returns:
        Schedule function: (progress_remaining: float) -> float
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining is for remaining_timesteps, not total
        # Convert to global progress
        remaining_steps = progress_remaining * (total_timesteps - initial_timesteps)
        global_progress_remaining = remaining_steps / total_timesteps

        if schedule_type == "constant":
            return start_lr
        elif schedule_type == "linear":
            return end_lr + global_progress_remaining * (start_lr - end_lr)
        elif schedule_type == "cosine":
            # Cosine annealing: slower decay at start/end, faster in middle
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (1 - global_progress_remaining)))
            return end_lr + (start_lr - end_lr) * cosine_decay
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}. Use 'linear', 'cosine', or 'constant'")
    
    return func



# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    # Print experiment info
    print("=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 60)
    print(f"Planet:       {PLANET}")
    print(f"Timesteps:    {TOTAL_TIMESTEPS:,}")
    print(f"Save path:    {SAVE_PATH}")
    print(f"Log path:     {LOG_PATH}")
    if EXPERIMENT_NOTES:
        print(f"Notes:        {EXPERIMENT_NOTES}")
    print("=" * 60)
    
    # Save config
    config = get_config()
    save_config(config, SAVE_PATH)
    
    # Set seeds
    set_seed(SEED)

    # Dynamics
    model_path = URANUS_MPC_PATH + '/models/'
    if PLANET.lower() == "earth":
        planet = Earth
    elif PLANET.lower() == "uranus":
        planet = Uranus
    if planet is Earth:
        model_s, _ = load_model(filename=model_path + '/earth_b_4d.eqx') 
    elif planet is Uranus:
        model_s, _ = load_model(filename=model_path + '/uranus_b_4d.eqx')
    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    dynamics_params = spacecraft_dynamics.dynamics_params
    
    # Create environments
    print("[INFO] Creating environments...")
    # train_env = make_env(dynamics=spacecraft_dynamics, seed=SEED)
    # eval_env = make_env(dynamics=spacecraft_dynamics, seed=SEED + 1000)
        # ====== 1. Create Training Env with VecNormalize ======
    train_env = DummyVecEnv([lambda: make_env(dynamics_params=dynamics_params, seed=SEED)])
    

    # Same for eval_env:
    # eval_env = DummyVecEnv([lambda: make_env(dynamics=spacecraft_dynamics, seed=SEED + 1000)])
    # eval_env = VecNormalize(
    #     eval_env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     training=False,
    #     clip_obs=10.0,
    # )
    
    # Test environment
    print("[DEBUG] Testing environment...")
    # obs, info = train_env.reset()
    obs = train_env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space:      {train_env.action_space}")
    
    # Create or load model
    if RESUME_TRAINING and RESUME_MODEL_PATH:
        print(f"[INFO] Resuming from: {RESUME_MODEL_PATH}")

        initial_timesteps = RESUME_TIMESTEPS
        lr_schedule = lr_scheduler(
                    LEARNING_RATE, LEARNING_RATE_SCHEDULE, LEARNING_RATE_FINAL,
                    TOTAL_TIMESTEPS, initial_timesteps
                )

        # Load vecnormalize statistics - try checkpoint-specific first, then fallback
        vecnorm_checkpoint_path = os.path.join(CHECKPOINT_PATH, f"vecnormalize_{RESUME_TIMESTEPS}_steps.pkl")
        vecnorm_final_path = os.path.join(SAVE_PATH, "vecnormalize_stats.pkl")
        
        if os.path.exists(vecnorm_checkpoint_path):
            train_env = VecNormalize.load(vecnorm_checkpoint_path, venv=train_env)
            print(f"[INFO] Loaded VecNormalize stats from: {vecnorm_checkpoint_path}")
        elif os.path.exists(vecnorm_final_path):
            train_env = VecNormalize.load(vecnorm_final_path, venv=train_env)
            print(f"[WARN] Checkpoint VecNormalize not found, using final stats: {vecnorm_final_path}")
        else:
            raise FileNotFoundError(
                f"Cannot resume: VecNormalize stats not found at {vecnorm_checkpoint_path} or {vecnorm_final_path}"
            )
        
        train_env.training = True  # Ensure training mode is set for VecNormalize

        model = PPO.load(RESUME_MODEL_PATH, env=train_env, tensorboard_log=TENSORBOARD_PATH, learning_rate=lr_schedule)
        
        # Try to restore timestep count from filename
        basename = os.path.basename(RESUME_MODEL_PATH)
        if "_steps" in basename:
            try:
                step_str = basename.split("_steps")[0].split("_")[-1]
                model.num_timesteps = initial_timesteps
                print(f"[INFO] Restored num_timesteps: {model.num_timesteps:,}")
            except ValueError:
                print("[WARN] Could not parse num_timesteps from checkpoint filename")
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=GAMMA,
        )
        print("[INFO] Creating new model...")

        policy_kwargs = dict(net_arch=NET_ARCH)

        model = PPO(
            policy=POLICY_TYPE,
            env=train_env,
            learning_rate=lr_scheduler(LEARNING_RATE, LEARNING_RATE_SCHEDULE, LEARNING_RATE_FINAL),
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            verbose=1,
            seed=SEED,
            tensorboard_log=TENSORBOARD_PATH,
            policy_kwargs=policy_kwargs,
        )
    
    # Print model info
    print(f"\n[INFO] Policy architecture:")
    print(f"  {model.policy}")
    
    # Setup callbacks
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=os.path.join(SAVE_PATH, "best_model"),
    #     log_path=os.path.join(LOG_PATH, "eval"),
    #     eval_freq=EVAL_FREQ,
    #     n_eval_episodes=N_EVAL_EPISODES,
    #     deterministic=True,
    #     verbose=1,
    # )
    
    checkpoint_callback = CheckpointCallbackWithVecNormalize(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_PATH,
        name_prefix="model",
        verbose=1,
    )
    
    # callbacks = CallbackList([eval_callback, checkpoint_callback])
    callbacks = CallbackList([checkpoint_callback])

    
    # Calculate remaining timesteps if resuming
    remaining_timesteps = TOTAL_TIMESTEPS
    if RESUME_TRAINING and hasattr(model, "num_timesteps"):
        remaining_timesteps = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
        print(f"[INFO] Remaining timesteps: {remaining_timesteps:,}")
    
    # Train
    print(f"\n[INFO] Starting training...")
    print("=" * 60)
    
    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=not RESUME_TRAINING,
            )
            print("[INFO] Training complete!")
        else:
            print("[INFO] Target timesteps already reached; skipping training.")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")


    # After training, sync once for final eval
    # eval_env.obs_rms = train_env.obs_rms
    # eval_env.ret_rms = train_env.ret_rms
        
    # Save final model
    final_model_path = os.path.join(SAVE_PATH, "final_model")
    model.save(final_model_path)
    print(f"[INFO] Final model saved to: {final_model_path}")

    # Save VecNormalize statistics
    vecnormalize_stats_path = os.path.join(SAVE_PATH, "vecnormalize_stats.pkl")
    train_env.save(vecnormalize_stats_path)
    print(f"[INFO] VecNormalize stats saved to: {vecnormalize_stats_path}")
    
    # Cleanup
    train_env.close()
    # eval_env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model:  {final_model_path}")
    print(f"Best model:   {os.path.join(SAVE_PATH, 'best_model')}")
    print(f"Checkpoints:  {CHECKPOINT_PATH}")
    print(f"Logs:         {LOG_PATH}")
    print(f"Config:       {os.path.join(SAVE_PATH, 'config.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()