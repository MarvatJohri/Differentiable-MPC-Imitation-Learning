"""
Training script for SpacecraftEnv with PPO using SBX.
"""

import os
import random
from typing import Dict
import numpy as np
from datetime import datetime

from sbx import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs import ExpConfig, PPOHyperparameters, SpacecraftEnvConfig
from simulation_env_simpler import SpacecraftEnv


# Reproducibility
SEED = ExpConfig().seed


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






def lr_scheduler(start_lr: float, schedule_type: str, end_lr: float = 1e-5, total_timesteps: int = 1_500_000, initial_timesteps: int = 0):
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


    # Load hyperparameters from configs file
    exp_config = ExpConfig()
    hyperparams = PPOHyperparameters()
    env_config = SpacecraftEnvConfig()

    PPO_BASE_SAVE_PATH = exp_config.ppo_base_save_path
    PPO_BASE_LOG_PATH = exp_config.ppo_base_log_path


    PPO_EXPERIMENT_NAME = exp_config.ppo_experiment_name
    PPO_EXPERIMENT_NOTES = exp_config.ppo_experiment_notes


    DYNAMICS_PARAMETERS = env_config.spacecraft_dynamics_parameters
    DT = env_config.dt
    MAX_EP_STEPS = env_config.max_ep_steps
    STATE_LIMITS = np.asarray(env_config.state_limits)
    CONTROL_LIMITS = np.asarray(env_config.control_limits)
    DYN_NOISE_STD = env_config.dyn_noise_std
    THETA_THRESHOLD = env_config.theta_threshold
    OMEGA_THRESHOLD = env_config.omega_threshold
    OMEGA_PENALTY = env_config.omega_penalty
    ACTION_PENALTY = env_config.action_penalty
    GOAL_REWARD = env_config.goal_reward
    THETA_STABILITY_TOL = env_config.theta_stability_tol
    OMEGA_STABILITY_TOL = env_config.omega_stability_tol


    POLICY_TYPE = hyperparams.policy_type
    NET_ARCH = hyperparams.net_arch
    LEARNING_RATE = hyperparams.learning_rate
    LEARNING_RATE_SCHEDULE = hyperparams.learning_rate_schedule
    LEARNING_RATE_FINAL = hyperparams.learning_rate_final
    N_ROLLOUTS = hyperparams.n_rollouts
    N_STEPS = hyperparams.n_steps
    N_EPOCHS = hyperparams.n_epochs
    BATCH_SIZE = hyperparams.batch_size
    GAMMA = hyperparams.gamma
    GAE_LAMBDA = hyperparams.gae_lambda
    CLIP_RANGE = hyperparams.clip_range
    ENT_COEF = hyperparams.ent_coef
    VF_COEF = hyperparams.vf_coef
    MAX_GRAD_NORM = hyperparams.max_grad_norm
    TOTAL_TIMESTEPS = hyperparams.total_timesteps
    CHECKPOINT_FREQ = hyperparams.checkpoint_freq





    # Create directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    
    # Print experiment info
    print("=" * 60)
    print(f"EXPERIMENT: {PPO_EXPERIMENT_NAME}")
    print("=" * 60)
    print(f"Timesteps:    {TOTAL_TIMESTEPS:,}")
    print(f"Save path:    {SAVE_PATH}")
    print(f"Log path:     {LOG_PATH}")
    if PPO_EXPERIMENT_NOTES:
        print(f"Notes:        {PPO_EXPERIMENT_NOTES}")
    print("=" * 60)




    # Make env function for DummyVecEnv

    def make_env(dynamics_params: Dict, seed: int = None):
        """Create and wrap environment."""
        env = SpacecraftEnv(
            dynamics_params=dynamics_params,
            dt=DT,
            max_ep_steps=MAX_EP_STEPS,
            dyn_noise_std=DYN_NOISE_STD,
            state_limits=np.array(STATE_LIMITS),
            control_limits=CONTROL_LIMITS,
            theta_threshold=THETA_THRESHOLD,
            omega_threshold=OMEGA_THRESHOLD,
            omega_penalty=OMEGA_PENALTY,
            action_penalty=ACTION_PENALTY,
            goal_reward=GOAL_REWARD,
        )
        
        # Wrap with Monitor for episode logging
        env = Monitor(env)
        
        if seed is not None:
            env.reset(seed=seed)
        
        return env




    # =============================================================================
    # DERIVED PATHS (don't edit)
    # =============================================================================

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_PATH = os.path.join(PPO_BASE_SAVE_PATH, PPO_EXPERIMENT_NAME)
    LOG_PATH = os.path.join(PPO_BASE_LOG_PATH, PPO_EXPERIMENT_NAME)
    CHECKPOINT_PATH = os.path.join(SAVE_PATH, "checkpoints")
    TENSORBOARD_PATH = os.path.join(LOG_PATH, "tensorboard")



    # Resume training
    RESUME_TRAINING = False
    RESUME_TIMESTEPS = 1_000_000
    RESUME_MODEL_PATH = os.path.join(CHECKPOINT_PATH, f"model_{RESUME_TIMESTEPS}_steps.zip")  # Path to checkpoint to resume from






    
    # Set seeds
    set_seed(SEED)

    
    # Create environments
    print("[INFO] Creating environments...")

        # ====== 1. Create Training Env with VecNormalize ======
    train_env = DummyVecEnv([lambda: make_env(dynamics_params=DYNAMICS_PARAMETERS, seed=SEED)])
    

    
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