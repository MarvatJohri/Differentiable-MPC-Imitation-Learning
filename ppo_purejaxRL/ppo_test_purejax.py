import jax
import jax.numpy as jnp
import optax
from typing import Sequence, NamedTuple, Any
# from wrappers import (
#     LogWrapper,
#     BraxGymnaxWrapper,
#     VecEnv,
#     NormalizeVecObservation,
#     NormalizeVecReward,
#     ClipAction,
# )
import os

import equinox as eqx
from simulation_env_jax import SpacecraftEnvJax, generate_n_trajectories
from configs import ExpConfig, SpacecraftEnvConfig, PPOHyperparameters
from mj_utils import make_dummy_controller, compute_metrics, print_metrics

from ppo_model import ActorCritic, save_network, load_network


def get_policy(model: ActorCritic):

    def policy_fn(obs):
        mean = model.mean(obs)
        return mean

    return policy_fn




def main():
    exp_config = ExpConfig()
    hyperparams = PPOHyperparameters()
    env_config = SpacecraftEnvConfig()

    PPO_BASE_SAVE_PATH = exp_config.ppo_base_save_path
    PPO_BASE_LOG_PATH = exp_config.ppo_base_log_path


    PPO_EXPERIMENT_NAME = exp_config.ppo_experiment_name
    PPO_EXPERIMENT_NOTES = exp_config.ppo_experiment_notes

    NX = exp_config.nx
    NU = exp_config.nu

    RESUME_TRAINING = exp_config.resume_ppo_training
    RESUME_TIMESTEPS = exp_config.ppo_resume_timesteps
    RESUME_MODEL_PATH = exp_config.ppo_resume_model_path
    SEED = exp_config.seed


    DYNAMICS_PARAMETERS = env_config.spacecraft_dynamics_parameters
    DT = env_config.dt
    MAX_EP_STEPS = env_config.max_ep_steps
    STATE_LIMITS = jnp.asarray(env_config.state_limits)
    CONTROL_LIMITS = jnp.asarray(env_config.control_limits)
    MAX_TORQUE = env_config.max_torque
    DYN_NOISE_STD = env_config.dyn_noise_std
    THETA_THRESHOLD = env_config.theta_threshold
    OMEGA_THRESHOLD = env_config.omega_threshold
    OMEGA_PENALTY = env_config.omega_penalty
    OMEGA_FAIL_PENALTY = env_config.omega_fail_penalty
    ACTION_PENALTY = env_config.action_penalty
    THETA_THRESHOLD_REWARD = env_config.theta_threshold_reward
    GOAL_REWARD = env_config.goal_reward
    THETA_STABILITY_TOL = env_config.theta_stability_tol
    OMEGA_STABILITY_TOL = env_config.omega_stability_tol

    # Make env
    env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMETERS,
                            dt=DT,
                            max_ep_steps=MAX_EP_STEPS,
                            state_limits=STATE_LIMITS,
                            control_limits=CONTROL_LIMITS,
                            max_torque=MAX_TORQUE,
                            dyn_noise_std=DYN_NOISE_STD,
                            theta_threshold=THETA_THRESHOLD,
                            omega_threshold=OMEGA_THRESHOLD,
                            theta_threshold_reward=THETA_THRESHOLD_REWARD,
                            omega_penalty=OMEGA_PENALTY,
                            omega_fail_penalty=OMEGA_FAIL_PENALTY,
                            action_penalty=ACTION_PENALTY,
                            goal_reward=GOAL_REWARD)


    config = {
        "LEARNING_RATE": hyperparams.learning_rate,
        "NET_ARCH": hyperparams.net_arch,
        "LEARNING_RATE_SCHEDULE": hyperparams.learning_rate_schedule,
        "LEARNING_RATE_FINAL": hyperparams.learning_rate_final,
        "NUM_ENVS": hyperparams.n_envs,
        "NUM_STEPS": hyperparams.n_steps,
        "TOTAL_TIMESTEPS": hyperparams.total_timesteps,
        "RESUME_TRAINING": exp_config.resume_ppo_training,
        "RESUME_TIMESTEPS": exp_config.ppo_resume_timesteps,
        "RESUME_MODEL_PATH": exp_config.ppo_resume_model_path,
        "UPDATE_EPOCHS": hyperparams.n_epochs,
        "NUM_MINIBATCHES": hyperparams.n_minibatches,
        "BATCH_SIZE": hyperparams.batch_size,
        "GAMMA": hyperparams.gamma,
        "GAE_LAMBDA": hyperparams.gae_lambda,
        "CLIP_EPS": hyperparams.clip_range,
        "ENT_COEF": hyperparams.ent_coef,
        "VF_COEF": hyperparams.vf_coef,
        "CLIP_VALUE_LOSS": hyperparams.clip_val_loss,
        "VF_CLIP_EPS": hyperparams.vf_clip_eps,
        "MAX_GRAD_NORM": hyperparams.max_grad_norm,
        "ACTIVATION": "tanh",
        "DEBUG": True,
        "ACT_DIM": NU,
        "OBS_DIM": NX,
        "NORMALIZE_REWARD": True,
        "BOOTSTRAP_TIMEOUTS": True,
        "LOG_EVERY": hyperparams.log_every,
    }

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    model = ActorCritic(
        obs_dim=config["OBS_DIM"],
        act_dim=config["ACT_DIM"],
        layers=config["NET_ARCH"],
        key=subkey,
    )

    # Load model
    SAVE_PATH = os.path.join(PPO_BASE_SAVE_PATH, PPO_EXPERIMENT_NAME)
    LOG_PATH = os.path.join(PPO_BASE_LOG_PATH, PPO_EXPERIMENT_NAME)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "final_model.eqx")
    model = load_network(model, MODEL_SAVE_PATH)


    # Make dummy controller for testing
    dummy_controller = make_dummy_controller(NX, NU)

    # Get policy function
    ppo_policy = get_policy(model)

    # Load env
    env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMETERS,
                            dt=DT,
                            max_ep_steps=MAX_EP_STEPS,
                            state_limits=STATE_LIMITS,
                            control_limits=CONTROL_LIMITS,
                            max_torque=MAX_TORQUE,
                            dyn_noise_std=DYN_NOISE_STD,
                            theta_threshold=THETA_THRESHOLD,
                            omega_threshold=OMEGA_THRESHOLD,
                            theta_threshold_reward=THETA_THRESHOLD_REWARD,
                            omega_penalty=OMEGA_PENALTY,
                            omega_fail_penalty=OMEGA_FAIL_PENALTY,
                            action_penalty=ACTION_PENALTY,
                            goal_reward=GOAL_REWARD)

    # Generate trajectories using the loaded model
    trajectories, keys = generate_n_trajectories(env, 
                                                 dummy_controller,
                                                 ppo_policy,
                                                 key,
                                                 1.0,
                                                 max_steps=MAX_EP_STEPS,
                                                 n_trajectories=100)

    metrics = compute_metrics(trajectories)
    print_metrics(metrics)

    