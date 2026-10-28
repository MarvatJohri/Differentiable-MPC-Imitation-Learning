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
import sys
from pathlib import Path

HERE = Path(__file__).parent
MJ_WORK = HERE.parent
sys.path.append(str(MJ_WORK))

import equinox as eqx
import matplotlib.pyplot as plt
from simulation_env_jax import SpacecraftEnvJax
from rl_utils.generate_trajs import n_rollouts_controller
from rl_utils.plotting_and_evaluate import (
    extract_trajectory_data,
    plot_rewards,
    plot_control_trajectories,
    plot_cost_weights,
)
# from simulation_env_jax_test1 import SpacecraftEnvJax
from configs import ExpConfig, SpacecraftEnvConfig, ACMPCHyperparameters, get_env
from mj_utils import compute_metrics, print_metrics

from acmpc_model import ActorCriticMPC, load_acmpc_network, get_acmpc_controller





def main():

    exp_config = ExpConfig()
    hyperparams = ACMPCHyperparameters()
    env_config = SpacecraftEnvConfig()

    SEED = 42

    ACMPC_BASE_SAVE_PATH = exp_config.acmpc_base_save_path
    ACMPC_BASE_LOG_PATH = exp_config.acmpc_base_log_path


    ACMPC_EXPERIMENT_NAME = exp_config.acmpc_experiment_name
    ACMPC_EXPERIMENT_NOTES = exp_config.acmpc_experiment_notes

    NX = exp_config.nx
    NU = exp_config.nu

    RESUME_TRAINING = exp_config.resume_ppo_training
    RESUME_TIMESTEPS = exp_config.ppo_resume_timesteps
    RESUME_MODEL_PATH = exp_config.ppo_resume_model_path


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
    env = get_env(env_name="spacecraft_env")




    config = {
        "LEARNING_RATE": hyperparams.learning_rate,
        "ACTOR_LEARNING_RATE": hyperparams.actor_learning_rate,
        "CRITIC_LEARNING_RATE": hyperparams.critic_learning_rate,
        "NET_ARCH": hyperparams.net_arch,
        "LEARNING_RATE_SCHEDULE": hyperparams.learning_rate_schedule,
        "LEARNING_RATE_FINAL": hyperparams.learning_rate_final,
        "NUM_ENVS": hyperparams.n_envs,
        "NUM_STEPS": hyperparams.n_steps,
        "MPC_HORIZON": hyperparams.mpc_horizon,
        "OUTPUT_ACTIVATION": hyperparams.output_activation,
        "NETWORK_EPSILON": hyperparams.network_epsilon,
        "DECOMPOSITION_TYPE": hyperparams.decomposition_type,
        "QR_OUTPUT_HORIZON": hyperparams.qr_output_horizon,
        "REPLAN_FREQUENCY": hyperparams.replan_frequency,
        "TOTAL_TIMESTEPS": hyperparams.total_timesteps,
        "RESUME_TRAINING": exp_config.resume_ppo_training,
        "RESUME_TIMESTEPS": exp_config.ppo_resume_timesteps,
        "RESUME_MODEL_PATH": exp_config.ppo_resume_model_path,
        "UPDATE_EPOCHS": hyperparams.n_epochs,
        "NUM_MINIBATCHES": hyperparams.n_minibatches,
        "MINIBATCH_SIZE": hyperparams.minibatch_size,
        "GAMMA": hyperparams.gamma,
        "LOG_STD_INIT": hyperparams.log_std_init,
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
        "LOG_EVERY_UPDATES": hyperparams.log_every_updates,
    }

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    model = ActorCriticMPC(
        obs_dim=config["OBS_DIM"],
        act_dim=config["ACT_DIM"],
        layers=config["NET_ARCH"],
        key=subkey,
        init_log_std=config["LOG_STD_INIT"],
        activation=config["ACTIVATION"],
        output_activation=config["OUTPUT_ACTIVATION"],
        qr_output_horizon=config["QR_OUTPUT_HORIZON"],
        decomposition_type=config["DECOMPOSITION_TYPE"],
        eps=config["NETWORK_EPSILON"],
    )

    # Load model
    SAVE_PATH = os.path.join(ACMPC_BASE_SAVE_PATH, ACMPC_EXPERIMENT_NAME)
    LOG_PATH = os.path.join(ACMPC_BASE_LOG_PATH, ACMPC_EXPERIMENT_NAME)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "final_model.eqx")
    model = load_acmpc_network(model, MODEL_SAVE_PATH)


    # Get policy function
    acmpc_controller = get_acmpc_controller(model)


    # Generate trajectories using the loaded model
    trajectories, keys = n_rollouts_controller(env,
                                               acmpc_controller,
                                               key,horizon=acmpc_controller.mpc_horizon,
                                               max_ep_steps=MAX_EP_STEPS,
                                               n_rollouts=100,
                                               replan_freq=config["REPLAN_FREQUENCY"],
                                               record_qr=True)
    print("Calculated trajectories")
    # rewards = trajectories[-1]
    # returns = rewards.sum(axis=1)

    # print("returns per rollout:", returns)
    # print("mean return:", returns.mean())
    # print("median return:", jnp.median(returns))
    # print("min/max:", returns.min(), returns.max())

    # large_positive_counts = (rewards > 20.0).sum(axis=1)
    # print("large positive events per rollout:", large_positive_counts)
    metrics = compute_metrics(
        trajectories,
        dt=env.dt,
        angle_threshold=float(jnp.rad2deg(env.theta_threshold)),
        omega_threshold=float(jnp.rad2deg(env.omega_threshold)),
        angle_tol_stability=float(jnp.rad2deg(THETA_STABILITY_TOL)),
        omega_tol_stability=float(jnp.rad2deg(OMEGA_STABILITY_TOL)),
        tail_length=100,
        label="ACMPC",
    )
    print_metrics(metrics)

    runs = {
        "ACMPC": extract_trajectory_data(
            trajectories,
            dt=DT,
            max_torque=MAX_TORQUE,
            action_low=CONTROL_LIMITS[:, 0],
            action_high=CONTROL_LIMITS[:, 1],
            action_units="torque",
        )
    }
    figure_dir = Path(LOG_PATH) / "figures"
    plot_rewards(runs, save_path=figure_dir / "rewards.png")
    plot_control_trajectories(
        runs, episode=0, save_path=figure_dir / "controls_episode_0.png",
    )
    plot_cost_weights(
        runs, episode=0, save_path=figure_dir / "qr_weights_episode_0.png",
    )
    plt.show()


if __name__=="__main__":

    main()
    
