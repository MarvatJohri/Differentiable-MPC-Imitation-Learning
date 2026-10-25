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
from simulation_env_jax import SpacecraftEnvJax
from utils.generate_trajs import n_rollouts_controller
# from simulation_env_jax_test1 import SpacecraftEnvJax
from configs import ExpConfig, SpacecraftEnvConfig, MPCParameters
from mj_utils import compute_metrics, print_metrics

from mpc_model import MPCController




def main():

    exp_config = ExpConfig()
    hyperparams = MPCParameters()
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
    STATE_LIMITS_MRP = jnp.asarray(env_config.state_limits_mrp)
    CONTROL_LIMITS = jnp.asarray(env_config.control_limits)
    CONTROL_LIMITS_TORQUE = jnp.asarray(env_config.control_limits_torque)
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




    MPC_HORIZON = hyperparams.mpc_horizon
    Q = hyperparams.Q
    R_normalized = hyperparams.R_normalized


    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    mpc_controller = MPCController(Q=Q,
                               R_normalized=R_normalized,
                               mpc_horizon=MPC_HORIZON, 
                               dt=DT,
                               state_limits=STATE_LIMITS_MRP, 
                               control_limits=CONTROL_LIMITS_TORQUE,
                               dynamics_params = DYNAMICS_PARAMETERS,
                               nx = NX,
                               nu = NU,)

    



    # Generate trajectories using the loaded model
    trajectories, keys = n_rollouts_controller(env,
                                               mpc_controller,
                                               subkey,
                                               horizon=MPC_HORIZON,
                                               max_ep_steps=MAX_EP_STEPS,
                                               n_rollouts=100,
                                               replan_freq=1)
    print("Calculated trajectories")
    # rewards = trajectories[-1]
    # returns = rewards.sum(axis=1)

    # print("returns per rollout:", returns)
    # print("mean return:", returns.mean())
    # print("median return:", jnp.median(returns))
    # print("min/max:", returns.min(), returns.max())

    # large_positive_counts = (rewards > 20.0).sum(axis=1)
    # print("large positive events per rollout:", large_positive_counts)
    metrics = compute_metrics(trajectories)
    print_metrics(metrics)


if __name__=="__main__":

    main()
    