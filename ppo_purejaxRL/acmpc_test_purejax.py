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
from configs import ExpConfig, SpacecraftEnvConfig, ACMPCHyperparameters
from mj_utils import make_dummy_controller, compute_metrics, print_metrics

from acmpc_model import ActorCriticMPC, save_network, load_network, get_acmpc_policy


def rollout_controller(env: SpacecraftEnvJax,
                       controller,
                       key: jax.random.PRNGKey,
                       horizon: int,
                       max_ep_steps: int,
                       replan_freq: int = 1,
                       nx: int = 7,
                       nu: int = 3):


    """
    
    Function specifically for MPC type controller(s)
    of form controller(obs, goal_state, nominal_traj, nominal_cntrl) -> action, new_nominal_traj, new_nominal_cntrl
    
    """


    max_torque = env.max_torque

    def scan_step(carry, _):

        env_state, obs, i, nominal_traj, nominal_cntrl, key = carry


        def controller_wrapper(operand):
            return controller(*operand)

        def no_update(operand):

            _, __, nominal_traj, nominal_cntrl = operand

            # Shift nominal trajectory and control by one step
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]

            return action, nominal_traj, nominal_cntrl

        # Get controller action
        action, new_nominal_traj, new_nominal_cntrl = jax.lax.cond(
            i%replan_freq == 0,
            controller_wrapper,
            no_update,
            operand=(obs, env_state.goal_state, nominal_traj, nominal_cntrl))

        # Normalize controller action
        action = action / max_torque

        # Extract Q, R from controller for debugging
        # Q, R = controller.Q, controller.R


        # Run executed action on the environment using the step key inside the env_state
        new_env_state, new_obs, reward, done, info = env.step(env_state, action)

        new_carry = (new_env_state, new_obs, i+1, new_nominal_traj, new_nominal_cntrl, key)

        # outputs = (obs, action*max_torque, Q, R)
        outputs = (obs, action*max_torque, reward)

        return new_carry, outputs
            


    # Env
    key, subkey = jax.random.split(key)
    # seed = jax.random.randint(subkey, shape=(), minval=0, maxval=2**32 - 1)
    # init_env_state, init_obs, info = env.reset(seed)
    init_env_state, init_obs, info = env.reset(subkey)

    # Generate initial nominal trajectories
    key, subkey = jax.random.split(key)
    nominal_traj = jnp.tile(init_env_state.state, (horizon + 1, 1))
    nominal_cntrl = 1e-8 * jax.random.normal(subkey, shape=(horizon, nu), dtype=jnp.float64)

    init_carry = (init_env_state, init_obs, 0, nominal_traj, nominal_cntrl, key)

    # Do the for loop
    final_carry, outputs = jax.lax.scan(scan_step, init_carry, xs=None, length=max_ep_steps)

    final_env_state, final_obs, final_i, final_nominal_traj, final_nominal_cntrl, final_key = final_carry
    # observations, actions, Q_list, R_list = outputs
    observations, actions, rewards = outputs

    observations = jnp.concatenate([observations, final_obs[None, :]], axis=0)

    # nominal_trajs = jnp.concatenate([nominal_trajs, final_nominal_traj[None, :]], axis=0)
    # nominal_cntrls = jnp.concatenate([nominal_cntrls, final_nominal_cntrl[None, :]], axis=0)

    # trajectory = (observations, init_env_state.goal_state, actions, Q_list, R_list)
    trajectory = (observations, init_env_state.goal_state, actions, rewards)

    return trajectory, final_key


def n_rollouts_controller(env: SpacecraftEnvJax,
                      controller,
                      key: jax.random.PRNGKey,
                      horizon: int,
                      max_ep_steps: int,
                      n_rollouts: int,
                      replan_freq: int = 1):

    key, subkey = jax.random.split(key)
    batched_keys = jax.random.split(subkey, n_rollouts)
    trajectories, _ = jax.vmap(rollout_controller, in_axes=(None, None, 0, None, None, None))(env, 
                                                                                              controller, 
                                                                                              batched_keys, 
                                                                                              horizon, 
                                                                                              max_ep_steps, 
                                                                                              replan_freq)
    return trajectories, key



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
    model = load_network(model, MODEL_SAVE_PATH)


    # Get policy function
    acmpc_policy = get_acmpc_policy(model)

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
    trajectories, keys = n_rollouts_controller(env,
                                               acmpc_policy,
                                               key,horizon=config["MPC_HORIZON"],
                                               max_ep_steps=MAX_EP_STEPS,
                                               n_rollouts=100,
                                               replan_freq=1)
    print("Calculated trajectories")
    rewards = trajectories[-1]
    returns = rewards.sum(axis=1)

    print("returns per rollout:", returns)
    print("mean return:", returns.mean())
    print("median return:", jnp.median(returns))
    print("min/max:", returns.min(), returns.max())

    large_positive_counts = (rewards > 20.0).sum(axis=1)
    print("large positive events per rollout:", large_positive_counts)
    metrics = compute_metrics(trajectories)
    print_metrics(metrics)


if __name__=="__main__":

    main()
    