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

from ppo_train_purejax import load_model_state

from rl_utils.running_mean_std import RunningMeanStd, init_obs_rms, init_ret_rms, update_rms, normalize_obs

from matplotlib import pyplot as plt


HERE = Path(__file__).parent
MJ_WORK = HERE.parent
sys.path.append(str(MJ_WORK))

from configs import ExpConfig, SpacecraftEnvConfig, PPOHyperparameters, get_env
from mj_utils import compute_metrics, print_metrics

from rl_utils.generate_trajs import n_rollouts_policy
from rl_utils.plotting_and_evaluate import (
    extract_trajectory_data,
    plot_rewards,
    plot_control_trajectories,
    plot_cost_weights,
)



from ppo_model import ActorCritic


def get_policy(model: ActorCritic, obs_rms):

    def policy_fn(obs):
        obs = normalize_obs(obs, obs_rms)
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
        "BATCH_SIZE": hyperparams.minibatch_size,
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
    MODEL_STATS_PATH = os.path.join(SAVE_PATH, "final_model_stats.eqx")
    print("loading model ", MODEL_SAVE_PATH)
    
    checkpoint_file = MODEL_SAVE_PATH
    model_stats_file = MODEL_STATS_PATH
    model, obs_rms, ret_rms = load_model_state(model, MODEL_SAVE_PATH, MODEL_STATS_PATH)

    # Get policy function
    ppo_policy = get_policy(model, obs_rms)

    # Load env
    # env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMETERS,
    #                         dt=DT,
    #                         max_ep_steps=MAX_EP_STEPS,
    #                         state_limits=STATE_LIMITS,
    #                         control_limits=CONTROL_LIMITS,
    #                         max_torque=MAX_TORQUE,
    #                         dyn_noise_std=DYN_NOISE_STD,
    #                         theta_threshold=THETA_THRESHOLD,
    #                         omega_threshold=OMEGA_THRESHOLD,
    #                         theta_threshold_reward=THETA_THRESHOLD_REWARD,
    #                         omega_penalty=OMEGA_PENALTY,
    #                         omega_fail_penalty=OMEGA_FAIL_PENALTY,
    #                         action_penalty=ACTION_PENALTY,
    #                         goal_reward=GOAL_REWARD)

    env = get_env(env_name="spacecraft_env_noisy")


    

    # Generate trajectories using the loaded model
    trajectories, keys = n_rollouts_policy(env=env, 
                                           policy=ppo_policy,
                                           key=key,
                                           max_ep_steps=MAX_EP_STEPS,
                                           n_rollouts=100)

    metrics = compute_metrics(trajectories)
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
    plt.show()



if __name__=="__main__":

    main()
    