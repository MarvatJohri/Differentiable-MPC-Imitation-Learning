"""

Test file for the diff mpc controller


"""
import os


import jax
import jax.numpy as jnp

from configs import ExpConfig, SpacecraftEnvConfig, DaggerHyperparameters, ControllerConfig
from mj_utils import load_controller, save_controller, make_dummy_expert, compute_metrics,compute_metrics_multi, plot_metrics_bar, print_metrics, plot_metrics_comparison
from simulation_env_jax import SpacecraftEnvJax
from diffmpc_controller import DiffMPCController, FeedForwardNetwork



exp_config = ExpConfig()
env_config = SpacecraftEnvConfig()
hyperparameters = DaggerHyperparameters()
controller_config = ControllerConfig()


# Load Hyperparameters

PPO_BASE_SAVE_PATH = exp_config.ppo_base_save_path
PPO_BASE_LOG_PATH = exp_config.ppo_base_log_path

DAGGER_BASE_SAVE_PATH = exp_config.dagger_base_save_path

PPO_EXPERIMENT_NAME = exp_config.ppo_experiment_name
DAGGER_EXPERIMENT_NAME = exp_config.dagger_experiment_name
DAGGER_EXPERIMENT_NOTES = exp_config.dagger_experiment_notes



SAVE_PATH = os.path.join(DAGGER_BASE_SAVE_PATH, DAGGER_EXPERIMENT_NAME)
EVAL_SAVE_PATH = os.path.join(SAVE_PATH, "eval_results")







RESUME = exp_config.resume_dagger_training

SEED = exp_config.seed


LOAD_FROM_CHECKPOINT = False
LOAD_ITERATION = 0

if LOAD_FROM_CHECKPOINT:
    MODEL_FILE = os.path.join(SAVE_PATH, f"model_{LOAD_ITERATION}_steps.eqx")
else:
    MODEL_FILE = os.path.join(SAVE_PATH, f"final_model.eqx")





DYNAMICS_PARAMS = env_config.spacecraft_dynamics_parameters
DT = env_config.dt
DYN_NOISE_STD = env_config.dyn_noise_std
MAX_EP_STEPS = env_config.max_ep_steps
STATE_LIMITS = env_config.state_limits
STATE_LIMITS_MRP = env_config.state_limits_mrp
MAX_TORQUE = env_config.max_torque
CONTROL_LIMITS = env_config.control_limits
CONTROL_LIMITS_TORQUE = env_config.control_limits_torque
THETA_THRESHOLD = env_config.theta_threshold
OMEGA_THRESHOLD = env_config.omega_threshold
OMEGA_PENALTY = env_config.omega_penalty
ACTION_PENALTY = env_config.action_penalty
GOAL_REWARD = env_config.goal_reward
THETA_THRESHOLD_REWARD = env_config.theta_threshold_reward
THETA_STABILITY_TOL = env_config.theta_stability_tol
OMEGA_STABILITY_TOL = env_config.omega_stability_tol


LAYERS = hyperparameters.layers
ACTIVATION = hyperparameters.activation
OUTPUT_ACTIVATION = hyperparameters.output_activation



NX = controller_config.nx
NU = controller_config.nu
NET_EPS = controller_config.network_epsilon
DECOMPOSITION_TYPE = controller_config.decomposition_type
QR_OUTPUT_HORIZON = controller_config.qr_output_horizon
MPC_HORIZON = controller_config.mpc_horizon
REPLAN_FREQUENCY = controller_config.replan_frequency






def main():

    # Make env
    env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMS,
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
                            action_penalty=ACTION_PENALTY,
                            goal_reward=GOAL_REWARD)


    # rng stuff
    key = jax.random.PRNGKey(SEED)


    key, subkey = jax.random.split(key)



    # Make network
    network = FeedForwardNetwork(input_size=NX, 
                                 output_size=NU, 
                                 key=subkey,
                                 layers=LAYERS, 
                                 activation=ACTIVATION, 
                                 output_activation=OUTPUT_ACTIVATION,
                                 qr_output_horizon=QR_OUTPUT_HORIZON,
                                 epsilon=NET_EPS,
                                 decomposition_type=DECOMPOSITION_TYPE)


    # Make controller
    controller = DiffMPCController(network=network,
                                   mpc_horizon=MPC_HORIZON,
                                   dt = DT,
                                   state_limits=STATE_LIMITS_MRP,
                                   control_limits=CONTROL_LIMITS_TORQUE,
                                   DYNAMICS_PARAMS=DYNAMICS_PARAMS)

    # Load controller 
    controller = load_controller(controller, DAGGER_BASE_SAVE_PATH, DAGGER_EXPERIMENT_NAME, SEED, RESUME)


    # Make dummy expert
    expert = make_dummy_expert(env, controller, key)