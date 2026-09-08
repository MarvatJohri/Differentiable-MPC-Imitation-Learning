"""

Code for Imitation Learning agent implemented using differentiable MPC

"""

from pathlib import Path
import sys
import os
from datetime import datetime
import time
from typing import Callable, Dict, List, Tuple
import json
import logging




# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PPO_BASE_SAVE_PATH = str(HERE / "ppo_results")
PPO_BASE_LOG_PATH = str(HERE / "ppo_logs")

DAGGER_BASE_SAVE_PATH = str(HERE / "dagger_results")
# IL_BASE_LOG_PATH = str(HERE / "imitation_logs")

# Ensure directories exist

os.makedirs(DAGGER_BASE_SAVE_PATH, exist_ok=True)
# os.makedirs(IL_BASE_LOG_PATH, exist_ok=True)

# URANUS_MPC_PATH = str((ROOT / "uranus-mpc").resolve())
# sys.path.append(URANUS_MPC_PATH)

# from utils.propagate import TrajectoryGenerator
# from dynamics.spacecraft_dynamics import SpacecraftDynamics
# from dynamics.planetary_params import Earth, Uranus
# from utils.learning import load_model








from marimo import state
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# from diff_mpc_functions import *
import time
import numpy as np

import equinox as eqx
import optax 

from sbx import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

# Import my stuff

from simulation_env_simpler import SpacecraftEnv
from replay_buffer import ReplayBuffer, init_buffer, add_trajectories_to_buffer, sample_from_buffer, can_sample_buffer
from mj_utils import make_dummy_controller, make_dummy_expert, compute_metrics, plot_metrics_bar, print_metrics, compute_metrics_multi, plot_metrics_comparison
# from propagate_functions import sample_episode_context, generate_trajectory
from diffmpc_controller import DiffMPCController, FeedForwardNetwork, build_mpc_solver
from simulation_env_jax import SpacecraftEnvJax, generate_trajectory, generate_n_trajectories
from config import configs











# TODO: Consider making this packaage more modular, 
# with separate files for the network, the MPC solver, and the agent class.
# Consider writing a config file to contain all hyperparams
# utils file to contain utility functions like tree_add, tree_div, etc.


# Experiment params
PPO_EXPERIMENT_NAME = "spacecraft_ppo_omega_hard_limit_test"
DAGGER_EXPERIMENT_NAME = "spacecraft_ppo_imitation_dagger_v1_torque_only_experiment1"
DAGGER_EXPERIMENT_NOTES = "Initial imitation learning on Earth orbit"


DYNAMICS_PARAMS = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
DYNAMICS_PARAMS["inertia_inv"] = jnp.linalg.inv(DYNAMICS_PARAMS["inertia"])

# Environment params
DT = 0.1                         # Simulation timestep
DYN_NOISE_STD = 1e-6             # Dynamics noise
MAX_EP_STEPS = 1500         # Max episode length


# State/action limits
STATE_LIMITS = [[-1, 1]] * 4 + [[-2, 2]] * 3  # [quat, omega]
STATE_LIMITS_MRP = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
CONTROL_LIMIT_SCALE = 1        # Scales [-1, 1] control limits
MAX_TORQUE = 5e-5
CONTROL_LIMITS = jnp.array([[-CONTROL_LIMIT_SCALE, CONTROL_LIMIT_SCALE]] * 3, dtype=jnp.float64) # [normalized torque]
CONTROL_LIMITS_TORQUE = jnp.array([[-MAX_TORQUE, MAX_TORQUE]] * 3, dtype=jnp.float64) # [torque]

# Reward shaping
THETA_THRESHOLD = np.deg2rad(15.0)  # Convert to radians
OMEGA_THRESHOLD = np.deg2rad(5.0)                 # Angular velocity tolerance (rad/s)
OMEGA_PENALTY = 0.5               # Penalty weight for omega error
ACTION_PENALTY = 0.1              # Penalty weight for action magnitude
GOAL_REWARD = 50.0              # Bonus for reaching goal
THETA_THRESHOLD_REWARD = 10.0        # Bonus for staying within theta threshold



# Imitation learning hyperparameters
LEARNING_RATE = 3e-4
LEARNING_RATE_FINAL = 1e-5  
LEARNING_RATE_SCHEDULE = "constant"  # constant, linear, cosine annealing
BATCH_SIZE = 512

BETA_DECAY = 0.95                       # Beta decay for DAgger
NUM_EPS_STORED = 100
MAX_BUFFER_SIZE = MAX_EP_STEPS * NUM_EPS_STORED                # Replay buffer size
NUM_ITERATIONS = 100
NUM_TRAJECTORIES = 10
NUM_GRADIENT_STEPS = 100

# Imitation Learning Architecture
LAYERS = [256, 256]                  # Hidden layers for the neural network
# TOTAL_TIMESTEPS = 1_000_000
ACTIVATION = "relu"
OUTPUT_ACTIVATION = "tanh"
NETWORK_EPSILON = 1e-3
DECOMPOSITION_TYPE = "diagonal"  # diagonal, full, cholesky
QR_OUTPUT_HORIZON = 1
MPC_HORIZON = 10


LOG_EVERY = 1
CHECKPOINT_FREQUENCY = 10

EVAL_FREQUENCY = 10
NUM_EVAL_EPS = 100



# RNG Seed
SEED = 42



# =============================================================================
# DERIVED PATHS (don't edit)
# =============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RL_SAVE_PATH = os.path.join(PPO_BASE_SAVE_PATH, PPO_EXPERIMENT_NAME)
SAVE_PATH = os.path.join(DAGGER_BASE_SAVE_PATH, DAGGER_EXPERIMENT_NAME)
FIGURES_PATH = os.path.join(SAVE_PATH, "figures")
os.makedirs(FIGURES_PATH, exist_ok=True)
# LOG_PATH = os.path.join(SAVE_PATH, "logs")
LOG_FILENAME = os.path.join(SAVE_PATH,"logs")
CHECKPOINT_PATH = os.path.join(SAVE_PATH, "checkpoints")
# TENSORBOARD_PATH = os.path.join(LOG_PATH, "tensorboard")

# Resume
RESUME_TRAINING = False
RESUME_ITERATIONS = 5
RESUME_CHECKPOINT_FILE = os.path.join(CHECKPOINT_PATH, f"model_{RESUME_ITERATIONS}_steps.eqx")

os.makedirs(SAVE_PATH, exist_ok=True)
# os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# os.makedirs(TENSORBOARD_PATH, exist_ok=True)


# RED ALERT

# CONTROLLER NOMINAL TRAJECTORIES PROBABLY DO NEED NOISE FOR PLANNING



def make_env(dynamics_params: Dict, seed: int = None):
    """Create and wrap environment."""
    env = SpacecraftEnv(
        dynamics_params=dynamics_params,
        dt=DT,
        max_ep_steps=MAX_EP_STEPS,
        state_limits=np.array(STATE_LIMITS),
        control_limits=CONTROL_LIMIT_SCALE * np.array([[-1, 1]] * 3),
        max_torque=MAX_TORQUE,
        dyn_noise_std=DYN_NOISE_STD,
        theta_threshold=THETA_THRESHOLD,
        omega_threshold=OMEGA_THRESHOLD,
        theta_threshold_reward=THETA_THRESHOLD_REWARD,
        omega_penalty=OMEGA_PENALTY,
        action_penalty=ACTION_PENALTY,
        goal_reward=GOAL_REWARD,
    )
    
    # Wrap with Monitor for episode logging
    env = Monitor(env)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env




def setup_logging(log_dir: str, log_filename: str = "training.log") -> logging.Logger:
    """Setup logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(DAGGER_EXPERIMENT_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Format
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger




def get_config() -> dict:
    """Return all config as a dictionary for saving."""
    return {
        "experiment": {
            "name": DAGGER_EXPERIMENT_NAME,
            "notes": DAGGER_EXPERIMENT_NOTES,
            "timestamp": TIMESTAMP,
        },
        "environment": {
            "dt": DT,
            "max_episode_length": MAX_EP_STEPS,
            "dyn_noise_std": DYN_NOISE_STD,
            "state_limits_quat": STATE_LIMITS if STATE_LIMITS is not None else None,
            "state_limits_mrp": STATE_LIMITS_MRP.tolist() if STATE_LIMITS_MRP is not None else None,
            "control_limits": CONTROL_LIMITS.tolist() if CONTROL_LIMITS is not None else None,
            "control_limits_torque": CONTROL_LIMITS_TORQUE.tolist() if CONTROL_LIMITS_TORQUE is not None else None,
            "theta_threshold": THETA_THRESHOLD,
            "omega_threshold": OMEGA_THRESHOLD,
            "theta_threshold_reward": THETA_THRESHOLD_REWARD,
            "omega_penalty": OMEGA_PENALTY,
            "action_penalty": ACTION_PENALTY,
            "goal_reward": GOAL_REWARD,
        },
        "dagger": {
            "num_iterations": NUM_ITERATIONS,
            "num_trajectories": NUM_TRAJECTORIES,
            "num_gradient_steps": NUM_GRADIENT_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "beta_decay": BETA_DECAY,
            "max_buffer_size": MAX_BUFFER_SIZE,
        },
        "network": {
            "layers": LAYERS,
        },
        "mpc": {
            "horizon": MPC_HORIZON,
        },
        "training": {
            "seed": SEED,
            "log_every": LOG_EVERY,
            "checkpoint_frequency": CHECKPOINT_FREQUENCY,
        },
        "paths": {
            "expert_model": RL_SAVE_PATH,
            "checkpoint_dir": CHECKPOINT_PATH,
        },
        "resume": {
            "resumed": RESUME_TRAINING,
            "resumed_from": RESUME_CHECKPOINT_FILE,
        },
    }


def save_config(config: dict, path: str):
    """Save config to JSON file."""
    os.makedirs(path, exist_ok=True)
    config_file = os.path.join(path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Config saved to: {config_file}")







def load_il_model(controller: DiffMPCController,
                    replay_buffer: ReplayBuffer,
                    opt_state: optax.OptState,
                    checkpoint_file: str):


    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")

    checkpoint = {
            "iteration": jnp.array(0),
            "network_params": eqx.filter(controller.network, eqx.is_array),
            "beta": jnp.array(1.0),
            "replay_buffer": replay_buffer,
            "replay_buffer_max_size": jnp.array(replay_buffer.max_size),
            "opt_state": opt_state,
            "key": jax.random.PRNGKey(0),
        }

    checkpoint = eqx.tree_deserialise_leaves(checkpoint_file, checkpoint)

    # controller = eqx.tree_at(lambda c: c.network, controller, checkpoint["network_params"])
    network_params = checkpoint["network_params"]
    static_network = eqx.filter(controller.network, lambda x: not eqx.is_array(x))
    new_network = eqx.combine(network_params, static_network)
    # controller = eqx.tree_at(lambda c: eqx.filter(c.network, eqx.is_array), controller, network_params)
    controller = eqx.tree_at(lambda c: c.network, controller, new_network)

    replay_buffer = checkpoint["replay_buffer"]
    replay_buffer = replay_buffer.replace(max_size=int(checkpoint["replay_buffer_max_size"]))
    opt_state = checkpoint["opt_state"]
    beta = checkpoint["beta"]
    iteration = checkpoint["iteration"]
    key = checkpoint["key"]

    print(f"Checkpoint loaded from {checkpoint_file} at iteration {iteration}")

    return controller, replay_buffer, opt_state, beta, iteration, key



def save_il_model(controller: DiffMPCController,
                    replay_buffer: ReplayBuffer,
                    opt_state: optax.OptState,
                    beta: float,
                    iteration: int,
                    key: jax.random.PRNGKey,
                    checkpoint_path: str,
                    final: bool = False
                    ):


    os.makedirs(checkpoint_path, exist_ok=True)

    if final:
        checkpoint_file = os.path.join(checkpoint_path, f"final_model.eqx")
    else:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(checkpoint_path, f"model_{iteration}_steps.eqx")

    checkpoint = {
        "iteration": jnp.array(iteration),
        "network_params": eqx.filter(controller.network, eqx.is_array),
        "beta": jnp.array(beta),
        "replay_buffer": replay_buffer,
        "replay_buffer_max_size": jnp.array(replay_buffer.max_size),
        "opt_state": opt_state,
        "key": key,
    }


    eqx.tree_serialise_leaves(checkpoint_file, checkpoint)

    print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")





def loss_fn(controller: DiffMPCController,
            data_batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    

    """

    MSE Loss b/w predicted and expert actions

    """

    def single_forward(obs, goal_state, nominal_traj, nominal_cntrl):
        pred_action, _, _ = controller(obs, goal_state, nominal_traj, nominal_cntrl)
        return pred_action

    predicted_actions = jax.vmap(single_forward)(data_batch["obs"],
                                                 data_batch["goal_state"],
                                                 data_batch["nominal_traj"],
                                                 data_batch["nominal_cntrl"])

    predicted_actions = predicted_actions / MAX_TORQUE  # Scale predicted actions to [-1, 1] range for loss computation




    loss = jnp.mean((predicted_actions - data_batch["expert_actions"]) ** 2)      
                          

    return loss

@eqx.filter_value_and_grad
def loss_and_grad(controller: DiffMPCController,
                  data_batch: Dict[str, jnp.ndarray]):

    return loss_fn(controller, data_batch)



# def train_iteration(controller: DiffMPCController,
#                     optimizer,
#                     opt_state,)

@eqx.filter_jit
# @partial(eqx.filter_jit, static_argnames=("num_gradient_steps", "batch_size"))
def train_loop(controller: DiffMPCController,
               optimizer: optax.GradientTransformation,
               opt_state: optax.OptState,
               replay_buffer: ReplayBuffer,
               num_gradient_steps: int,
               batch_size: int,
               key: jax.random.PRNGKey):

    def scan_step(carry,_):

        controller, opt_state, key = carry
        key, subkey = jax.random.split(key)

        data_batch = sample_from_buffer(replay_buffer, subkey, batch_size)

        # loss = loss_fn(controller, data_batch)
        # grad

        loss, grad = loss_and_grad(controller, data_batch)

        # updates, opt_state = optimizer.update(grad, opt_state, controller.network)

        network_grad = grad.network

        updates, opt_state = optimizer.update(network_grad, opt_state, controller.network)

        # controller.network = eqx.apply_updates(controller.network, updates)
        new_network = eqx.apply_updates(controller.network, updates)
        controller = eqx.tree_at(lambda c: c.network, controller, new_network)

        return (controller, opt_state, key), loss



    init_carry = (controller, opt_state, key)
    (controller, opt_state, key), losses = jax.lax.scan(scan_step, init_carry, None, length=num_gradient_steps)


    return controller, opt_state, jnp.mean(losses), key



# @eqx.filter_jit
# def train_step(controller: DiffMPCController,
#                optimizer: optax.GradientTransformation,
#                opt_state: optax.OptState,
#                replay_buffer: ReplayBuffer,
#                batch_size: int,
#                key: jax.random.PRNGKey):


#     key, subkey = jax.random.split(key)
#     data_batch = sample_from_buffer(replay_buffer, subkey, batch_size)

#     loss, grad = loss_and_grad(controller, data_batch)

#     network_grad = grad.network

#     updates, opt_state = optimizer.update(network_grad, opt_state, controller.network)

#     new_network = eqx.apply_updates(controller.network, updates)
#     controller = eqx.tree_at(lambda c: c.network, controller, new_network)

#     return controller, opt_state, loss, key


@eqx.filter_jit
def train_iteration(env: SpacecraftEnvJax,
                    controller: DiffMPCController,
                    expert_policy: PPO, 
                    replay_buffer: ReplayBuffer,
                    beta: float,
                    optimizer: optax.GradientTransformation,
                    opt_state: optax.OptState,
                    key: jax.random.PRNGKey,
                    num_trajectories: int,
                    max_ep_steps: int,
                    num_gradient_steps: int,
                    batch_size: int,
                    beta_decay: float,
                    replan_frequency: int = 1):


    # Collect trajectories
    trajectories, key = generate_n_trajectories(env,
                                                controller,
                                                expert_policy,
                                                key,
                                                beta,
                                                max_ep_steps,
                                                num_trajectories,
                                                replan_frequency)
    # Debug: Check trajectory shapes before adding
    # for i, traj in enumerate(trajectories):
    #     print(f"Trajectory {i}:")
    #     print(f"  obs: {traj['obs'].shape}")
    #     print(f"  nominal_traj: {traj['nominal_traj'].shape}")
    #     print(f"  nominal_cntrl: {traj['nominal_cntrl'].shape}")



    # Add trajectories to buffer
    replay_buffer = add_trajectories_to_buffer(replay_buffer,trajectories)

    # Do training updates
    controller, opt_state, mean_loss, key = train_loop(controller, optimizer, opt_state, replay_buffer, num_gradient_steps, batch_size, key)
    # losses = []
    # for _ in range(num_gradient_steps):
    #     controller, opt_state, loss, key = train_step(controller, optimizer, opt_state, replay_buffer, batch_size, key)
    #     losses.append(loss)

    # mean_loss = jnp.mean(jnp.array(losses))

    # Update beta
    beta = beta * beta_decay

    return controller, replay_buffer, opt_state, mean_loss, beta, key


def evaluate(env: SpacecraftEnvJax,
             controller: DiffMPCController, 
             expert_policy: PPO,
             max_steps: int,
             num_episodes: int, 
             key: jax.random.PRNGKey):

    print("Evaluating Imitation Learning Agent")

    



def evaluate(controller: DiffMPCController, 
            max_ep_steps: int,
            num_episodes: int, 
            key: jax.random.PRNGKey):

    print("Evaluating Imitation Learning Agent")


    # Evaluate using the gen trajectories function

    dummy_expert = make_dummy_expert(action_dim=3)

    # Make env
    env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMS,
                            dt=DT,
                            max_ep_steps=max_ep_steps,
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

    # Generate 100 trajectories using controller and dummy expert
    start_time = time.time()
    trajectories, key = generate_n_trajectories(env=env,
                                                controller=controller,
                                                expert_policy=dummy_expert,
                                                key=key,
                                                beta=0,
                                                max_ep_steps=max_ep_steps,
                                                n_trajectories=num_episodes)

    print("Evaluation Done")
    print("Time taken: ",time.time() - start_time)

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

    # _ = system.plot_costs(trajectories, target_states, plot_stats=True)

    # system.plot_violin_and_bar(trajectories, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)


    # compute metrics
    df = compute_metrics(trajectories,
                         dt=DT,
                         angle_threshold=angle_threshold,
                         omega_threshold=omega_threshold,
                         angle_tol_stability=angle_tol,
                         omega_tol_stability=omega_tol,
                         tail_length=tail_length)

    print_metrics(df, "Evaluation Metrics")








def learn(env: SpacecraftEnvJax, 
          controller: DiffMPCController, 
          expert_policy: Callable,
          replay_buffer: ReplayBuffer,
          optimizer: optax.GradientTransformation,
          opt_state: optax.OptState,
          key: jax.random.PRNGKey,
          num_iterations: int,
          num_trajectories: int,
          max_ep_steps: int,
          num_gradient_steps: int,
          batch_size: int,
          beta_decay: float,
          log_frequency: int,
          checkpoint_frequency: int,
          checkpoint_path: str,
          logger,
          evaluate_freq: int,
          num_eval_eps: int,
          replan_frequency: int = 1,
          resume: bool = False):



    t_so_far = 0
    start_time = time.time()

    # Initialize Beta
    beta = 1

    logger.info("Starting training")

    itrs_done = 0
    if resume:
        controller, replay_buffer, opt_state, beta, iteration, key = load_il_model(controller, replay_buffer, opt_state, RESUME_CHECKPOINT_FILE)
        itrs_done = iteration
        # Print stuff
        # print(f"Resuming training from iteration {itrs_done}, beta={beta}")
        logger.info("Resuming training from iteration {itrs_done}, beta={beta}")

    for itr in range(itrs_done, num_iterations):

        itr_start_time = time.time()
        # Do a train iteration
        controller, replay_buffer, opt_state, mean_loss, beta, key = train_iteration(env,
                                                                                     controller,
                                                                                     expert_policy,
                                                                                     replay_buffer,
                                                                                     beta,
                                                                                     optimizer,
                                                                                     opt_state,
                                                                                     key,
                                                                                     num_trajectories,
                                                                                     max_ep_steps,
                                                                                     num_gradient_steps,
                                                                                     batch_size,
                                                                                     beta_decay,
                                                                                     replan_frequency=replan_frequency)


        itr_end_time = time.time()

        # Update t_so_far

        # do logging
        if itr % log_frequency == 0:

            # print(f"Iteration {itr}, Mean loss: {mean_loss}, time: {itr_end_time - start_time}")
            logger.info(f"Iteration {itr}, Mean loss: {mean_loss}, beta: {beta}, time: {itr_end_time - itr_start_time}")

        # save checkpoint

        if itr % checkpoint_frequency == 0:
            
            save_il_model(controller, replay_buffer, opt_state, beta, itr, key, checkpoint_path)

        # Evaluate
        if itr % evaluate_freq == 0:
            key, subkey = jax.random.split(key)
            evaluate(controller, max_ep_steps, num_eval_eps, subkey)

    return controller, replay_buffer, opt_state, beta, key



def get_expert_policy(expert_policy: PPO, vec_env: VecNormalize):


    # Extract the expert policy's actor module and parameters
    actor_module = expert_policy.policy.actor
    actor_params = expert_policy.policy.actor_state.params

    # Extract the vector normalization statistics
    obs_mean = jnp.array(vec_env.obs_rms.mean, dtype=jnp.float32)
    obs_var = jnp.array(vec_env.obs_rms.var, dtype=jnp.float32)
    obs_count = vec_env.obs_rms.count
    obs_eps = vec_env.epsilon
    obs_clip = vec_env.clip_obs

    def expert_policy_fn(obs: jnp.ndarray) -> jnp.ndarray:
        # Cast down to float32 for the actor module
        obs = obs.astype(jnp.float32)

        # Normalize the observation
        normalized_obs = (obs - obs_mean) / jnp.sqrt(obs_var + obs_eps)
        normalized_obs = jnp.clip(normalized_obs, -obs_clip, obs_clip)

        # Pass through the actor module to get the action
        dist = actor_module.apply(actor_params, normalized_obs[None, :])
        action = dist.mode()[0]  # Get the mode of the distribution and remove the batch dimension
        action = jnp.clip(action, -1.0, 1.0)  # Ensure action is within [-1, 1]
        return action.astype(jnp.float64)  # Cast back to float64 for consistency


    return expert_policy_fn






def main():

    # Save config
    config = get_config()
    save_config(config,SAVE_PATH)

    # Make the env
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



    # Not changing defaults for now (don't really need to besides ep length)
    # env = SpacecraftEnv()
    dummy_vec_env = DummyVecEnv([lambda: make_env(dynamics_params=DYNAMICS_PARAMS, seed=None)])
    dummy_vec_env = VecNormalize.load(os.path.join(RL_SAVE_PATH, "vecnormalize_stats.pkl"), dummy_vec_env)
    dummy_vec_env.training = False
    dummy_vec_env.norm_reward = False

    # set rng
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)

    # Initialize model
    network = FeedForwardNetwork(nx=7, 
                               nu=3, 
                               key=subkey, 
                               layers=LAYERS, 
                               activation=ACTIVATION, 
                               output_activation=OUTPUT_ACTIVATION, 
                               qr_output_horizon=QR_OUTPUT_HORIZON, 
                               eps=NETWORK_EPSILON, 
                               decomposition_type=DECOMPOSITION_TYPE)


    # Initialize controller
    controller = DiffMPCController(network,MPC_HORIZON,DT,STATE_LIMITS_MRP,CONTROL_LIMITS_TORQUE, DYNAMICS_PARAMS)

    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(controller.network,eqx.is_array))



    # Initialize replay buffer
    replay_buffer = init_buffer(max_size=MAX_BUFFER_SIZE,
                                obs_dim=network.nx,
                                state_dim=network.nx,
                                action_dim=network.nu,
                                horizon=MPC_HORIZON)

    # Load expert policy
    rl_path = os.path.join(RL_SAVE_PATH, "final_model.zip")
    rl_model = PPO.load(rl_path, env=dummy_vec_env)

    expert_policy = get_expert_policy(rl_model, dummy_vec_env)


    # Setup logger stuff
    logger = setup_logging(SAVE_PATH,LOG_FILENAME)

    # Learn stuff
    # key, subkey = jax.random.split(key)
    controller, replay_buffer, opt_state, beta, key = learn(env,
                                                            controller,
                                                            expert_policy,
                                                            replay_buffer,
                                                            optimizer,
                                                            opt_state,
                                                            key,
                                                            num_iterations=NUM_ITERATIONS,
                                                            num_trajectories=NUM_TRAJECTORIES,
                                                            max_ep_steps=MAX_EP_STEPS,
                                                            num_gradient_steps=NUM_GRADIENT_STEPS,
                                                            batch_size=BATCH_SIZE,
                                                            beta_decay=BETA_DECAY,
                                                            log_frequency=LOG_EVERY,
                                                            checkpoint_frequency=CHECKPOINT_FREQUENCY,
                                                            checkpoint_path=CHECKPOINT_PATH,
                                                            logger=logger,
                                                            resume=RESUME_TRAINING,
                                                            evaluate_freq=EVAL_FREQUENCY,
                                                            num_eval_eps=NUM_EVAL_EPS)



    # model, optimizer_state, beta, replay_buffer, t_so_far, key = learn(env, model, expert_policy, TOTAL_TIMESTEPS, optimizer, key)

    # Save the final model and optimizer state
    save_il_model(controller, replay_buffer, opt_state, beta, NUM_ITERATIONS, key, SAVE_PATH)

    # # Save the model and optimizer state
    # model_save_path = os.path.join(SAVE_PATH, "final_model.eqx")
    # optimizer_state_save_path = os.path.join(SAVE_PATH, "final_optimizer_state.eqx")

    # eqx.tree_serialise_leaves(model_save_path, model)
    # eqx.tree_serialise_leaves(optimizer_state_save_path, optimizer_state)

    # Do evaluations
    evaluate(controller, max_ep_steps=MAX_EP_STEPS, num_episodes=100, key=key)


def dry_test():


    # Some test code to check how the model works with no training

    # Make the env
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



    # Not changing defaults for now (don't really need to besides ep length)
    # env = SpacecraftEnv()
    dummy_vec_env = DummyVecEnv([lambda: make_env(dynamics_params=DYNAMICS_PARAMS, seed=None)])
    dummy_vec_env = VecNormalize.load(os.path.join(RL_SAVE_PATH, "vecnormalize_stats.pkl"), dummy_vec_env)
    dummy_vec_env.training = False
    dummy_vec_env.norm_reward = False
    

    # Initialize model
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    network = FeedForwardNetwork(nx=7, 
                               nu=3, 
                               key=subkey, 
                               layers=LAYERS, 
                               activation='relu', 
                               output_activation='tanh', 
                               qr_output_horizon=1, 
                               eps=1e-3, 
                               decomposition_type='diagonal')

    controller = DiffMPCController(network,
                                    mpc_horizon=10,
                                    dt=DT,
                                    state_limits=STATE_LIMITS_MRP,
                                    control_limits=CONTROL_LIMITS_TORQUE,
                                    dynamics_params=DYNAMICS_PARAMS)


    # Test evaluate function

    print("Testing evaluate function with dummy expert and controller")

    # evaluate(controller, max_steps=MAX_EPISODE_LENGTH, num_episodes=10, key=key)

    # Get expert policy
    rl_path = os.path.join(RL_SAVE_PATH, "final_model.zip")
    rl_model = PPO.load(rl_path, env=dummy_vec_env)
    expert_policy = get_expert_policy(rl_model, dummy_vec_env)


    # evaluate(controller, max_steps = env.num_steps, num_episodes=10, key=key)

    dummy_expert = make_dummy_expert(action_dim=3)
    dummy_controller = make_dummy_controller(state_dim=7, action_dim=3)

    # Generate 10 trajs using controller and dummy expert
    controller_trajectories, keys = generate_n_trajectories(env=env,
                                                            controller=controller,
                                                            expert_policy=dummy_expert,
                                                            key=key,
                                                            beta=0,
                                                            max_ep_steps=MAX_EP_STEPS,
                                                            n_trajectories=100)

    # Generate 10 trajs using controller and rl expert
    rl_trajectories, keys = generate_n_trajectories(env=env,
                                                    controller=dummy_controller,
                                                    expert_policy=expert_policy,
                                                    key=key,
                                                    beta=1,
                                                    max_ep_steps=MAX_EP_STEPS,
                                                    n_trajectories=100)

    # compute metrics


    controller_df = compute_metrics(controller_trajectories)
    rl_df = compute_metrics(rl_trajectories)

    print_metrics(controller_df, "Controller with Dummy Expert")
    print_metrics(rl_df, "Controller with RL Expert")

    # Plot stuff
    plot_metrics_bar(controller_df, "Controller with Dummy Expert","dummy_model", FIGURES_PATH)
    plot_metrics_bar(rl_df, "Controller with RL Expert", "rl_model", FIGURES_PATH)

    trajs = [controller_trajectories, rl_trajectories]
    labels = ["Controller with Dummy Expert", "Controller with RL Expert"]

    # df = compute_metrics_multi(trajs, labels)

    dfs = [controller_df, rl_df]

    plot_metrics_comparison(dfs, labels, "Comparison of Controller with Dummy Expert and RL Expert", "comparison_plot", FIGURES_PATH)




def test_jax_env():
    # Test the JAX environment wrapper
    env = SpacecraftEnvJax()

    # Make the env
    # Not changing defaults for now (don't really need to besides ep length)
    # env = SpacecraftEnv()
    vec_env = DummyVecEnv([lambda: make_env(dynamics_params=DYNAMICS_PARAMS, seed=None)])
    vec_env = VecNormalize.load(os.path.join(RL_SAVE_PATH, "vecnormalize_stats.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # set rng
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)

    # Initialize model
    network = FeedForwardNetwork(nx=7, 
                                nu=3, 
                                key=subkey, 
                                layers=LAYERS, 
                                activation=ACTIVATION, 
                                output_activation=OUTPUT_ACTIVATION, 
                                qr_output_horizon=QR_OUTPUT_HORIZON, 
                                eps=NETWORK_EPSILON, 
                                decomposition_type=DECOMPOSITION_TYPE)


    # Initialize controller
    controller = DiffMPCController(network,MPC_HORIZON,DT,STATE_LIMITS_MRP,CONTROL_LIMITS_TORQUE,DYNAMICS_PARAMS)

    # rl policy
    rl_path = os.path.join(RL_SAVE_PATH, "final_model.zip")
    rl_model = PPO.load(rl_path, env=vec_env)
    expert_policy = get_expert_policy(rl_model, vec_env)

    start = time.time()
    trajectory, key = generate_trajectory(env=env,
                                          controller=controller,
                                          expert_policy=expert_policy,
                                          key=key,
                                          beta=0,
                                          max_ep_steps=MAX_EP_STEPS)
    end = time.time()

    print("Time take for generating one trajectory: ", end-start)

    start = time.time()
    trajectory, key = generate_trajectory(env=env,
                                            controller=controller,
                                            expert_policy=expert_policy,
                                            key=key,
                                            beta=0,
                                            max_ep_steps=MAX_EP_STEPS)
    end = time.time()
    print("Time take for generating one trajectory AFTER jit compiling: ", end-start)


    start = time.time()
    trajectories, keys = generate_n_trajectories(env=env,
                                                    controller=controller,
                                                    expert_policy=expert_policy,
                                                    key=key,
                                                    beta=1.0,
                                                    max_ep_steps=MAX_EP_STEPS,
                                                    n_trajectories=100)
    end = time.time()
    print("Time take for generating 10 trajectories AFTER jit compiling using vmap: ", end-start)


    start = time.time()
    # key = keys[0]
    trajectories, keys = generate_n_trajectories(env=env,
                                                    controller=controller,
                                                    expert_policy=expert_policy,
                                                    key=key,
                                                    beta=1.0,
                                                    max_ep_steps=MAX_EP_STEPS,
                                                    n_trajectories=100)
    end = time.time()
    print("Time take for generating 10 trajectories AFTER jit compiling using vmap AFTER vmap jit compiles: ", end-start)


    # Also test collecting trajectories for training
    # key = keys[0]
    start = time.time()
    trajectories, key = collect_trajectory(env=vec_env,
                                            expert_policy=rl_model,
                                            controller=controller,
                                            max_episode_length=MAX_EP_STEPS,
                                            beta=0.5,
                                            key=key)
    end = time.time()
    print("Time take for collecting one trajectory not using jax stuff: ", end-start)

    
    





if __name__ == "__main__":


    main()

    # test_jax_env()

    # dry_test()
