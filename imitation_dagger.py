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

RL_BASE_SAVE_PATH = str(HERE / "results")
RL_BASE_LOG_PATH = str(HERE / "logs")

IL_BASE_SAVE_PATH = str(HERE / "imitation_results")
# IL_BASE_LOG_PATH = str(HERE / "imitation_logs")

# Ensure directories exist

os.makedirs(IL_BASE_SAVE_PATH, exist_ok=True)
# os.makedirs(IL_BASE_LOG_PATH, exist_ok=True)

URANUS_MPC_PATH = str((ROOT / "uranus-mpc").resolve())
sys.path.append(URANUS_MPC_PATH)

from utils.propagate import TrajectoryGenerator
from dynamics.spacecraft_dynamics import SpacecraftDynamics
from dynamics.planetary_params import Earth, Uranus
from utils.learning import load_model








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
from mj_utils import network_output_to_QR
from propagate_functions import sample_episode_context, generate_trajectory, get_state_error, rk4_step
from diffmpc_controller import DiffMPCController, FeedForwardNetwork, build_mpc_solver



# TODO: Consider making this packaage more modular, 
# with separate files for the network, the MPC solver, and the agent class.
# Consider writing a config file to contain all hyperparams
# utils file to contain utility functions like tree_add, tree_div, etc.


# Experiment params
RL_EXPERIMENT_NAME = "spacecraft_ppo_v1_torque_only"
EXPERIMENT_NAME = "spacecraft_ppo_imitation_dagger_v1_torque_only_experiment1"
EXPERIMENT_NOTES = "Initial imitation learning on Earth orbit"

# Environment params
PLANET = Earth                 # "earth" or "uranus"
DT = 0.1                         # Simulation timestep
N_ORBIT = 5000                   # Orbit trajectory length
DYN_NOISE_STD = 1e-6             # Dynamics noise
MAX_EPISODE_LENGTH = 1500         # Max episode length


# State/action limits
STATE_LIMITS = [[-1, 1]] * 4 + [[-2, 2]] * 3  # [quat, omega]
STATE_LIMITS_MRP = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
CONTROL_LIMIT_SCALE = 1        # Scales [-1, 1] control limits
MAX_TORQUE = 5e-5
CONTROL_LIMITS = jnp.array([[-MAX_TORQUE, MAX_TORQUE]] * 3, dtype=jnp.float64) # [torque]

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


# MPC Parameters
REPLAN_FREQ = 1
NOISE_STD = 1e-6


# Imitation learning hyperparameters
LEARNING_RATE = 1e-4
LEARNING_RATE_FINAL = 1e-4  
LEARNING_RATE_SCHEDULE_TYPE = "constant"  # constant, linear, cosine annealing
BATCH_SIZE = 256

BETA_DECAY = 0.99                       # Beta decay for DAgger
NUM_EPS_STORED = 100
MAX_BUFFER_SIZE = MAX_EPISODE_LENGTH * NUM_EPS_STORED                # Replay buffer size
NUM_ITERATIONS = 50
NUM_TRAJECTORIES = 10
NUM_GRADIENT_STEPS = 50

# Imitation Learning Architecture
LAYERS = [256, 256]                  # Hidden layers for the neural network
# TOTAL_TIMESTEPS = 1_000_000
ACTIVATION = "relu"
OUTPUT_ACTIVATION = "tanh"
NETWORK_EPSILON = 1e-3
DECOMPOSITION_TYPE = "diagonal"  # diagonal, full, cholesky
OUTPUT_HORIZON = 1
HORIZON = 10


LOG_EVERY = 1
CHECKPOINT_FREQUENCY = 10

EVAL_FREQUENCY = 10
NUM_EVAL_EPS = 10



# RNG Seed
SEED = 100



# =============================================================================
# DERIVED PATHS (don't edit)
# =============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RL_SAVE_PATH = os.path.join(RL_BASE_SAVE_PATH, RL_EXPERIMENT_NAME)
SAVE_PATH = os.path.join(IL_BASE_SAVE_PATH, EXPERIMENT_NAME)
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
    planet = PLANET

    env = SpacecraftEnv(
        dynamics_params=dynamics_params,
        planet=planet,
        dt=DT,
        N_orbit=N_ORBIT,
        # num_steps=MAX_EPISODE_STEPS,
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




def setup_logging(log_dir: str, log_filename: str = "training.log") -> logging.Logger:
    """Setup logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(EXPERIMENT_NAME)
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
            "name": EXPERIMENT_NAME,
            "notes": EXPERIMENT_NOTES,
            "timestamp": TIMESTAMP,
        },
        "environment": {
            "planet": "earth" if PLANET is Earth else "uranus",
            "dt": DT,
            "n_orbit": N_ORBIT,
            "max_episode_length": MAX_EPISODE_LENGTH,
            "dyn_noise_std": DYN_NOISE_STD,
            "state_limits_quat": STATE_LIMITS if STATE_LIMITS is not None else None,
            "state_limits_mrp": STATE_LIMITS_MRP.tolist() if STATE_LIMITS_MRP is not None else None,
            "control_limits": CONTROL_LIMITS.tolist() if CONTROL_LIMITS is not None else None,
            "theta_threshold": THETA_THRESHOLD,
            "omega_threshold": OMEGA_THRESHOLD,
        },
        "dagger": {
            "num_iterations": NUM_ITERATIONS,
            "num_trajectories": NUM_TRAJECTORIES,
            "num_gradient_steps": NUM_GRADIENT_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            # "beta_init": BETA,
            "beta_decay": BETA_DECAY,
            "max_buffer_size": MAX_BUFFER_SIZE,
        },
        "network": {
            "layers": LAYERS,
        },
        "mpc": {
            "horizon": HORIZON,
            # "q_diag": Q_DIAG,
            # "r_diag": R_DIAG,
            # "qf_diag": QF_DIAG,
        },
        "training": {
            "seed": SEED,
            "log_every": LOG_EVERY,
            "checkpoint_frequency": CHECKPOINT_FREQUENCY,
            # "eval_frequency": EVAL_FREQUENCY,
            # "n_eval_episodes": N_EVAL_EPISODES,
        },
        "paths": {
            "expert_model": RL_SAVE_PATH,
            # "expert_vecnorm": EXPERT_VECNORM_PATH,
            "checkpoint_dir": CHECKPOINT_PATH,
            # "log_dir": LOG_DIR,
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





# def collect_trajectory(env: VecNormalize,
#                        expert_policy: PPO,
#                        controller: DiffMPCController,
#                        max_episode_length: int,
#                        beta:float,
#                        key: jax.random.PRNGKey) -> Tuple[Dict[str, jnp.ndarray], jax.random.PRNGKey]:



#     raw_env = env.envs[0].unwrapped

#     key, subkey = jax.random.split(key)
#     seed = int(jax.random.randint(subkey, shape=(), minval=0, maxval=2**32 - 1))
#     raw_obs, info = raw_env.reset(seed=seed)

#     initial_state = info["initial_state"]
#     goal_state = info["goal_state"]

#     key, subkey = jax.random.split(key)
#     nominal_traj = jnp.tile(initial_state,(controller.horizon+1,1))
#     nominal_cntrl = 0.001*jax.random.normal(subkey, shape=((controller.horizon, controller.nu)))

#     obs_list = []
#     expert_action_list = []
#     nominal_traj_list = []
#     nominal_cntrl_list = []

#     for _ in range(max_episode_length):


#         normalized_obs = env.normalize_obs(raw_obs)
#         expert_action, _ = expert_policy.predict(normalized_obs, deterministic=True)

#         controller_action, controller_nominal_traj, controller_nominal_cntrl = controller(jnp.array(raw_obs), 
#                                                                                           jnp.array(goal_state), 
#                                                                                           nominal_traj, 
#                                                                                           nominal_cntrl)

#         obs_list.append(jnp.array(raw_obs))
#         expert_action_list.append(jnp.array(expert_action))
#         nominal_traj_list.append(nominal_traj)
#         nominal_cntrl_list.append(nominal_cntrl)
        

#         key, subkey = jax.random.split(key)
#         if float(jax.random.uniform(subkey)) < beta:
#             executed_action = expert_action

#             # Three options when it comes to updating nominals 
#             # for DAgger as I see it
#             # Option 1: Let MPC make a plan and store that as part of augmented dataset
#             # Option 2: Let expert make a plan and store that as part of augmented dataset
#             # Option 3: Don't make a plan at each timestep and use previous MPC calls for replanning

#             # Looking at Patrick's code, he doesn't replan unless he is required to
#             # That is most similar to option 3

#             # That said, Patrick's code assumes that the controller
#             # policy is actually being run
#             # that is not true in our case
            
#             # For option 1, below use controller_nominal_traj and controller_nominal_cntrl
#             # For option 2 we need to call expert policy to make a plan

#             # Option 3
#             nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
#             nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)

#             # # Option 2
#             # nominal_traj = controller_nominal_traj
#             # nominal_cntrl = controller_nominal_cntrl

#         else:
#             executed_action = np.array(controller_action) / MAX_TORQUE  # Scale action to [-1, 1] range for env
#             # # Check if executed action is within control limits
#             # control_limits = jnp.array([[-1, 1]] * 3, dtype=jnp.float64)
#             # if not jnp.all((executed_action >= control_limits[:, 0]) & (executed_action <= control_limits[:, 1])):
#             #     raise ValueError(f"Executed action {executed_action} is out of control limits {control_limits}")
#             # executed_action = np.clip(np.array(controller_action) / MAX_TORQUE, -1.0, 1.0)
#             nominal_traj = controller_nominal_traj
#             nominal_cntrl = controller_nominal_cntrl

            

        

#         # Step env
#         raw_obs, reward, done, truncated, info = raw_env.step(executed_action)

#     trajectory = {
#         "obs": jnp.array(obs_list),
#         "goal_state": jnp.array(goal_state),
#         "expert_actions": jnp.array(expert_action_list),
#         "nominal_traj": jnp.array(nominal_traj_list),
#         "nominal_cntrl": jnp.array(nominal_cntrl_list),
#     }

#     return trajectory, key



def collect_trajectory(initial_state: jnp.ndarray,
                       target_state: jnp.ndarray,
                       key: jax.random.PRNGKey,
                       dt: float,
                       expert_policy: Callable,
                       controller: DiffMPCController,
                       max_episode_length: int,
                       beta: float,
                       replan_freq: int,
                       noise_std: float,) -> Tuple[Dict[str, jnp.ndarray], jax.random.PRNGKey]:


    initial_state = jnp.array(initial_state, dtype=jnp.float64)
    target_state = jnp.array(target_state, dtype=jnp.float64)

    def scan_step(carry, _):

        state, key, i, nominal_traj, nominal_cntrl = carry
        t_current = i * dt
        key, rk4_key, dagger_key = jax.random.split(key, 3)

        state_error = get_state_error(state, target_state)

        def controller_wrapper(operand):
            return controller(*operand)
            
        def no_update(operand):
            _, __, nominal_traj, nominal_cntrl = operand
            # Shift nominal trajectory and nominal cntrl
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]
            return action, nominal_traj, nominal_cntrl

        # Get controller action accounting for replan frequency
        controller_action, controller_nominal_traj, controller_nominal_cntrl = jax.lax.cond(
            i % replan_freq == 0,
            controller_wrapper,
            no_update,
            operand=(state_error, target_state, nominal_traj, nominal_cntrl)
        )

        controller_action = controller_action / MAX_TORQUE  # Scale action to [-1, 1] range for loss computation
        # controller and expert action are now in [-1, 1] range, maintained for use in loss function
        expert_action = expert_policy(state_error)

        use_expert = jax.random.uniform(dagger_key) < beta
        executed_action = jnp.where(use_expert, expert_action, controller_action) * MAX_TORQUE  # Scale back to original range for rk4 step

        # Shift the nominal trajectory and control
        shifted_traj  = jnp.concatenate([nominal_traj[1:],  nominal_traj[-1:]],  axis=0)
        shifted_cntrl = jnp.concatenate([nominal_cntrl[1:], nominal_cntrl[-1:]], axis=0)
        new_nominal_traj  = jnp.where(use_expert, shifted_traj,  controller_nominal_traj)
        new_nominal_cntrl = jnp.where(use_expert, shifted_cntrl, controller_nominal_cntrl)

        # Step dynamics using RK4
        new_state = rk4_step(state, executed_action, t_current, dt, rk4_key, noise_std)

        new_carry = (new_state, key, i + 1, new_nominal_traj, new_nominal_cntrl)
        outputs = (state_error, expert_action, nominal_traj, nominal_cntrl)

        return new_carry, outputs
        



    # Def init stuff
    # key, subkey = jax.random.split(key)

    init_nom_traj = jnp.tile(initial_state,(controller.horizon+1,1))
    init_nom_control = jnp.zeros(shape=(init_nom_traj.shape[0]-1, controller.nu))

    init_carry = (initial_state, key, 0, init_nom_traj, init_nom_control)

    final_carry, outputs = jax.lax.scan(scan_step, init_carry, None, length=max_episode_length)

    final_state, key, final_step, final_nom_traj, final_nom_cntrl = final_carry

    observations, expert_actions, nominal_trajs, nominal_cntrls = outputs

    return {
        "obs": observations,
        # "goal_state": jnp.broadcast_to(target_state, (max_episode_length, target_state.shape[0])),
        "goal_state": target_state,
        "expert_actions": expert_actions,
        "nominal_traj": nominal_trajs,
        "nominal_cntrl": nominal_cntrls,
    }, key  

@eqx.filter_jit
def collect_trajectories(expert_policy: Callable,
                         controller: DiffMPCController,
                         dt: float,
                         max_episode_length: int,
                         beta: float,
                         key: jax.random.PRNGKey,
                         replan_freq: int,
                         noise_std: float,
                         num_trajectories: int) -> Tuple[List[Dict[str, jnp.ndarray]], jax.random.PRNGKey]:


    # Sample initial and target states for all trajectories
    key, subkey = jax.random.split(key)
    initial_states, target_states = sample_episode_context(subkey, batch_size=num_trajectories)

    keys = jax.random.split(key, num_trajectories + 1)  # Split keys for each trajectory
    key = keys[0]  # Update key 
    keys = keys[1:num_trajectories + 1]  # Use num_trajectories keys for trajectory collection
    # Get trajs using vmap
    trajs, _ = jax.vmap(collect_trajectory, in_axes=(0, 0, 0, None, None, None, None, None, None, None))(
        initial_states,
        target_states,
        keys,
        dt,
        expert_policy,
        controller,
        max_episode_length,
        beta,
        replan_freq,
        noise_std
    )

    return trajs, key  # Return trajectories and updated key








# def collect_trajectories(env: VecNormalize,
#                         expert_policy: Callable,
#                         controller: DiffMPCController,
#                         num_trajectories: int,
#                         max_episode_length: int,
#                         beta: float,
#                         key: jax.random.PRNGKey) -> Tuple[List[Dict[str, jnp.ndarray]], jax.random.PRNGKey]:

#     trajectories = []
#     for _ in range(num_trajectories):
#         trajectory, key = collect_trajectory(env, expert_policy, controller, max_episode_length, beta, key)
#         trajectories.append(trajectory)

#     return trajectories, key




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

        # Compute gradient norm for debugging
        # grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
        # grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        
        # jax.debug.print("Loss: {}, Grad norm: {}", loss, grad_norm)  # Uncomment to debug

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



def train_iteration(controller: DiffMPCController,
                    expert_policy: PPO, 
                    dt: float,
                    replay_buffer: ReplayBuffer,
                    beta: float,
                    optimizer: optax.GradientTransformation,
                    opt_state: optax.OptState,
                    key: jax.random.PRNGKey,
                    num_trajectories: int,
                    max_episode_length: int,
                    num_gradient_steps: int,
                    replan_freq: int,
                    noise_std: float,
                    batch_size: int,
                    beta_decay: float):


    # Collect trajectories
    # trajectories, key = collect_trajectories(env, expert_policy, controller, num_trajectories, max_episode_length, beta, key)

    trajectories, key = collect_trajectories(expert_policy=expert_policy,
                                             controller=controller,
                                             dt=dt,
                                             max_episode_length=max_episode_length,
                                             beta=beta,
                                             key=key,
                                             replan_freq=replan_freq,
                                             noise_std=noise_std,
                                             num_trajectories=num_trajectories)

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



def evaluate(controller: DiffMPCController, 
            max_steps: int,
            num_episodes: int, 
            key: jax.random.PRNGKey):

    print("Evaluating Imitation Learning Agent")

    # Load learned magnetic field models
    model_path = URANUS_MPC_PATH + '/models/'
    if PLANET is Earth:
        model_s, _ = load_model(filename=model_path + '/earth_b_4d.eqx') 
    elif PLANET is Uranus:
        model_s, _ = load_model(filename=model_path + '/uranus_b_4d.eqx')
    else:
        raise ValueError("PLANET should be either Earth or Uranus")


    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=PLANET)
    dynamics_params = spacecraft_dynamics.dynamics_params
    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=DT)


    trajectories = jnp.zeros((num_episodes, max_steps+1, 7))
    target_states = jnp.zeros((num_episodes, 7))
    step_actions = jnp.zeros((num_episodes, max_steps, 3))  # Store actions for each episode

    start_time = time.time()

    # TODO: Change evaluation to use collect_trajectories function for efficiency
    # Will also need to make my own plotting functions to handle this then
    # Which I really don't want to do right now

    for ep in range(num_episodes):
        ep_start_time = time.time()

        # print("Episode Number: ",ep)
        key, init_key, traj_key = jax.random.split(key,3)
        # Sample initial state and target state for the episode
        initial_state, target_state = sample_episode_context(init_key)
        # print("Initial State: ", initial_state)
        # print("Target State: ", target_state)

        # Generate a trajectory
        trajectory, controls = generate_trajectory(initial_state, 
                                         target_state, 
                                         controller.dt, 
                                         traj_key, 
                                         max_steps,
                                         controller)

        # print("Trajectory: ", trajectory)

        trajectories = trajectories.at[ep].set(trajectory)
        target_states = target_states.at[ep].set(target_state)
        step_actions = step_actions.at[ep].set(controls)


        traj_nans = jnp.isnan(trajectories).any()
        traj_infs = jnp.isinf(trajectories).any()
        cntrl_nans = jnp.isnan(controls).any()
        cntrl_infs = jnp.isinf(controls).any()

        has_issues = traj_nans or traj_infs or cntrl_nans or cntrl_infs

        if has_issues:
            prefix = f"Episode {ep}: " if ep is not None else ""

        if traj_nans:
            nan_count = jnp.isnan(trajectories).sum()
            nan_locations = jnp.where(jnp.isnan(trajectories).any(axis=-1))
            first_nan_step = nan_locations[0][0] if len(nan_locations[0]) > 0 else "N/A"
            print(f"{prefix}Trajectory has {nan_count} NaN values! First NaN at step {first_nan_step}")
            
        if traj_infs:
            inf_count = jnp.isinf(trajectories).sum()
            print(f"{prefix}Trajectory has {inf_count} Inf values!")
            
        if cntrl_nans:
            nan_count = jnp.isnan(controls).sum()
            nan_locations = jnp.where(jnp.isnan(controls).any(axis=-1))
            first_nan_step = nan_locations[0][0] if len(nan_locations[0]) > 0 else "N/A"
            print(f"{prefix}Controls have {nan_count} NaN values! First NaN at step {first_nan_step}")
            
        if cntrl_infs:
            inf_count = jnp.isinf(controls).sum()
            print(f"{prefix}Controls have {inf_count} Inf values!")


        ep_end_time = time.time()

        # print("Time taken: ",ep_end_time - ep_start_time)

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

    _ = system.plot_costs(trajectories, target_states, plot_stats=True)

    system.plot_violin_and_bar(trajectories, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)










def learn(controller: DiffMPCController, 
          expert_policy: Callable,
          dt: float,
          replay_buffer: ReplayBuffer,
          optimizer: optax.GradientTransformation,
          opt_state: optax.OptState,
          key: jax.random.PRNGKey,
          num_iterations: int,
          num_trajectories: int,
          max_episode_length: int,
          num_gradient_steps: int,
          batch_size: int,
          beta_decay: float,
          log_frequency: int,
          checkpoint_frequency: int,
          checkpoint_path: str,
          logger,
          evaluate_freq: int,
          num_eval_eps: int,
          replan_freq: int = 1,
          noise_std: float = 1e-6,
          eval_after: int = 0,
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
        controller, replay_buffer, opt_state, mean_loss, beta, key = train_iteration(controller,
                                                                                     expert_policy,
                                                                                     dt,
                                                                                     replay_buffer,
                                                                                     beta,
                                                                                     optimizer,
                                                                                     opt_state,
                                                                                     key,
                                                                                     num_trajectories,
                                                                                     max_episode_length,
                                                                                     num_gradient_steps,
                                                                                     replan_freq,
                                                                                     noise_std,
                                                                                     batch_size,
                                                                                     beta_decay)


        itr_end_time = time.time()

        # Update t_so_far

        # do logging
        if itr % log_frequency == 0:

            # print(f"Iteration {itr}, Mean loss: {mean_loss}, time: {itr_end_time - start_time}")
            logger.info(f"Iteration {itr}, Mean loss: {mean_loss}, beta: {beta}, itr time take: {itr_end_time - itr_start_time}")

        # save checkpoint

        if itr % checkpoint_frequency == 0:
            
            save_il_model(controller, replay_buffer, opt_state, beta, itr, key, checkpoint_path)

        # Evaluate
        if itr % evaluate_freq == 0 and itr > eval_after:
            key, subkey = jax.random.split(key)
            evaluate(controller, max_episode_length, num_eval_eps, subkey)

    # final print
    logger.info(f"itr {itr}, Mean loss: {mean_loss}, beta: {beta}, total time taken: {time.time() - start_time}")

    # Save final model
    logger.info("Training done, saving final model")

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

    model_path = URANUS_MPC_PATH + '/models/'

    planet = PLANET

    if planet is Earth:
        model_s, _ = load_model(filename=model_path + '/earth_b_4d.eqx') 
    elif planet is Uranus:
        model_s, _ = load_model(filename=model_path + '/uranus_b_4d.eqx')


    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    dynamics_params = spacecraft_dynamics.dynamics_params
    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=DT)

    # Make the env
    # Not changing defaults for now (don't really need to besides ep length)
    # env = SpacecraftEnv()
    vec_env = DummyVecEnv([lambda: make_env(dynamics_params=dynamics_params, seed=None)])
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
                               output_horizon=OUTPUT_HORIZON, 
                               eps=NETWORK_EPSILON, 
                               decomposition_type=DECOMPOSITION_TYPE)


    # Initialize controller
    controller = DiffMPCController(network,HORIZON,DT,STATE_LIMITS_MRP,CONTROL_LIMITS)

    
    # Initialize optimizer
    # optimizer = optax.adam(learning_rate=LEARNING_RATE)

    # Initialize optimizer + gradient clipping 
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to prevent exploding
        optax.adam(learning_rate=LEARNING_RATE)
    )

    opt_state = optimizer.init(eqx.filter(controller.network,eqx.is_array))



    # Initialize replay buffer
    replay_buffer = init_buffer(max_size=MAX_BUFFER_SIZE,
                                obs_dim=network.nx,
                                state_dim=network.nx,
                                action_dim=network.nu,
                                horizon=HORIZON)

    # Load expert policy
    rl_path = os.path.join(RL_SAVE_PATH, "final_model.zip")
    expert_policy = PPO.load(rl_path, env=vec_env)


    # Setup logger stuff
    logger = setup_logging(SAVE_PATH,LOG_FILENAME)



    # Extract flax expert policy from stable-baselines3 PPO model
    # actor_module = expert_policy.policy.actor
    # actor_params = expert_policy.policy.actor_state.params


    # # Extract vector normalization statistics from the VecNormalize wrapper
    # obs_mean = jnp.array(vec_env.obs_rms.mean, dtype=jnp.float64)
    # obs_var = jnp.array(vec_env.obs_rms.var, dtype=jnp.float64)
    # obs_count = vec_env.obs_rms.count
    # obs_eps = vec_env.epsilon
    # obs_clip = vec_env.clip_obs


    expert_policy = get_expert_policy(expert_policy, vec_env)





    # Learn stuff
    # key, subkey = jax.random.split(key)
    controller, replay_buffer, opt_state, beta, key = learn(controller,
                                                            expert_policy,
                                                            dt=DT,
                                                            replay_buffer=replay_buffer,
                                                            optimizer=optimizer,
                                                            opt_state=opt_state,
                                                            key=key,
                                                            num_iterations=NUM_ITERATIONS,
                                                            num_trajectories=NUM_TRAJECTORIES,
                                                            max_episode_length=MAX_EPISODE_LENGTH,
                                                            num_gradient_steps=NUM_GRADIENT_STEPS,
                                                            batch_size=BATCH_SIZE,
                                                            beta_decay=BETA_DECAY,
                                                            log_frequency=LOG_EVERY,
                                                            checkpoint_frequency=CHECKPOINT_FREQUENCY,
                                                            checkpoint_path=CHECKPOINT_PATH,
                                                            logger=logger,
                                                            evaluate_freq=EVAL_FREQUENCY,
                                                            num_eval_eps=NUM_EVAL_EPS,
                                                            replan_freq=REPLAN_FREQ,
                                                            noise_std=NOISE_STD,
                                                            resume=RESUME_TRAINING)



    # model, optimizer_state, beta, replay_buffer, t_so_far, key = learn(env, model, expert_policy, TOTAL_TIMESTEPS, optimizer, key)

    # Save the final model and optimizer state
    save_il_model(controller, replay_buffer, opt_state, beta, NUM_ITERATIONS, key, SAVE_PATH)

    # # Save the model and optimizer state
    # model_save_path = os.path.join(SAVE_PATH, "final_model.eqx")
    # optimizer_state_save_path = os.path.join(SAVE_PATH, "final_optimizer_state.eqx")

    # eqx.tree_serialise_leaves(model_save_path, model)
    # eqx.tree_serialise_leaves(optimizer_state_save_path, optimizer_state)

    # Do evaluations
    evaluate(controller, max_steps=MAX_EPISODE_LENGTH, num_episodes=100, key=key)


def dry_test():


    # Some test code to check how the model works with no training

    # Make env
    env = SpacecraftEnv()
    

    # Initialize model
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    network = FeedForwardNetwork(nx=7, 
                               nu=3, 
                               key=subkey, 
                               layers=LAYERS, 
                               activation='relu', 
                               output_activation='tanh', 
                               output_horizon=1, 
                               eps=1e-3, 
                               decomposition_type='diagonal')

    controller = DiffMPCController(network,
                                    horizon=10,
                                    dt=DT,
                                    state_limits=STATE_LIMITS_MRP,
                                    control_limits=CONTROL_LIMITS)


    evaluate(controller, max_steps = env.num_steps, num_episodes=10, key=key)







if __name__ == "__main__":


    main()

    # dry_test()
