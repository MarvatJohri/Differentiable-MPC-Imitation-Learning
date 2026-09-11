"""

Code for Imitation Learning agent implemented using differentiable MPC

"""

import sys
import os
from datetime import datetime
import time
from typing import Callable, Dict, List, Tuple
import logging



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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import my stuff

from simulation_env_simpler import SpacecraftEnv
from replay_buffer import ReplayBuffer, init_buffer, add_trajectories_to_buffer, sample_from_buffer
from mj_utils import make_dummy_controller, make_dummy_expert, compute_metrics, plot_metrics_bar, print_metrics, plot_metrics_comparison
from diffmpc_controller import DiffMPCController, FeedForwardNetwork
from simulation_env_jax import SpacecraftEnvJax, generate_trajectory, generate_n_trajectories
from configs import ExpConfig, SpacecraftEnvConfig, DaggerHyperparameters


exp_config = ExpConfig()
hyperparameters = DaggerHyperparameters()
env_config = SpacecraftEnvConfig()


# Load Hyperparameters

PPO_BASE_SAVE_PATH = exp_config.ppo_base_save_path
PPO_BASE_LOG_PATH = exp_config.ppo_base_log_path

DAGGER_BASE_SAVE_PATH = exp_config.dagger_base_save_path

PPO_EXPERIMENT_NAME = exp_config.ppo_experiment_name
DAGGER_EXPERIMENT_NAME = exp_config.dagger_experiment_name
DAGGER_EXPERIMENT_NOTES = exp_config.dagger_experiment_notes

RESUME = exp_config.resume_dagger_training

SEED = exp_config.seed



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
NETWORK_EPSILON = hyperparameters.network_epsilon
DECOMPOSITION_TYPE = hyperparameters.decomposition_type
QR_OUTPUT_HORIZON = hyperparameters.qr_output_horizon
MPC_HORIZON = hyperparameters.mpc_horizon
REPLAN_FREQUENCY = hyperparameters.replan_frequency
LEARNING_RATE = hyperparameters.learning_rate
LEARNING_RATE_FINAL = hyperparameters.learning_rate_final
LEARNING_RATE_SCHEDULE = hyperparameters.learning_rate_schedule
BATCH_SIZE = hyperparameters.batch_size
BETA_DECAY = hyperparameters.beta_decay
NUM_EPS_STORED = hyperparameters.num_eps_stored
MAX_BUFFER_SIZE = hyperparameters.max_buffer_size
NUM_ITERATIONS = hyperparameters.num_iterations
NUM_TRAJECTORIES = hyperparameters.num_trajectories
NUM_GRADIENT_STEPS = hyperparameters.num_gradient_steps
LOG_EVERY = hyperparameters.log_every
CHECKPOINT_FREQUENCY = hyperparameters.checkpoint_frequency
EVAL_FREQUENCY = hyperparameters.eval_frequency
NUM_EVAL_EPS = hyperparameters.num_eval_eps



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
RESUME_TRAIN_STATE_FILE = os.path.join(CHECKPOINT_PATH, f"train_state_{RESUME_ITERATIONS}_steps.eqx")

os.makedirs(SAVE_PATH, exist_ok=True)
# os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# os.makedirs(TENSORBOARD_PATH, exist_ok=True)
# Ensure directories exist

os.makedirs(DAGGER_BASE_SAVE_PATH, exist_ok=True)




def make_env(dynamics_params: Dict, seed: int = None):
    """Create and wrap environment."""
    env = SpacecraftEnv(
        dynamics_params=dynamics_params,
        dt=DT,
        max_ep_steps=MAX_EP_STEPS,
        state_limits=np.array(STATE_LIMITS),
        control_limits=CONTROL_LIMITS,
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



def load_network_params(controller: DiffMPCController, checkpoint_file: str) -> DiffMPCController:
    """Load network parameters from a checkpoint file into the controller."""
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")
    
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_file, {
        "network_params": eqx.filter(controller.network, eqx.is_array)
    })
    
    network_params = checkpoint["network_params"]
    static_network = eqx.filter(controller.network, lambda x: not eqx.is_array(x))
    new_network = eqx.combine(network_params, static_network)
    
    controller = eqx.tree_at(lambda c: c.network, controller, new_network)
    
    print(f"Network parameters loaded from {checkpoint_file}")
    
    return controller


def load_train_state(replay_buffer: ReplayBuffer, 
                     opt_state: optax.OptState, 
                     checkpoint_file: str) -> Tuple[ReplayBuffer, optax.OptState, float, int, jax.random.PRNGKey]:


    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")

    train_state = {
        "iteration": jnp.array(0),
        "beta": jnp.array(1.0),
        "replay_buffer": replay_buffer,
        "replay_buffer_max_size": jnp.array(replay_buffer.max_size),
        "opt_state": opt_state,
        "key": jax.random.PRNGKey(0),
    }

    train_state = eqx.tree_deserialise_leaves(checkpoint_file, train_state)
    replay_buffer = train_state["replay_buffer"].replace(max_size=int(train_state["replay_buffer_max_size"]))


    print(f"Train state loaded from {checkpoint_file} at iteration {train_state['iteration']}")

    return replay_buffer, train_state["opt_state"], float(train_state["beta"]), int(train_state["iteration"]), train_state["key"]




def load_il_model(controller: DiffMPCController,
                    replay_buffer: ReplayBuffer,
                    opt_state: optax.OptState,
                    checkpoint_file: str,
                    train_state_file: str):


    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")

    if not os.path.exists(train_state_file):
        raise FileNotFoundError(f"Train state file {train_state_file} does not exist.")

    # Load network parameters
    controller = load_network_params(controller, checkpoint_file)
    # Load train state
    replay_buffer, opt_state, beta, iteration, key = load_train_state(replay_buffer, opt_state, train_state_file)

    return controller, replay_buffer, opt_state, beta, iteration, key


def save_network_params(controller: DiffMPCController, checkpoint_file: str):
    """Save network parameters from the controller to a checkpoint file."""
    
    checkpoint = {
        "network_params": eqx.filter(controller.network, eqx.is_array)
    }
    
    eqx.tree_serialise_leaves(checkpoint_file, checkpoint)
    
    print(f"Network parameters saved to {checkpoint_file}")


def save_train_state(replay_buffer: ReplayBuffer,
                        opt_state: optax.OptState,
                        beta: float,
                        iteration: int,
                        key: jax.random.PRNGKey,
                        train_state_file: str):


    train_state = {
        "iteration": jnp.array(iteration),
        "beta": jnp.array(beta),
        "replay_buffer": replay_buffer,
        "replay_buffer_max_size": jnp.array(replay_buffer.max_size),
        "opt_state": opt_state,
        "key": key,
    }

    eqx.tree_serialise_leaves(train_state_file, train_state)

    print(f"Train state saved at iteration {iteration} to {train_state_file}")



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
        train_state_file = os.path.join(checkpoint_path, f"final_train_state.eqx")
    else:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(checkpoint_path, f"model_{iteration}_steps.eqx")
        train_state_file = os.path.join(checkpoint_path, f"train_state_{iteration}_steps.eqx")

    # Save network parameters
    save_network_params(controller, checkpoint_file)

    # Save train state
    save_train_state(replay_buffer, opt_state, beta, iteration, key, train_state_file)






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


    # Add trajectories to buffer
    replay_buffer = add_trajectories_to_buffer(replay_buffer,trajectories)

    # Do training updates
    controller, opt_state, mean_loss, key = train_loop(controller, optimizer, opt_state, replay_buffer, num_gradient_steps, batch_size, key)


    # Update beta
    beta = beta * beta_decay

    return controller, replay_buffer, opt_state, mean_loss, beta, key





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
          resume: bool = False,
          resume_checkpoint_file: str = None,
          resume_train_state_file: str = None):



    # Initialize Beta
    beta = 1

    logger.info("Starting training")

    itrs_done = 0
    if resume:
        controller, replay_buffer, opt_state, beta, iteration, key = load_il_model(controller, replay_buffer, opt_state, resume_checkpoint_file, resume_train_state_file)
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
                                                            resume_checkpoint_file=RESUME_CHECKPOINT_FILE,
                                                            resume_train_state_file=RESUME_TRAIN_STATE_FILE,
                                                            replan_frequency=REPLAN_FREQUENCY,
                                                            evaluate_freq=EVAL_FREQUENCY,
                                                            num_eval_eps=NUM_EVAL_EPS)



    # Save the final model and optimizer state
    save_il_model(controller, replay_buffer, opt_state, beta, NUM_ITERATIONS, key, SAVE_PATH)


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


    





if __name__ == "__main__":


    main()

    # test_jax_env()

    # dry_test()
