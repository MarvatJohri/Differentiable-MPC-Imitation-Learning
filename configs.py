"""

dict of all hyperparams used in models

"""

import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass, field

HERE = Path(__file__).parent
ROOT = HERE.parent

PPO_BASE_SAVE_PATH = str(HERE / "ppo_results")
PPO_BASE_LOG_PATH = str(HERE / "ppo_logs")

DAGGER_BASE_SAVE_PATH = str(HERE / "dagger_results")

SPACECRAFT_DYNAMICS_PARAMETERS = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
SPACECRAFT_DYNAMICS_PARAMETERS["inertia_inv"] = jnp.linalg.inv(SPACECRAFT_DYNAMICS_PARAMETERS["inertia"]) 
STATE_LIMITS = jnp.asarray([[-1, 1]]*4 + [[-2,2]]*3, dtype=jnp.float64)
STATE_LIMITS_MRP = jnp.asarray([[-180, 180]]*3 + [[-2,2]]*3, dtype=jnp.float64)
CONTROL_LIMITS = jnp.asarray(jnp.array([[-1, 1]] * 3), dtype=jnp.float64)
MAX_TORQUE = 5e-5
CONTROL_LIMITS_TORQUE = jnp.asarray(jnp.array([[-MAX_TORQUE, MAX_TORQUE]] * 3), dtype=jnp.float64)

NET_ARCH = [256, 256]
DAGGER_LAYERS = [256, 256]


@dataclass
class ExpConfig:

    ppo_base_save_path: str = PPO_BASE_SAVE_PATH
    ppo_base_log_path: str = PPO_BASE_LOG_PATH
    dagger_base_save_path: str = DAGGER_BASE_SAVE_PATH

    ppo_experiment_name: str = "spacecraft_ppo_omega_hard_limit_test"
    ppo_experiment_notes: str = "Initial PPO training on Earth orbit"

    dagger_experiment_name: str = "spacecraft_ppo_dagger_exp1"
    dagger_experiment_notes: str = "Initial DAgger training on Earth orbit"

    resume_ppo_training: bool = False
    resume_dagger_training: bool = False

    ppo_resume_timesteps: int = 0
    dagger_resume_timesteps: int = 0


    ppo_resume_model_path: str = field(init=False)
    dagger_resume_model_path: str = field(init=False)

    nx: int = 7
    nu: int = 3

    def __post_init__(self):

        self.ppo_resume_model_path: str = self.ppo_base_save_path + f"/{self.ppo_experiment_name}" + "/checkpoints" + f"/model_{self.ppo_resume_timesteps}_steps.zip"
        self.dagger_resume_model_path: str = self.dagger_base_save_path + f"/{self.dagger_experiment_name}" + "/checkpoints" + f"/model_{self.dagger_resume_timesteps}_steps.eqx"


    seed: int = 42

    # Evaluation stuff
    num_eval_eps: int = 100



@dataclass
class SpacecraftEnvConfig:
    
    # spacecraft_dynamics_parameters: dict = SPACECRAFT_DYNAMICS_PARAMETERS
    dt: float = 0.1
    max_ep_steps: int = 1500
    state_limits: jnp.ndarray = field(default_factory=lambda: STATE_LIMITS)
    state_limits_mrp: jnp.ndarray = field(default_factory=lambda: STATE_LIMITS_MRP)
    control_limits: jnp.ndarray = field(default_factory=lambda: CONTROL_LIMITS)
    max_torque: float = MAX_TORQUE
    control_limits_torque: jnp.ndarray = field(default_factory=lambda: CONTROL_LIMITS_TORQUE)
    dyn_noise_std: float = 1e-6
    theta_threshold: float = float(jnp.deg2rad(15.0))
    omega_threshold: float = float(jnp.deg2rad(5.0))
    theta_stability_tol: float = float(jnp.deg2rad(10.0))
    omega_stability_tol: float = float(jnp.deg2rad(5.0))
    theta_threshold_reward: float = 10.0
    goal_reward: float = 50.0
    omega_fail_penalty: float = 50.0
    omega_penalty: float = 0.5
    action_penalty: float = 0.1

    spacecraft_dynamics_parameters: dict = field(default_factory=lambda: SPACECRAFT_DYNAMICS_PARAMETERS)



@dataclass
class ControllerConfig:

    nx: int = 7
    nu: int = 3
    mpc_horizon: int = 10
    replan_frequency: int = 1
    decomposition_type: str = 'diagonal'
    qr_output_horizon: int = 1
    network_epsilon: float = 1e-3



@dataclass
class PPOHyperparameters:

    policy_type: str = 'MlpPolicy'
    net_arch: list = field(default_factory=lambda: NET_ARCH)
    learning_rate: float = 1e-3
    learning_rate_schedule: str = 'constant'
    learning_rate_final: float = 1e-5
    n_rollouts: int = 3
    n_steps: int = 4500
    n_epochs: int = 10
    batch_size: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_500_000
    checkpoint_freq: int = 100_000


@dataclass
class DaggerHyperparameters:
    
    layers: list = field(default_factory=lambda: DAGGER_LAYERS)
    learning_rate: float = 3e-4
    learning_rate_schedule: str = 'constant'
    learning_rate_final: float = 1e-5
    batch_size: int = 512
    beta_decay: float = 0.95
    num_eps_stored: int = 100
    max_buffer_size: int = 150_000
    num_iterations: int = 100
    num_trajectories: int = 10
    num_gradient_steps: int = 100
    activation: str = 'relu'
    output_activation: str = 'tanh'
    network_epsilon: float = 1e-3
    decomposition_type: str = 'diagonal'
    qr_output_horizon: int = 1
    mpc_horizon: int = 10
    replan_frequency: int = 1
    log_every: int = 1
    checkpoint_frequency: int = 10
    eval_frequency: int = 10
    num_eval_eps: int = 100








