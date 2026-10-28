"""Evaluate the imitation-learned differentiable MPC controller."""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from configs import (
    ExpConfig, SpacecraftEnvConfig, DaggerHyperparameters, ControllerConfig,
    get_env,
)
from diffmpc_controller import DiffMPCController, FeedForwardNetwork
from mj_utils import load_controller, compute_metrics, print_metrics
from rl_utils.generate_trajs import n_rollouts_controller


exp_config = ExpConfig()
env_config = SpacecraftEnvConfig()
hyperparameters = DaggerHyperparameters()
controller_config = ControllerConfig()

# Switch to "spacecraft_env" to evaluate the original torque environment.
ENV_NAME = "spacecraft_env"
N_ROLLOUTS = 100
SEED = exp_config.seed

SAVE_PATH = os.path.join(
    exp_config.dagger_base_save_path, exp_config.dagger_experiment_name,
)
EVAL_SAVE_PATH = os.path.join(SAVE_PATH, "eval_results", ENV_NAME)

LOAD_FROM_CHECKPOINT = True
LOAD_ITERATION = 0
if LOAD_FROM_CHECKPOINT:
    MODEL_FILE = os.path.join(
        SAVE_PATH, "checkpoints", f"model_{LOAD_ITERATION}_steps.eqx",
    )
else:
    MODEL_FILE = os.path.join(SAVE_PATH, "final_model.eqx")


def main():
    # The learned controller keeps its original nominal dynamics. Disturbances
    # are applied by the evaluation environment.
    key = jax.random.PRNGKey(SEED)
    key, network_key = jax.random.split(key)
    network = FeedForwardNetwork(
        nx=controller_config.nx,
        nu=controller_config.nu,
        key=network_key,
        layers=hyperparameters.layers,
        activation=hyperparameters.activation,
        output_activation=hyperparameters.output_activation,
        qr_output_horizon=controller_config.qr_output_horizon,
        eps=controller_config.network_epsilon,
        decomposition_type=controller_config.decomposition_type,
    )
    controller = DiffMPCController(
        network=network,
        mpc_horizon=controller_config.mpc_horizon,
        dt=env_config.dt,
        state_limits=env_config.state_limits_mrp,
        control_limits=env_config.control_limits_torque,
        dynamics_params=env_config.spacecraft_dynamics_parameters,
    )
    controller = load_controller(controller, MODEL_FILE)

    env = get_env(env_name=ENV_NAME)
    trajectories, _ = n_rollouts_controller(
        env=env,
        controller=controller,
        key=key,
        horizon=controller_config.mpc_horizon,
        max_ep_steps=env.max_ep_steps,
        n_rollouts=N_ROLLOUTS,
        replan_freq=controller_config.replan_frequency,
        record_qr=False,
    )

    # compute_metrics converts observed errors to degrees and degrees/second.
    metrics = compute_metrics(
        trajectories,
        dt=env.dt,
        angle_threshold=float(jnp.rad2deg(env.theta_threshold)),
        omega_threshold=float(jnp.rad2deg(env.omega_threshold)),
        angle_tol_stability=float(jnp.rad2deg(env_config.theta_stability_tol)),
        omega_tol_stability=float(jnp.rad2deg(env_config.omega_stability_tol)),
        tail_length=100,
        label=ENV_NAME,
    )
    print_metrics(metrics)
    os.makedirs(EVAL_SAVE_PATH, exist_ok=True)
    metrics.to_csv(os.path.join(EVAL_SAVE_PATH, "metrics.csv"), index=False)


if __name__ == "__main__":
    main()
