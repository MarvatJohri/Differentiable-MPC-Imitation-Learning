"""

Contains code to simulate attitude control system in JAX

Abstracts from stuff cause I need it to look similar to the gym env


"""
import sys
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

# Enforce dtype of float64 for all jax arrays
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
jax.devices()


from quaternion_functions import q_left, q_conj, get_rotation, q_to_mrp, skew, quaternion_projection, quaternion_jacobian


import equinox as eqx




class SpacecraftEnvJax(eqx.Module):

    # Static params
    inertia: Float[Array, "3 3"] = eqx.field(static=True)
    inertia_inv: Float[Array, "3 3"] = eqx.field(static=True)
    mass: float = eqx.field(static=True)  

    # Env params
    dt: float = eqx.field(static=True)
    max_env_steps: int = eqx.field(static=True)
    max_torque: float = eqx.field(static=True)
    dyn_noise_std: float = eqx.field(static=True)

    # Reward params
    theta_threshold: float = eqx.field(static=True)
    omega_threshold: float = eqx.field(static=True)
    omega_penalty: float = eqx.field(static=True)
    action_penalty: float = eqx.field(static=True)
    goal_reward: float = eqx.field(static=True)
    theta_threshold_reward: float = eqx.field(static=True)

    # Limits
    state_limits: Float[Array, "7 2"] = eqx.field(static=True)
    control_limits: Float[Array, "3 2"] = eqx.field(static=True)
    max_omega_norm: float = eqx.field(static=True)
    max_action_norm: float = eqx.field(static=True)

    

    def __init__(self,
                 dynamics_params,
                 dt: Optional[float] = 0.1,
                 max_env_steps: Optional[int] = 1500,
                 state_limits: Array = None,
                 control_limits: Array = None,
                 max_torque: Optional[float] = 5e-5,
                 dyn_noise_std: Optional[float] = 1e-6,
                 theta_threshold: Optional[float] = 0.5,
                 omega_threshold: Optional[float] = 0.1,
                 theta_threshold_reward: Optional[float] = 10.0,
                 omega_penalty: Optional[float] = 0.1,
                 action_penalty: Optional[float] = 0.1,
                 goal_reward: Optional[float] = 10.0,
                 ):



        self.inertia = jnp.array(dynamics_params['inertia'], dtype=jnp.float64)
        self.inertia_inv = jnp.linalg.inv(self.inertia)
        self.mass = dynamics_params['mass']

        self.dt = dt
        self.max_env_steps = max_env_steps
        self.max_torque = max_torque
        self.dyn_noise_std = dyn_noise_std

        self.theta_threshold = theta_threshold
        self.omega_threshold = omega_threshold
        self.omega_penalty = omega_penalty
        self.action_penalty = action_penalty
        self.goal_reward = goal_reward
        self.theta_threshold_reward = theta_threshold_reward

        if state_limits is None:
            self.state_limits = jnp.array([[-1, 1]]*4 + [[-2, 2]]*3, dtype=jnp.float64)
        else:
            self.state_limits = jnp.array(state_limits, dtype=jnp.float64)
            
        if control_limits is None:
            self.control_limits = jnp.array([[-1, 1]] * 3, dtype=jnp.float64)
        else:
            self.control_limits = jnp.array(control_limits, dtype=jnp.float64)

        self.max_omega_norm = jnp.linalg.norm(self.state_limits[4:, 1])
        self.max_action_norm = jnp.linalg.norm(self.control_limits[:, 1])

    # Dynamics
    def state_dot(self, state: jnp.ndarray, control: jnp.ndarray, u_noise: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the time derivative of the state given the current state and control input.

        Args:
            state: Current state vector (7-dimensional).
            control: Control input vector (3-dimensional).
            u_noise: Noise to be added to the control input (3-dimensional).

        Returns:
            The time derivative of the state vector (7-dimensional).


        NOTE: For the purposes of the simulation, the noise added is assumed 
        to be part of the control input. I.e., we assume an external additional torque
        is applied to the system, not part of the actual control input.

        Might want to consider changing in the future to a more high fidelity model

        """

        q = state[:4]
        w = state[4:7]

        tau = control + u_noise

        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w)) # equivalent of q_mul(quat_conj(q), other_stuff)
        w_dot = self.inertia_inv @ (tau - skew(w) @ self.inertia @ w) # cross product a x b = a_skew_symmetric @ b

        state_dot = jnp.concatenate((q_dot, w_dot))

        return state_dot

    def rk4_step(self,
                 state: jnp.ndarray,
                 control: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Performs a single Runge-Kutta 4th order integration step.

        Args:
            state: Current state vector (7-dimensional).
            control: Control input vector (3-dimensional).
            key: Random key for generating noise.

        Returns:
            The updated state vector (7-dimensional).
        """
        def dynamics(state, key):
            key, subkey = jax.random.split(key)
            u_noise = jax.random.normal(subkey, shape=(3,), dtype=jnp.float64) * self.dyn_noise_std
            return self.state_dot(state, control, u_noise)

        # RK4 integration
        k1 = dynamics(state, key)
        k2 = dynamics(quaternion_projection(state + 0.5 * self.dt * k1), key)
        k3 = dynamics(quaternion_projection(state + 0.5 * self.dt * k2), key)
        k4 = dynamics(quaternion_projection(state + self.dt * k3), key)

        dx = (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return quaternion_projection(state + dx)

        


        



