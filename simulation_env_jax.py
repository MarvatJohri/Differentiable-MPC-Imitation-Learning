"""

Contains code to simulate attitude control system in JAX

Abstracts from stuff cause I need it to look similar to the gym env


"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, NamedTuple
import time

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
# jax.config.update("jax_default_dtype_bits", "64")
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx

# Enforce dtype of float64 for all jax arrays
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
jax.devices()


from quaternion_functions import q_left, q_conj, q_mul, skew, quaternion_projection, quaternion_jacobian
from mj_utils import sample_state
from diffmpc_controller import DiffMPCController


DYNAMICS_PARAMS = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
DYNAMICS_PARAMS["inertia_inv"] = jnp.linalg.inv(DYNAMICS_PARAMS["inertia"])


class EnvState(NamedTuple):
    state: jnp.ndarray
    goal_state: jnp.ndarray
    step_count: int
    step_key: jax.random.PRNGKey
    



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
                 dynamics_params=DYNAMICS_PARAMS,
                 dt: Optional[float] = 0.1,
                 max_env_steps: Optional[int] = 1500,
                 state_limits: Optional[jnp.ndarray] = None,
                 control_limits: Optional[jnp.ndarray] = None,
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

    def _sample_episode_context(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:

        init_key, target_key = jax.random.split(key)

        target_state = sample_state(1, target_key)[0]
        initial_state = sample_state(1, init_key)[0]

        return initial_state, target_state

 

    def _get_obs(self, state: jnp.ndarray, goal_state: jnp.ndarray) -> jnp.ndarray:


        q = state[:4]
        w = state[4:7]

        q_goal = goal_state[:4]
        w_goal = goal_state[4:7]

        q_err = q_mul(q_conj(q_goal), q)  # equivalent of q_mul(quat_conj(q_goal), q)
        # Fix sign ambiguity 
        q_err = jnp.where(q_err[0] < 0, -q_err, q_err)
        # Normalize
        q_err = q_err / jnp.linalg.norm(q_err)
        w_err = w - w_goal

        return jnp.concatenate([q_err, w_err], axis=0)


    def reset(self, seed, options=None):

        # Unlike normal gym envs, seed is REQUIRED
        # (Cause I use seeds in original collect trajectory)

        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        initial_state, target_state = self._sample_episode_context(subkey)
        step_key = key
        step_count = 0

        # Make initial env state
        init_env_state = EnvState(state=initial_state,
                                  goal_state=target_state,
                                  step_count=step_count,
                                  step_key=step_key)

        init_obs = self._get_obs(initial_state, target_state)

        info = (initial_state, target_state) # The only info thats really needed

        # All returns will be in this format, so that it can be used in jax.jit and jax.vmap
        return init_env_state, init_obs, info

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

        new_state = quaternion_projection(state + dx)
        # Clip only angular velocity, quaternion taken care of inside quaternion_projection
        # new_state = jnp.concatenate([new_state[:4], jnp.clip(new_state[4:7], self.state_limits[4:, 0], self.state_limits[4:, 1])], axis=0)
        
        # Clipping ang vel SHOULDN'T be necessary
        
        return new_state
    

    def step(self, 
             env_state: EnvState, 
             action: jnp.ndarray):


        """
        
        Don't need to do reward computation
        
        """

        # Clip action to be within control limits
        action = jnp.clip(action, self.control_limits[:, 0], self.control_limits[:, 1])

        # Convert action to torque
        action = action * self.max_torque

        # Unpack env state
        state = env_state.state
        goal_state = env_state.goal_state
        step_count = env_state.step_count
        step_key = env_state.step_key

        step_key, noise_key = jax.random.split(step_key)

        new_state = self.rk4_step(state, action, noise_key)

        obs = self._get_obs(new_state, goal_state)

        step_count += 1

        # Update env state
        new_env_state = EnvState(state=new_state,
                                 goal_state=goal_state,
                                 step_count=step_count,
                                 step_key=step_key)

        info = (new_state, goal_state, step_count, step_key)

        return new_env_state, obs, info


        
@eqx.filter_jit
def generate_trajectory(env: SpacecraftEnvJax, 
                        controller: DiffMPCController,
                        expert_policy: Callable, 
                        key: jax.random.PRNGKey, 
                        beta: float,
                        max_ep_steps: int,
                        replan_freq: int = 1) -> Dict[str, jnp.ndarray]:


    max_torque = env.max_torque

    def scan_step(carry, _):

        env_state, obs, i, nominal_traj, nominal_cntrl, key = carry

        # Get expert action
        expert_action = expert_policy(obs)

        def controller_wrapper(operand):
            return controller(*operand)

        def no_update_nom_traj(operand):

            _, __, nominal_traj, nominal_cntrl = operand

            # Shift nominal trajectory and control by one step
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]

            return action, nominal_traj, nominal_cntrl

        # Get controller action
        controller_action, controller_nominal_traj, controller_nominal_cntrl = jax.lax.cond(
            i%replan_freq == 0,
            controller_wrapper,
            no_update_nom_traj,
            operand=(obs, env_state.goal_state, nominal_traj, nominal_cntrl))

        # Normalize controller action
        controller_action = controller_action / max_torque

        key, subkey = jax.random.split(key)
        use_expert = jax.random.uniform(subkey) < beta

        executed_action = jnp.where(use_expert, expert_action, controller_action)

        shifted_traj  = jnp.concatenate([nominal_traj[1:],  nominal_traj[-1:]],  axis=0)
        shifted_cntrl = jnp.concatenate([nominal_cntrl[1:], nominal_cntrl[-1:]], axis=0)
        new_nominal_traj  = jnp.where(use_expert, shifted_traj,  controller_nominal_traj)
        new_nominal_cntrl = jnp.where(use_expert, shifted_cntrl, controller_nominal_cntrl)

        # Run executed action on the environment using the step key inside the env_state
        new_env_state, new_obs, info = env.step(env_state, executed_action)

        new_carry = (new_env_state, new_obs, i+1, new_nominal_traj, new_nominal_cntrl, key)

        outputs = (obs, expert_action, nominal_traj, nominal_cntrl)

        return new_carry, outputs
            





    # Initialize stuff

    # Env
    key, subkey = jax.random.split(key)
    seed = jax.random.randint(subkey, shape=(), minval=0, maxval=2**32 - 1)
    init_env_state, init_obs, info = env.reset(seed)

    # Generate initial nominal trajectories
    key, subkey = jax.random.split(key)
    nominal_traj = jnp.tile(init_env_state.state, (controller.horizon + 1, 1))
    nominal_cntrl = 1e-3 * jax.random.normal(subkey, shape=(controller.horizon, controller.nu), dtype=jnp.float64)

    init_carry = (init_env_state, init_obs, 0, nominal_traj, nominal_cntrl, key)

    # Do the for loop
    final_carry, outputs = jax.lax.scan(scan_step, init_carry, xs=None, length=max_ep_steps)

    final_env_state, final_obs, final_i, final_nominal_traj, final_nominal_cntrl, final_key = final_carry
    observations, expert_actions, nominal_trajs, nominal_cntrls = outputs

    trajectory = (observations, init_env_state.goal_state, expert_actions, nominal_trajs, nominal_cntrls)

    return trajectory, final_key

    # return {
    #     "obs": observations,
    #     "goal_state": init_env_state.goal_state,
    #     "expert_actions": expert_actions,
    #     "nominal_traj": nominal_trajs,
    #     "nominal_cntrl": nominal_cntrls
    # }, final_key

@eqx.filter_jit
def generate_batch_trajectories(env: SpacecraftEnvJax,
                                controller: DiffMPCController,
                                expert_policy: Callable,
                                key: jax.random.PRNGKey,
                                beta: float,
                                max_ep_steps: int,
                                n_trajectories: int,
                                replan_freq: int = 1):

    return jax.vmap(generate_trajectory, in_axes=(None, None, None, 0, None, None, None))(env, controller, expert_policy, jax.random.split(key, n_trajectories), beta, max_ep_steps, replan_freq)


@eqx.filter_jit
def generate_n_trajectories(env: SpacecraftEnvJax, 
                            controller: DiffMPCController,
                            expert_policy: Callable, 
                            key: jax.random.PRNGKey, 
                            beta: float,
                            max_ep_steps: int,
                            n_trajectories: int,
                            replan_freq: int = 1):

    def scan_step(carry, _):
        key = carry
        key, subkey = jax.random.split(key)
        trajectory, new_key = generate_trajectory(env, controller, expert_policy, subkey, beta, max_ep_steps, replan_freq)
        return new_key, trajectory

    init_carry = key
    final_key, trajectories = jax.lax.scan(scan_step, init_carry, xs=None, length=n_trajectories)

    return trajectories, final_key

    
if __name__ == "__main__":

    # Stuff to test timing and stuff, things prof asked for

    env = SpacecraftEnvJax()        


    start = time.time()
    trajectory, key = generate_trajectory(env, None, None, jax.random.PRNGKey(0), beta=0.5, max_ep_steps=100)

        



