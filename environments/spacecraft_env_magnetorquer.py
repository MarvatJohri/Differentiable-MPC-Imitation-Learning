"""

Contains code to simulate attitude control system in JAX

Abstracts from stuff cause I need it to look similar to the gym env


"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, NamedTuple
import time


HERE = Path(__file__).resolve().parent
MJ_WORK = HERE.parent
ROOT = MJ_WORK.parent
URANUS_MPC_PATH = ROOT / "uranus-mpc"


sys.path.append(str(URANUS_MPC_PATH))
sys.path.append(str(MJ_WORK))


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


#--------------------URANUS-MPC IMPORTS--------------------#


from utils.propagate import TrajectoryGenerator, sample_initial_states
from utils.coord_transforms import coord
from utils.plotting import save_data_to_pd_df

from dynamics.base_dynamics import Dynamics
from dynamics.spacecraft_dynamics import SpacecraftDynamics
from dynamics.orbit_dynamics import OrbitDynamics
from dynamics.planetary_params import Earth, Uranus, PlanetParams
from dynamics.magnetic_field import MagneticFieldModel

from utils.learning import Trainer, load_model, save_model




#----------------------------------------------------------


from quaternion_functions import q_left, q_conj, q_mul, skew, quaternion_projection, quaternion_jacobian, get_rotation


DYNAMICS_PARAMS = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
DYNAMICS_PARAMS["inertia_inv"] = jnp.linalg.inv(DYNAMICS_PARAMS["inertia"])


class EnvState(NamedTuple):
    state: jnp.ndarray
    goal_state: jnp.ndarray
    orbit_traj: jnp.ndarray
    orbit_xyz: jnp.ndarray
    b_traj: jnp.ndarray
    step_count: int
    step_key: jax.random.PRNGKey
    prev_angle_error: float # Added to track previous angle error for reward computation
    



class SpacecraftEnvNoisy(eqx.Module):

    # Static params
    inertia: Float[Array, "3 3"] = eqx.field(static=True)
    inertia_inv: Float[Array, "3 3"] = eqx.field(static=True)
    mass: float = eqx.field(static=True)  

    # Env params
    dt: float = eqx.field(static=True)
    max_ep_steps: int = eqx.field(static=True)
    max_torque: float = eqx.field(static=True)
    dyn_noise_std: float = eqx.field(static=True)

    # Reward params
    theta_threshold: float = eqx.field(static=True)
    omega_threshold: float = eqx.field(static=True)
    omega_penalty: float = eqx.field(static=True)
    omega_fail_penalty: float = eqx.field(static=True)
    action_penalty: float = eqx.field(static=True)
    goal_reward: float = eqx.field(static=True)
    theta_threshold_reward: float = eqx.field(static=True)

    # Limits
    state_limits: Float[Array, "7 2"] = eqx.field(static=True)
    control_limits: Float[Array, "3 2"] = eqx.field(static=True)
    max_omega_norm: float = eqx.field(static=True)
    max_action_norm: float = eqx.field(static=True)

    # Orbit params
    mag_field_horizon: int = eqx.field(static=True)
    const_mag_field: bool = eqx.field(static=True)
    b_noise_std: float = eqx.field(static=True)
    N_orbit: int = eqx.field(static=True)

    b_bias: jax.Array
    planet: PlanetParams
    orbit_dynamics: OrbitDynamics
    orbit_system: TrajectoryGenerator

    

    def __init__(self,
                 dynamics_params=DYNAMICS_PARAMS,
                 dt: Optional[float] = 0.1,
                 max_ep_steps: Optional[int] = 1500,
                 state_limits: Optional[jnp.ndarray] = None,
                 control_limits: Optional[jnp.ndarray] = None,
                 max_torque: Optional[float] = 5e-5,
                 dyn_noise_std: Optional[float] = 1e-6,
                 theta_threshold: Optional[float] = 0.5,
                 omega_threshold: Optional[float] = 0.1,
                 theta_threshold_reward: Optional[float] = 10.0,
                 omega_penalty: Optional[float] = 0.1,
                 omega_fail_penalty: Optional[float] = 50.0,
                 action_penalty: Optional[float] = 0.1,
                 goal_reward: Optional[float] = 10.0,
                 N_orbit: Optional[int] = 5000,
                 planet=Earth,
                 mag_field_horizon: Optional[int] = 1,
                 normalize_mag: Optional[bool] = True,
                 mag_field_scale: Optional[float] = 1e-4,
                 const_mag_field: Optional[bool] = False,
                 b_noise_std: Optional[float] = 0.0,
                 b_bias: Optional[jnp.ndarray] = jnp.zeros((3,)),
                 ):



        self.inertia = jnp.array(dynamics_params['inertia'], dtype=jnp.float64)
        self.inertia_inv = jnp.linalg.inv(self.inertia)
        self.mass = dynamics_params['mass']

        self.dt = dt
        self.max_ep_steps = max_ep_steps
        self.max_torque = max_torque
        self.dyn_noise_std = dyn_noise_std

        self.theta_threshold = theta_threshold
        self.omega_threshold = omega_threshold
        self.omega_penalty = omega_penalty
        self.omega_fail_penalty = omega_fail_penalty
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

        # Orbit imports, reference taken from Patrick Schwarz Uranus-MPC code
        self.mag_field_horizon = mag_field_horizon
        # self.mag_field_scale = mag_field_scale
        # self.normalize_mag = normalize_mag
        self.const_mag_field = const_mag_field
        self.b_noise_std = b_noise_std
        self.b_bias = b_bias

        self.planet = planet
        self.N_orbit = N_orbit
        self.orbit_dynamics = OrbitDynamics(Planet=self.planet)
        self.orbit_system = TrajectoryGenerator(dynamics=self.orbit_dynamics, dt=self.dt)

        # Make a check to ensure N_orbit has enough steps for horizon + num_steps
        if self.N_orbit < self.mag_field_horizon + self.max_ep_steps:
            raise ValueError(f"N_orbit ({self.N_orbit}) must be greater than or equal to episode length ({self.num_steps}) + mag_field_horizon ({self.mag_field_horizon}) = {self.mag_field_horizon + self.num_steps}.")




    # def sample_initial_states(self, batch_size, key, state_specs):
    
        
    #     # Reference: Patrick Schwarz Uranus-MPC code
        

    #     """
    #     Generates a batch of random initial states based on provided specs.

    #     Args:
    #         batch_size: Number of states to sample.
    #         key: JAX key.
    #         state_specs: List of dicts, e.g., 
    #             [{'name': 'pos', 'shape': (3,), 'dist': 'uniform', 'min': -1, 'max': 1},
    #             {'name': 'rot', 'shape': (4,), 'dist': 'quaternion'}]
    #     """
    #     states = []
        
    #     for spec in state_specs:
    #         key, subkey = jax.random.split(key)
    #         dist_type = spec.get('dist', 'uniform')
    #         shape = (batch_size,) + spec.get('shape', (1,))

    #         if dist_type == 'uniform':
    #             val = jax.random.uniform(subkey, shape=shape, 
    #                                 minval=spec['min'], maxval=spec['max'])
            
    #         elif dist_type == 'normal':
    #             val = spec.get('mean', 0.0) + spec.get('std', 1.0) * jax.random.normal(subkey, shape=shape)

    #         elif dist_type == 'quaternion':
    #             # Specialized sampler for unit quaternions
    #             q = jax.random.normal(subkey, shape=shape)
    #             val = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    #         elif dist_type == 'constant':
    #             return jnp.broadcast_to(jnp.array(spec['value']), shape)

    #         states.append(val.reshape(batch_size, -1))
    #     return jnp.concatenate(states, axis=-1)
    

    # def _sample_episode_context(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:

    #     init_key, target_key = jax.random.split(key)

    #     target_state = self._sample_state(1, target_key)[0]
    #     initial_state = self._sample_state(1, init_key)[0]

    #     return initial_state, target_state


    def _sample_episode_context(self, key):
    
    
        # init_key, orbit_key, target_key, mag_key, mag_new_key = jrandom.split(key, 5)
        init_key, orbit_key, target_key, mag_key = jax.random.split(key, 4)


        # Generate orbit and magnetic field trajectory
        r_planet = self.planet.radius
        orbit_specs = [
            {"dist": "uniform", "min": r_planet + 200.0, "max": r_planet + 400.0},
            {"dist": "uniform", "min": 0.0, "max": 0.0},
            {"dist": "uniform", "min": 0.0, "max": jnp.pi / 2},
            {"dist": "uniform", "min": 0.0, "max": 2 * jnp.pi},
            {"dist": "uniform", "min": 0.0, "max": 2 * jnp.pi},
            {"dist": "uniform", "min": 0.0, "max": 2 * jnp.pi},
        ]

        orbit_states = sample_initial_states(batch_size=1, key=orbit_key, state_specs=orbit_specs)[0]
        orbit_initial_state = coord.orbital_elements_to_pci(orbit_states, planet=self.planet)

        orbit_traj, _ = self.orbit_system.generate_trajectory_batch(
            initial_states=orbit_initial_state,
            target_states=None,
            key=orbit_key,
            num_steps=self.N_orbit,
            batch_size=1,
        )

        b_traj = self.orbit_dynamics.mag_model.generate_magnetometer_data(
            trajectory = orbit_traj,
            dt = self.dt,
            num_steps = self.N_orbit,
            key = mag_key,
            noise_std = self.b_noise_std,
            bias = self.b_bias,
            batch_size=1,
        )[0]

        if self.const_mag_field:
            b_traj = jnp.broadcast_to(b_traj[0], b_traj.shape)

        # Compute normalization constant - max absolute value across all components and timesteps
        # self.B_max = np.max(np.abs(np.array(b_traj))) 
        
        # Debug print (remove after verification)
        # print(f"B_max: {self.B_max:.2e} nT, B_range: [{np.min(b_traj):.2e}, {np.max(b_traj):.2e}]")


        # Generate initial and target states for spacecraft
        init_state_specs = [
                    {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                    {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
        ]

        target_state_specs = [
                    {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                    {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
        ]

        target_state = sample_initial_states(batch_size=1, key=target_key, state_specs=target_state_specs)[0]
        initial_state = sample_initial_states(batch_size=1, key=init_key, state_specs=init_state_specs)[0]



        """
        
        I have absolutely zero idea if I am supposed to do this but I'm copying his code so whatever
        
        """


        # # Get random b-field, always start at beginning of orbit (orbit start points already randomized)
        # b_batch_idx = jnp.arange(b_traj.shape[0])
        # mag_new_key, b_key  = jrandom.split(mag_new_key)
        # shuffled_indices = jrandom.permutation(b_key, b_batch_idx)
        # batch_indices = shuffled_indices[:1]  # Select the first index from the shuffled indices
        # b_batch = b_traj[batch_indices, :, :]

        context = {
            "orbit_elements": orbit_states,
            "orbit_initial_state": orbit_initial_state,
            "orbit_traj": orbit_traj,
            "orbit_xyz": orbit_traj[:, :, :3],
            "b_traj": b_traj,
            "initial_state": initial_state,
            "goal_state": target_state,
        }



        return context

 

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



    def reset(self, key: jax.random.PRNGKey, options=None):

        # Unlike normal gym envs, seed is REQUIRED
        # (Cause I use seeds in original collect trajectory)

        # key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        # initial_state, target_state = self._sample_episode_context(subkey)
        context = self._sample_episode_context(subkey)

        initial_state = context["initial_state"]
        target_state = context["goal_state"]
        orbit_traj = context["orbit_traj"]
        orbit_xyz = context["orbit_xyz"]
        b_traj = context["b_traj"]
        orbit_elements = context["orbit_elements"]
        orbit_initial_state = context["orbit_initial_state"]
        
        step_key = key
        step_count = 0

        init_obs = self._get_obs(initial_state, target_state)

        # Get initial angle error for reward computation
        q_err = init_obs[:4]
        angle_error = 2 * jnp.arccos(jnp.clip(jnp.abs(q_err[0]), 0, 1))


        # Make initial env state
        init_env_state = EnvState(state=initial_state,
                                  goal_state=target_state,
                                  step_count=step_count,
                                  step_key=step_key,
                                  prev_angle_error=angle_error,
                                  orbit_traj=orbit_traj,
                                  orbit_xyz=orbit_xyz,
                                  b_traj=b_traj,
                                  )  

        

        info = (initial_state, target_state) # The only info thats really needed

        # All returns will be in this format, so that it can be used in jax.jit and jax.vmap
        return init_env_state, init_obs, info


    def _orbit_position_interpolation(self, t, env_state: EnvState):

        # t -> time elapsed since episode's orbit began
        # in between k,k+1 time step since its used within rk4 integration

        i = env_state.step_count
        orbit = env_state.orbit_traj[0]

        r0 = orbit[i, 0:3]
        r1 = orbit[i + 1, 0:3]
        v0 = orbit[i, 3:6]
        v1 = orbit[i + 1, 3:6]

        # Hermite interpolation for position and velocity
        s = (t - i * self.dt) / self.dt  # normalized time between 0 and 1
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        r_interp = h00 * r0 + h10 * self.dt * v0 + h01 * r1 + h11 * self.dt * v1

        return r_interp


    def gravity_gradient_torque(self, t: float, env_state: EnvState, state: jnp.ndarray) -> jnp.ndarray:

        r_pci = self._orbit_position_interpolation(t, env_state)
        radius = jnp.linalg.norm(r_pci)

        q = state[:4]
        w = state[4:7]

        r_hat_body = get_rotation(q_conj(q)) @ (r_pci / radius)  

        tau_gg = 3 * self.planet.mu / radius**3 * jnp.cross(r_hat_body, self.inertia @ r_hat_body)

        return tau_gg
    

    def disturbance_deterministic(self, t: float, env_state: EnvState, state: jnp.ndarray) -> jnp.ndarray:
        """
        Deterministic external disturbance(s)


        Disturbances considered:

        1. Gravity gradient torque: T_gg = 3 * mu / r^3 * (r_hat x (I * r_hat))
        

        """

        
        tau_gg = self.gravity_gradient_torque(t, env_state, state)

        tau = tau_gg

        return tau

    def disturbance_stochastic(self, key: jax.random.PRNGKey, env_state: EnvState) -> jnp.ndarray:

        return jnp.zeros((3,))  # No stochastic disturbance for now, can be added later 



    # Dynamics
    def state_dot(self, 
                  t: float,
                  state: jnp.ndarray, 
                  control: jnp.ndarray, 
                  u_noise: jnp.ndarray, 
                  env_state: EnvState,
                  key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Computes the time derivative of the state given the current state and control input.

        Args:
            state: Current state vector (7-dimensional).
            t: Current time.
            control: Control input vector (3-dimensional).
            u_noise: Noise to be added to the control input (3-dimensional).

        Returns:
            The time derivative of the state vector (7-dimensional).

        """

        q = state[:4]
        w = state[4:7]

        disturbance_deterministic = self.disturbance_deterministic(t, env_state, state)
        disturbance_stochastic = self.disturbance_stochastic(key, env_state)

        tau = control + u_noise + disturbance_deterministic + disturbance_stochastic

        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w)) # equivalent of q_mul(quat_conj(q), other_stuff)
        w_dot = self.inertia_inv @ (tau - skew(w) @ self.inertia @ w) # cross product a x b = a_skew_symmetric @ b

        state_dot = jnp.concatenate((q_dot, w_dot))

        return state_dot

    def rk4_step(self,
                 t,
                 env_state: EnvState,
                 control: jnp.ndarray,
                 key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Performs a single Runge-Kutta 4th order integration step.

        Args:
            env_state: Current environment state.
            control: Control input vector (3-dimensional).
            key: Random key for generating noise.

        Returns:
            The updated state vector (7-dimensional).
        """
        def dynamics(t, state, key):
            key, subkey = jax.random.split(key)
            u_noise = jax.random.normal(subkey, shape=(3,), dtype=jnp.float64) * self.dyn_noise_std
            key, subkey = jax.random.split(key)
            return self.state_dot(t, state, control, u_noise, env_state, subkey)

        # RK4 integration
        state = env_state.state
        k1 = dynamics(t, state, key)
        k2 = dynamics(t + 0.5 * self.dt, quaternion_projection(state + 0.5 * self.dt * k1), key)
        k3 = dynamics(t + 0.5 * self.dt, quaternion_projection(state + 0.5 * self.dt * k2), key)
        k4 = dynamics(t + self.dt, quaternion_projection(state + self.dt * k3), key)

        dx = (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        new_state = quaternion_projection(state + dx)
        # Clip only angular velocity, quaternion taken care of inside quaternion_projection
        # new_state = jnp.concatenate([new_state[:4], jnp.clip(new_state[4:7], self.state_limits[4:, 0], self.state_limits[4:, 1])], axis=0)
        
        # Clipping ang vel SHOULDN'T be necessary
        
        return new_state


    # def _is_terminated(self):
    
    #     # Check if goal state is reached TECHNICALLY should also stop if it reaches failure states 
    #     # Like collisions

    #     # For now keep it in, else remove cause it makes code too clunky

    #     # terminated = self.reached

        

    #     # terminated = self.reached or self.failure
    #     terminated = self.failure

    #     # if self.reached:
    #     #     print(f"Terminating episode: goal reached at step {self.step_count}.")
    #     #     terminated = True

    #     return terminated

    # def _is_truncated(self):

    #     # Check if max steps reached or smthng idk what is put here actually

    #     truncated = self.step_count >= self.max_ep_steps

    #     return truncated


    def _get_reward(self, q_err, omega_err, action, new_state, prev_angle_error):
        # Angle error from quaternion (0 to pi)
        angle_error = 2 * jnp.arccos(jnp.clip(jnp.abs(q_err[0]), 0, 1))
        omega_error_norm = jnp.linalg.norm(omega_err)
        action_norm = jnp.linalg.norm(action)

        # Normalize for consistent scaling
        omega_error_normalized = omega_error_norm / self.max_omega_norm
        action_normalized = action_norm / self.max_action_norm

        reward = 0.0


        # Reward proposed in NASA paper
        ra = jnp.exp(-angle_error/(0.28*jnp.pi))


        not_reached_goal = angle_error > self.theta_threshold
        no_progress = angle_error > prev_angle_error

        reward = jnp.where(not_reached_goal & no_progress, ra - 1, ra)




        prev_angle_error = angle_error

        # Compute reward (negative costs)
        # reward -= self.quaternion_penalty * angle_error_normalized
        reward -= self.omega_penalty * omega_error_normalized
        reward -= self.action_penalty * action_normalized
        
     
        
        # Reward reaching threshold
        # reward = reward + jnp.where(~not_reached_goal, self.theta_threshold_reward, 0.0)

        # Reward reaching goal
        reward = reward + jnp.where(~not_reached_goal & (omega_error_norm < self.omega_threshold), self.goal_reward, 0.0)

        
        # Check if omega is beyond state limits, if so return a big negative reward
        
        # Add fail check
        failed = jnp.any(jnp.abs(new_state[4:]) > self.state_limits[4:, 1])

        reward = jnp.where(failed, -self.omega_fail_penalty, reward)

        return reward, prev_angle_error

    def _failed(self, state: jnp.ndarray) -> bool:
    
        # Check if it spins too much
        # Patrick hasn't mentioned this as a fail state
        # GPT suggests including it, might not actually be necessary


        # if np.linalg.norm(self.state[4:]) > 1.5:  # Arbitrary threshold for angular velocity
        #     self.failure = True
        #     return True



        return jnp.any(jnp.abs(state[4:]) > self.state_limits[4:, 1])

       


    def step(self, 
             env_state: EnvState, 
             action: jnp.ndarray):


        """
        
        Don't need to do reward computation
        
        """

        # Clip action to be within control limits
        action = jnp.clip(action, self.control_limits[:, 0], self.control_limits[:, 1])

        # Convert action to torque
        torque = action * self.max_torque

        # Unpack env state
        state = env_state.state
        goal_state = env_state.goal_state
        step_count = env_state.step_count
        step_key = env_state.step_key

        step_key, noise_key = jax.random.split(step_key)

        new_state = self.rk4_step(env_state.step_count*self.dt, 
                                  env_state, 
                                  torque, 
                                  noise_key)

        # Handle wrap around for quaternion 
        new_state = jax.lax.cond(new_state[0] < 0, lambda x: x.at[:4].set(-x[:4]), lambda x: x, new_state)

        obs = self._get_obs(new_state, goal_state)

        failed = self._failed(new_state)

        reward, prev_angle_error = self._get_reward(obs[:4], obs[4:7], action, new_state, env_state.prev_angle_error)

        step_count += 1

        terminated = failed
        truncated = step_count >= self.max_ep_steps

        done = terminated | truncated


        # Update env state
        new_env_state = env_state._replace(state=new_state,
                                           goal_state=goal_state,
                                           prev_angle_error=prev_angle_error,
                                           step_count=step_count,
                                           step_key=step_key)


        info = (new_state, goal_state, step_count, step_key)

        return new_env_state, obs, reward, done, info

    def step_autoreset(self, env_state: EnvState, action: jnp.ndarray):

        """Step + reset and continue env, to be used inside PPO"""
        new_env_state, obs, reward, done, info = self.step(env_state, action)

        # Fresh keys: one for the reset, one carried forward as the new step_key
        reset_key, step_key = jax.random.split(new_env_state.step_key)
        new_env_state = new_env_state._replace(step_key=step_key)


        # # Where done, swap in the fresh episode (works on every EnvState field)
        # new_env_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, new_env_state)
        # obs = jnp.where(done, reset_obs, obs)

        # return new_env_state, obs, reward, done, info

        def reset_episode(_):
            reset_state, reset_obs, _ = self.reset(reset_key)
            return reset_state, reset_obs

        def continue_episode(_):
            return new_env_state, obs

        new_env_state, obs = jax.lax.cond(
            done,
            reset_episode,
            continue_episode,
            operand=None,
        )

        return new_env_state, obs, reward, done, info


        