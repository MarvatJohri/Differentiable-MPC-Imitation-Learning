import sys
from pathlib import Path
from typing import Optional

URANUS_MPC_PATH = str((Path(__file__).resolve().parent / "uranus-mpc").resolve())


sys.path.append(URANUS_MPC_PATH)

import os
# Prevent memory pre-allocation for flexible memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9' # Use 70% of GPU memory default is '.75'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


os.environ['JAX_ENABLE_X64'] = 'false'

# Import packages
import numpy as np

import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
import jax.extend.backend as jeb

import matplotlib.pyplot as plt
import seaborn as sns
import marimo as mo
import pdb
import gc
import pickle
import pandas as pd

import optax

from utils.propagate import TrajectoryGenerator, sample_initial_states
from dynamics.quaternion_functions import S, q_left, q_conj, get_rotation, q_to_mrp
from utils.coord_transforms import coord
from utils.plotting import save_data_to_pd_df

from dynamics.base_dynamics import Dynamics
from dynamics.spacecraft_dynamics import SpacecraftDynamics
from dynamics.orbit_dynamics import OrbitDynamics
from dynamics.planetary_params import Earth, Uranus
from dynamics.magnetic_field import MagneticFieldModel

from utils.learning import Trainer, load_model, save_model


import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Jax configuration
jax.config.update("jax_default_matmul_precision", "highest") # required for vmap to work properly, weird bug
# jax.config.update("jax_enable_x64", True) # enables 64-bit precision

# Seaborn plotting style
sns.set_theme(context='paper', style='whitegrid', font='serif', font_scale=1.5)

jax.devices() # verify GPU is available



class SpacecraftEnv(gym.Env):

    def __init__(
            self,
            dynamics: Dynamics,
            dt: Optional[float] = 0.1,
            num_steps: Optional[int] = 100,
            render_mode: Optional[str] = None,
            state_limits: Optional[np.ndarray] = None,
            control_limits: Optional[np.ndarray] = None,
            mag_field_horizon: Optional[int] = 1,
            normalize_mag: Optional[bool] = True,
            mag_field_scale: Optional[float] = 1e-4,
            dyn_noise_std: Optional[float] = 1e-6,
            const_mag_field: Optional[bool] = False,
            planet=Earth,
            N_orbit: Optional[int] = 5000,
            b_noise_std: Optional[float] = 0.0,
            b_bias: Optional[np.ndarray] = np.zeros((3,)),
            theta_tol: Optional[float] = 0.5,
            omega_tol: Optional[float] = 0.1,
            omega_penalty: Optional[float] = 0.1,
            theta_proximity_tol: Optional[float] = 1,
            quaternion_penalty: Optional[float] = 10.0,
            action_penalty: Optional[float] = 0.1,
            action_parallel_penalty: Optional[float] = 0.1,
            action_perpendicular_penalty: Optional[float] = 1.0,
            goal_reward: Optional[float] = 10.0,
            proximity_penalty: Optional[float] = 1.0,
            theta_progress_coeff: Optional[float] = 1.0,
            omega_progress_coeff: Optional[float] = 1.0,
            ):
        super().__init__()
        self.dynamics = dynamics
        self.dt = dt
        self.num_steps = num_steps
        self.render_mode = render_mode
        self.mag_field_horizon = mag_field_horizon
        self.normalize_mag = normalize_mag 
        self.mag_field_scale = mag_field_scale
        self.const_mag_field = const_mag_field
        

        if state_limits is None:
            self.state_limits = np.asarray([[-1, 1]]*4 + [[-2,2]]*3, dtype=np.float32) # default state limits for quaternion and angular velocity
        else:
            self.state_limits = np.asarray(state_limits, dtype=np.float32)
        if control_limits is None:
            self.control_limits = np.asarray(0.8*np.array([[-1, 1]] * self.dynamics.num_controls), dtype=np.float32) # default control limits for dipole control
        else:
            self.control_limits = np.asarray(control_limits, dtype=np.float32)
        

        self.max_omega_norm = np.linalg.norm(self.state_limits[4:, 1])  # Assuming the last three entries are angular velocity limits
        self.max_action_norm = np.linalg.norm(self.control_limits[:, 1])  # Assuming the last three entries are control limits


        self.dyn_noise_std = dyn_noise_std



        self.planet = planet 
        self.b_noise_std = b_noise_std
        # Default b_bias is zeros
        self.b_bias = np.zeros((3,)) if b_bias is None else np.asarray(b_bias)


        # Tolerances for checking if goal reached
        self.theta_tol = theta_tol
        self.omega_tol = omega_tol
        self.theta_proximity_tol = theta_proximity_tol
        self.proximity_penalty = proximity_penalty
        # Reward parameters
        self.omega_penalty = omega_penalty
        self.quaternion_penalty = quaternion_penalty
        self.action_penalty = action_penalty
        self.action_parallel_penalty = action_parallel_penalty
        self.action_perpendicular_penalty = action_perpendicular_penalty
        self.goal_reward = goal_reward
        self.theta_progress_coeff = theta_progress_coeff
        self.omega_progress_coeff = omega_progress_coeff

        # Stuff related to orbit dynamics and magnetic field generation
        self.N_orbit = N_orbit
        self.orbit_dynamics = OrbitDynamics(Planet=self.planet)
        self.orbit_system = TrajectoryGenerator(dynamics=self.orbit_dynamics, dt=self.dt)

        # Make a check to ensure N_orbit has enough steps for horizon + num_steps
        if self.N_orbit < self.mag_field_horizon + self.num_steps:
            raise ValueError(f"N_orbit ({self.N_orbit}) must be greater than or equal to episode length ({self.num_steps}) + mag_field_horizon ({self.mag_field_horizon}) = {self.mag_field_horizon + self.num_steps}.")


        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.num_states = 7 # [q0, q1, q2, q3, wx, wy, wz] quat is scalar first
        self.num_obs = 7 
        self.num_controls = self.control_limits.shape[0]


        self.action_space = spaces.Box(low=self.control_limits[:, 0], high=self.control_limits[:, 1], dtype=np.float32)

        # Obs are errors
        # q_err[0] is maintained positive and q_err is normalized
        # Thus use [0, 1] for q_err[0] and [-1, 1] for q_err[1:3]
        # omega_err worst case is [-2, 2] for each component, thus use [-4 ,4] for each component of omega_err

        # Also pass in normalized magnetic field vector in observation space, which is [-1, 1] for each component


        obs_low_state = np.array([0, -1, -1, -1, -4, -4, -4], dtype=np.float32)
        obs_high_state = np.array([1, 1, 1, 1, 4, 4, 4], dtype=np.float32)

        if self.normalize_mag:
            # mag field over horizon
            obs_low_mag = -1 * np.ones((self.mag_field_horizon*3,), dtype=np.float32)
            obs_high_mag = 1 * np.ones((self.mag_field_horizon*3,), dtype=np.float32)
        else:
            # mag field over horizon
            obs_low_mag = -np.inf * np.ones((self.mag_field_horizon*3,), dtype=np.float32)
            obs_high_mag = np.inf * np.ones((self.mag_field_horizon*3,), dtype=np.float32)

        obs_low = np.concatenate([obs_low_state, obs_low_mag])
        obs_high = np.concatenate([obs_high_state, obs_high_mag])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)




        # There's probably a smarter way to do this but I'm lazy
        self.state_space = spaces.Box(low=self.state_limits[:, 0], high=self.state_limits[:, 1], dtype=np.float32)
    

        # Initial stuff
        self.state = None
        self.goal_state = None
        self.episode_initial_context = None
        self._step_key = None
        self.step_count = 0

        self.prev_angle_error = None
        self.prev_omega_error = None


        # Flag to check if goal has been reached
        self.reached = False

        self.failure = False


        # jit the step function
        self.rk4_step_jit = jax.jit(self.rk4_step)


    # def sample_state(self, key):
    #     # No reason to change this yet
    #     specs = [
    #         {"shape": (4,), "dist": "quaternion"},
    #         {"shape": (3,), "dist": "uniform", "min": 0.0, "max": 0.0},
    #     ]
    #     return sample_initial_states(batch_size=1, key=key, state_specs=specs)[0]
    

    def debug_episode(self):
        """Run one episode and print diagnostics."""
        obs, _ = self.reset()
        
        total_reward = 0
        quat_costs = []
        omega_norms = []
        action_norms = []
        
        for i in range(100):  # Just first 100 steps
            # Random action
            action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(action)
            
            # Get current errors
            q_err = obs[:4]  # Assuming obs is [q_err, omega_err]
            omega_err = obs[4:]
            
            quat_cost = 1.0 - np.abs(q_err[0])
            omega_norm = np.linalg.norm(omega_err)
            action_norm = np.linalg.norm(action)
            
            quat_costs.append(quat_cost)
            omega_norms.append(omega_norm)
            action_norms.append(action_norm)
            total_reward += reward
            
            if i % 20 == 0:
                print(f"Step {i}: quat_cost={quat_cost:.3f}, omega={omega_norm:.3f}, action={action_norm:.3f}, reward={reward:.3f}")
        
        print(f"\n=== Summary ===")
        print(f"Total reward (100 steps): {total_reward:.1f}")
        print(f"Quat cost:  min={min(quat_costs):.3f}, max={max(quat_costs):.3f}, mean={np.mean(quat_costs):.3f}")
        print(f"Omega norm: min={min(omega_norms):.3f}, max={max(omega_norms):.3f}, mean={np.mean(omega_norms):.3f}")
        print(f"Action norm: min={min(action_norms):.3f}, max={max(action_norms):.3f}, mean={np.mean(action_norms):.3f}")
        print(f"\nExpected per-step reward breakdown:")
        print(f"  -1.0 * quat_cost:  {-1.0 * np.mean(quat_costs):.3f}")
        print(f"  -0.1 * omega_norm: {-0.1 * np.mean(omega_norms):.3f}")
        print(f"  -0.1 * action_norm: {-0.1 * np.mean(action_norms):.3f}")


    def _sample_episode_context(self, key):


        # init_key, orbit_key, target_key, mag_key, mag_new_key = jrandom.split(key, 5)
        init_key, orbit_key, target_key, mag_key = jrandom.split(key, 4)


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
            b_init = b_traj[0]
            b_traj = jnp.tile(b_init, (self.N_orbit, 1))

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

        # Return stuff in np array format

        context = {k: np.array(v) for k, v in context.items()}



        return context



    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if seed is None:
            seed = int(self.np_random.integers(0, 2**32 - 1))

        key = jrandom.PRNGKey(seed)
        key, context_key = jrandom.split(key)

        self.episode_initial_context = self._sample_episode_context(context_key)
        self.state = self.episode_initial_context["initial_state"].astype(np.float64)
        self.goal_state = self.episode_initial_context["goal_state"].astype(np.float64)
        self._step_key = key
        self.step_count = 0

        obs = self._get_obs()
        # q_err = obs[:4]
        # omega_err = obs[4:7]
        # angle_err = 2*np.arccos(np.clip(np.abs(q_err[0]), 0, 1))  
        # self.prev_angle_error = angle_err
        # self.prev_omega_error = np.linalg.norm(omega_err)
        info = {
            "initial_state": self.episode_initial_context["initial_state"],
            "goal_state": self.goal_state,
            "orbit_elements": self.episode_initial_context["orbit_elements"],
            "orbit_initial_state": self.episode_initial_context["orbit_initial_state"],
            "orbit_traj": self.episode_initial_context["orbit_traj"],
            "orbit_xyz": self.episode_initial_context["orbit_xyz"],
            "b_traj": self.episode_initial_context["b_traj"],
        }

        # Reset flags
        self.reached = False
        self.failure = False


        return obs, info
    




    def rk4_step(self, state, control, t, external_dynamics_param, key, noise_std):
        # Ctrl C Ctrl V from Patrick's code
        def dynamics_wrapper(s,k):
            key, noise_key = jrandom.split(k)
            u_noise = jrandom.normal(noise_key, shape=control.shape) * noise_std
            return self.dynamics.state_dot(s, control, t, external_dynamics_param, u_noise)

        # Note: quaternion projection is just identity unless defined in Dynamics class
        k1 = dynamics_wrapper(state,key)
        k2 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k1),key)
        k3 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k2),key)
        k4 = dynamics_wrapper(self.dynamics.quaternion_projection(state + self.dt * k3),key)
    
        dx = (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # dx = self.dt*dynamics_wrapper(state) # euler

        # key1, key = jrandom.split(key)
        # noise = jrandom.normal(key, shape=state.shape) * noise_std * jnp.sqrt(self.dt)

        return self.dynamics.quaternion_projection(state + dx)
    


    def rk4_step_wrapper(self, state, control, t, external_dynamics_param, key, noise_std):
        
        # Patrick's rk4 step uses jax
        # Wrapper to use environment's parameters but convert to jnp for use in jax
        # And then convert back to np array for use in environment
        
        state_jnp = jnp.asarray(state)
        control_jnp = jnp.asarray(control)
        external_dynamics_param_jnp = jnp.asarray(external_dynamics_param)


        # state_next_jnp = self.rk4_step(state_jnp, control_jnp, t, external_dynamics_param_jnp, key, noise_std)
        state_next_jnp = self.rk4_step_jit(state_jnp, control_jnp, t, external_dynamics_param_jnp, key, noise_std)
        return np.array(state_next_jnp)
        
        




    def _get_info(self):

        # Figure out what to put here

        # Return stuff from context

        return {
            "initial_state": self.episode_initial_context["initial_state"],
            "goal_state": self.goal_state,
            "current_step": self.step_count,
            "current_state": self.state,
            "b_traj": self.episode_initial_context["b_traj"],
        }
    


    def quat_conj(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(self, q1, q2):
        # scalar-first quaternion multiplication
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def normalize_quat(self, q):
        norm = np.linalg.norm(q)
        # if norm < 1e-10:
        #     return np.array([1.0, 0.0, 0.0, 0.0])
        assert norm > 1e-10, "Quaternion norm is too small to normalize."
        return q / norm

    def quat_error(self, q, qg):
        # q_err = qg^* ⊗ q
        q_err = self.quat_mul(self.quat_conj(qg), q)

        # fix sign ambiguity
        if q_err[0] < 0:
            q_err = -q_err
        return q_err
    


    def _get_obs(self):

        # Need to decide if I just pass the state, i.e., the full quaternion and ang vel
        # Or if I pass error coordinates

        # Further need to decide if I should pass MRP if I'm using error coordinates

        # Patrick suggested against using MRP for error coordinates
        # Will stick to q_err and omega_err

        q_err = self.quat_error(self.state[:4], self.goal_state[:4])
        # Normalize
        q_err = self.normalize_quat(q_err)
        # # Account for sign
        # if q_err[0] < 0:
        #     q_err = -q_err
        omega_err = self.state[4:] - self.goal_state[4:]
            

        # Normalized magnetic field vector over defined horizon
        b_traj = self.episode_initial_context["b_traj"][self.step_count: self.step_count + self.mag_field_horizon]
        
        if self.normalize_mag:
            # b_normalized = b_traj / self.B_max if self.B_max > 0 else b_traj  # Avoid division by zero
            b_normalized = b_traj * self.mag_field_scale
        else:
            b_normalized = b_traj

        b_normalized = b_normalized.flatten()  # Flatten to 1D array for observation

        return np.concatenate([q_err, omega_err, b_normalized]).astype(np.float32)

    
    def _reached_goal(self,q_err,omega_err):

        # Extract theta from q_err
        theta = 2*np.arccos(np.clip(np.abs(q_err[0]), 0, 1))
        
        reached = theta < self.theta_tol and np.linalg.norm(omega_err) < self.omega_tol

        # reached = np.rad2deg(theta) < self.theta_tol_deg and np.linalg.norm(omega_err) < self.omega_tol

        return reached
    
    def _failed(self):

        # Check if it spins too much
        # Patrick hasn't mentioned this as a fail state
        # GPT suggests including it, might not actually be necessary


        # if np.linalg.norm(self.state[4:]) > 1.5:  # Arbitrary threshold for angular velocity
        #     self.failure = True
        #     return True

        return False



    def _get_reward(self, q_err, omega_err, action):

        # Look at goal tracking reward functions
        # Pretty sure its usually -norm(error), +10 if reached goal, -10 if failed, etc

        # Check if it reached goal or not

        # if self._failed():
        #     return -self.goal_reward  # Negative reward for failure

        # if self._reached_goal(q_err,omega_err):
            
        #     # Temporary hard coding of goal reward
        #     return self.goal_reward


        # reward = 0

        # # Cost for q_err -> if quaternions align, then no error, else +1 if orthogonal
        # # d = 1 - abs(q^Tq_goal) = 1 - abs(q_err[0])
        # # reward = -cost = abs(q_err[0]) - 1

        # reward += self.quaternion_penalty * (np.abs(q_err[0]) - 1)

        # # Add cost for omega_err

        # reward -= self.omega_penalty * np.linalg.norm(omega_err)

        # # Cost for action magnitude

        # reward -= self.action_penalty * np.linalg.norm(action)  # Assuming action is part of the state, adjust as needed

        # # TODO: Consider additional cases, collisions, failures, etc


        reward = 0

        angle_error = 2*np.arccos(np.clip(np.abs(q_err[0]), 0, 1))  
        omega_error_norm = np.linalg.norm(omega_err)
        action_norm = np.linalg.norm(action)

        # Normalize error terms
        angle_error_normalized = angle_error / np.pi
        omega_error_normalized = omega_error_norm / self.max_omega_norm

        action_normalized = action_norm / self.max_action_norm

        

        # error costs
        reward -= self.quaternion_penalty * (angle_error_normalized ** 2)  # Squared angle error
        reward -= self.omega_penalty * (omega_error_normalized ** 2)  # Squared angular velocity error

        # action costs
        reward -= self.action_penalty * (action_normalized ** 2)  # Squared action magnitude
        b_field = self.episode_initial_context["b_traj"][self.step_count]
        b_field_norm = np.linalg.norm(b_field)
        # if b_field_norm > 1e-6:  # Avoid division by zero
        #     b_hat = b_field / b_field_norm
        #     action_parallel = np.dot(action, b_hat) * b_hat
        #     action_perpendicular = action - action_parallel

        #     action_parallel_norm = np.linalg.norm(action_parallel)
        #     action_perpendicular_norm = np.linalg.norm(action_perpendicular)

        #     # Normalize
        #     action_parallel_normalized = action_parallel_norm / self.max_action_norm
        #     action_perpendicular_normalized = action_perpendicular_norm / self.max_action_norm

        #     reward -= self.action_parallel_penalty * (action_parallel_normalized ** 2)
        #     reward -= self.action_perpendicular_penalty * (action_perpendicular_normalized ** 2)

        if self.prev_angle_error is not None:

            # Might want to add progress reward
            # Keep commented for now, add as necessary

            angle_improvement = self.prev_angle_error - angle_error 
            omega_improvement = self.prev_omega_error - omega_error_norm 

            # progress reward
            # reward += self.theta_progress_coeff * angle_improvement
            # reward += self.omega_progress_coeff * omega_improvement
            
            # Try exponential progress reward
            
            reward += self.theta_progress_coeff * np.exp(-angle_error/(0.14*2*np.pi))
            if angle_improvement <= 0:
                reward -= 1 


        # Change prev error vals
        self.prev_angle_error = angle_error
        self.prev_omega_error = omega_error_norm

        # Reward for coming near goal
        if angle_error < self.theta_proximity_tol:
            proximity_normalized = angle_error / self.theta_proximity_tol
            reward += self.proximity_penalty * (1 - proximity_normalized)  # Reward
            # reward += self.proximity_penalty * (self.theta_proximity_tol - angle_error)  


        # Reward for reaching goal
        if self.reached:
            reward += self.goal_reward



        return reward


    # def _get_reward(self, q_err, omega_err, action):
    #     # Simple: just minimize angle error, small action regularization
    #     angle_error = 2 * np.arccos(np.clip(np.abs(q_err[0]), 0.0, 1.0))
        
    #     # Reward is negative angle error (in degrees for interpretability)
    #     angle_deg = np.degrees(angle_error)
        
    #     # Simple linear reward: 0 at 180°, +180 at 0°
    #     reward = 180.0 - angle_deg
        
    #     # Small action penalty (don't dominate)
    #     action_norm = np.linalg.norm(action)
    #     reward -= 0.1 * action_norm
        
    #     # Bonus for being close (within 10°)
    #     if angle_deg < 10.0:
    #         reward += 50.0
        
    #     # Bonus for reaching goal
    #     if angle_deg < 5.0 and np.linalg.norm(omega_err) < 0.01:
    #         reward += 100.0
        
    #     return reward


        

    
    def _is_terminated(self):

        # Check if goal state is reached TECHNICALLY should also stop if it reaches failure states 
        # Like collisions

        # For now keep it in, else remove cause it makes code too clunky

        # terminated = self.reached

        

        # terminated = self.reached or self.failure
        terminated = self.failure



        return terminated

    def _is_truncated(self):

        # Check if max steps reached or smthng idk what is put here actually

        truncated = self.step_count >= self.num_steps

        return truncated





    def step(self, action):
        action = np.clip(action, self.control_limits[:, 0], self.control_limits[:, 1])

        # For external dynamics param we need to get bfield at current step
        step_idx = self.step_count

        b_traj = self.episode_initial_context["b_traj"][step_idx]
        self._step_key, noise_key = jrandom.split(self._step_key)

        self.state = self.rk4_step_wrapper(self.state, action, step_idx * self.dt, b_traj, noise_key, self.dyn_noise_std)

        # Enforce state limits
        # self.state = np.clip(self.state, self.state_limits[:, 0], self.state_limits[:, 1])

        # Only clip angular velocity part of state, not quaternion
        self.state[4:] = np.clip(self.state[4:], self.state_limits[4:, 0], self.state_limits[4:, 1])

        # Normalize quaternion part of state
        # self.state[:4] = self.normalize_quat(self.state[:4])

        # Check sign of quaternion and flip if necessary
        if self.state[0] < 0:
            self.state[:4] = -self.state[:4]


        obs = self._get_obs()

        q_err = obs[0:4]
        omega_err = obs[4:7]
        b_normalized = obs[7:]

        # reward = self._get_reward(q_err, omega_err)

        self.reached = self._reached_goal(q_err, omega_err)
        if self.reached:
            print(f"Goal reached at step {self.step_count}!")
        self.failure = self._failed()


        #------------------Reward Computation----------------------
        if self.prev_angle_error is not None:
            angle_improvement = self.prev_angle_error - 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1))
            omega_improvement = self.prev_omega_error - np.linalg.norm(omega_err)
        else:
            angle_improvement = 0.0
            omega_improvement = 0.0
        reward = self._get_reward(q_err, omega_err, action)

        self.step_count += 1
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        if self.step_count % 100 == 0 or self.step_count == 1:
            # q_err = self._get_obs()[:4]
            # theta_err = 2*np.arccos(np.clip(np.abs(q_err[0]), 0, 1))
            theta_err = np.degrees(2*np.arccos(np.clip(np.abs(q_err[0]), 0, 1)))
            # omega_err = self._get_obs()[4:]
            # b_normalized = self._get_obs()[7:]
            print(f"Step {self.step_count}: reward={reward:.3f}, angle_imp={angle_improvement:.4f}, omega_imp={omega_improvement:.4f}, theta_err={theta_err:.3f}, omega_err_norm={np.linalg.norm(omega_err):.3f}, b_normalized_norm={np.linalg.norm(b_normalized):.3f}, action_norm={np.linalg.norm(action):.3f}")


        info = {
            "initial_state": self.episode_initial_context["initial_state"],
            "current_step": self.step_count,
            "current_state": self.state,
            "goal_state": self.goal_state,
            "b_t": b_traj,
            "orbit_xyz_t": self.episode_initial_context["orbit_xyz"][0, self.step_count - 1],
        }

        
        return obs, reward, terminated, truncated, info




def rollout_traj_env(env: SpacecraftEnv, actions: np.ndarray, initial_state: np.ndarray):

    traj = [np.asarray(initial_state)]
    state = initial_state
    for action in actions:
        state = env.step(action)[0]  # Get the observation after taking the step
        traj.append(np.asarray(state))
    return np.stack(traj)



def test_trajectory_rollout(env, num_steps=50):
    seed = 123
    
    # Reset and extract everything we need
    obs, info = env.reset(seed=seed)
    init_state = info["initial_state"].copy()
    b_traj = info["b_traj"].copy()
    
    # Grab the internal key BEFORE any steps
    step_key = env._step_key
    
    # Generate controls
    ctrl_key = jrandom.PRNGKey(999)
    controls = 0.3 * jrandom.normal(ctrl_key, shape=(num_steps, 3))
    controls = jnp.clip(controls, -0.8, 0.8)
    
    # === Manual rollout using extracted key ===
    manual_trajectory = [init_state.copy()]
    state = jnp.array(init_state)
    
    for i in range(num_steps):
        b_t = b_traj[i]
        step_key, noise_key = jrandom.split(step_key)
        
        next_state = env.rk4_step(
            state, 
            controls[i], 
            t=i * env.dt, 
            external_dynamics_param=jnp.array(b_t), 
            key=noise_key, 
            noise_std=env.dyn_noise_std
        )
        next_state = jnp.clip(next_state, env.state_limits[:, 0], env.state_limits[:, 1])
        manual_trajectory.append(np.array(next_state))
        state = next_state
    
    manual_trajectory = np.array(manual_trajectory)
    
    # === Gym env rollout ===
    env.reset(seed=seed)  # Reset again to same state
    env_trajectory = [env.state.copy()]
    
    for i in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(np.array(controls[i]))
        env_trajectory.append(env.state.copy())
        if terminated or truncated:
            break
    
    env_trajectory = np.array(env_trajectory)
    
    # Compare
    diff = np.abs(env_trajectory - manual_trajectory)
    print(f"Max difference: {diff.max():.2e}")
    
    return env_trajectory, manual_trajectory






if __name__ == "__main__":







    """

        CODE COPIED FROM PATRICK I HAVE NO IDEA WHAT THE HELL IS GOING ON YET

    """










    # # Define parameters that must be consistent for orbit and spacecraft attitude dynamics
    # dt = 0.1
    # planet = Earth # Earth, Uranus

    # # Load learned magnetic field models
    # if planet is Earth:
    #     model_s, _ = load_model(filename='models/earth_b_4d.eqx') 
    # elif planet is Uranus:
    #     model_s, _ = load_model(filename='models/uranus_b_4d.eqx')


    # # ==================== Orbit Dynamics =================== #
    # N_orbit = 5000
    # orbit_dynamics = OrbitDynamics(Planet=planet)
    # orbit_system = TrajectoryGenerator(dynamics=orbit_dynamics, dt=dt) # set noise_std=0 for deterministic trajectory
    # key_orbit = jrandom.key(1234)
    # r_planet = orbit_dynamics.planet.radius

    # # Generate random initial states
    # orbit_init_state_specs = [
    #             {'dist': 'uniform', 'min': r_planet+200, 'max': r_planet+400}, # a (semimajor axis)
    #             {'dist': 'uniform', 'min': 0.0, 'max': 0.0}, # eccentricity
    #             {'dist': 'uniform', 'min': 0, 'max': jnp.pi/2}, # inclination
    #             {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # right ascension of the ascending node
    #             {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # argument of periapsis
    #             {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # true anomaly
    # ]

    # orbit_batch_size = 200
    # orbit_states = sample_initial_states(batch_size=orbit_batch_size, key=key_orbit, state_specs=orbit_init_state_specs)
    # orbit_init_states = coord.orbital_elements_to_pci(orbit_states,planet=planet)

    # # Generate orbits
    # start_orbit_batch = time.time()
    # orbit_trajs, _ = orbit_system.generate_trajectory_batch(initial_states=orbit_init_states, num_steps=N_orbit, key=key_orbit, batch_size=orbit_batch_size)
    # orbit_trajs.block_until_ready()
    # print(f"Time: {time.time() - start_orbit_batch:.2f}s")
    # #t = jnp.linspace(0, dt*N_orbit, N_orbit+1)


    # # Generate true magnetic field data corresponding to orbits
    # chunk_size = 100 # need to chunk otherwise it exhausts memory

    # # Set noise and bias to zero because we are using these as our "True" values
    # noise_std_mag = 0 #[nT]
    # bias = jnp.array([0, 0, 0])

    # # In PCI coords
    # b_trajs = orbit_dynamics.mag_model.generate_magnetometer_data(orbit_trajs, dt, N_orbit, key_orbit, noise_std_mag, bias, chunk_size)




    # # ==================== Spacecraft Dynamics =================== #
    # spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)

    # system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=dt) #Probably don't need to use this

    # key = jrandom.key(4567)


    # """
    #     I think stuff here is for starting a simulation 
    # """

    # # Set general conditions for single runs
    # key1, key2, key3, init_cntrl_key = jrandom.split(key, 4) # split to try out a couple different noises before running batch

    # # Define standard deviation of random disturbance torque in simulations
    # noise_std_dyn = 1e-6 # [Nm]

    # batch_i = 86 # Select orbit/magnetic field/target state from one of the randomly sampled ones from batch

    # # Choose b-field
    # # b = b_batch[batch_i]

    # # Select orbit that corresponds with b field select to pass to learned model in dyn_params_est
    # # orbit_xyz = orbit_xyz_batch[batch_i]

    # state_limits = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
    # control_limits = 0.8*jnp.array([[-1, 1]] * spacecraft_dynamics.num_controls) # limits for dipole cntrl

    # init_state = jnp.array([1, 0, 0, 0, 0.0, 0.0, 0.0])
    # # target_state = target_states[batch_i]

    # quat_start = spacecraft_dynamics.params["quat_start"]
    # # quat_goal = target_state[quat_start:quat_start+4]



    """
    
    MY OWN STUFF ABANDON ALL SEMBLANCE OF LOGIC YE WHO SCROLL HERE
    
    """


    dt = 0.1
    planet = Earth # Earth, Uranus

    N_orbit = 5000

    # Path to magnetic field models
    model_path = URANUS_MPC_PATH + '/models/'

    # Load learned magnetic field models
    if planet is Earth:
        model_s, _ = load_model(filename=model_path + '/earth_b_4d.eqx') 
    elif planet is Uranus:
        model_s, _ = load_model(filename=model_path + '/uranus_b_4d.eqx')

    noise_std_mag = 0 #[nT]
    bias = np.array([0, 0, 0])

    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)

    # Here state/obs is actually just quaternion and angular velocity
    # Vary quaternion from -1 to 1, ang vel from -2 to 2 rad/s, control from -1 to 1 A-m^2
    
    state_limits = np.array([[-1, 1]]*4 + [[-2,2]]*3)
    control_limits = 0.8*np.array([[-1, 1]] * spacecraft_dynamics.num_controls) # limits for dipole cntrl

    noise_std_dyn = 1e-6 # [Nm]


    env1 = SpacecraftEnv(
        spacecraft_dynamics, 
        dt=dt, 
        num_steps=200, 
        state_limits=state_limits, 
        control_limits=control_limits,
        dyn_noise_std=noise_std_dyn,
        planet=planet,
        N_orbit=N_orbit,
        b_noise_std=noise_std_mag,
        b_bias=bias,
    )
    # obs, info = env.reset()
    # print(obs)
    # print(info)
    env2 = SpacecraftEnv(
        spacecraft_dynamics, 
        dt=dt, 
        num_steps=200, 
        state_limits=state_limits, 
        control_limits=control_limits,
        dyn_noise_std=0.0,
        planet=planet,
        N_orbit=N_orbit,
        b_noise_std=noise_std_mag,
        b_bias=bias,
    )


    # Test env2 (deterministic) rollout against manual rollout using rk4_step
    print("Testing deterministic rollout...")
    env_trajectory, manual_trajectory = test_trajectory_rollout(env2, num_steps=200)
    # print("Env trajectory:\n", env_trajectory)
    # print("Manual trajectory:\n", manual_trajectory)

    # Test env1 (stochastic) rollout against manual rollout using rk4_step
    print("Testing stochastic rollout...")
    env_trajectory, manual_trajectory = test_trajectory_rollout(env1, num_steps=200)
    # print("Env trajectory:\n", env_trajectory)
    # print("Manual trajectory:\n", manual_trajectory)

    env1.debug_episode()
