'''

This file contains functions
exclusively for the purpose of propagating the dynamics of the system

Functions here are reusable in the simulation env (NEED TO REWORK IT FIRST)
and for trajectory generation in MPC controller

'''

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from quaternion_functions import q_left, q_conj, q_mul, skew, quaternion_projection, quaternion_jacobian
from typing import Tuple, Callable
from mj_utils import sample_initial_states
import equinox as eqx
from functools import partial


# from DiffMPC_controller import DiffMPCController




dynamics_params = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
dynamics_params["inertia_inv"] = jnp.linalg.inv(dynamics_params["inertia"])

def get_state_error(x, xg):

    q = x[:4]
    qg = xg[:4]

    q_err = q_mul(q_conj(qg), q)

    # fix sign ambiguity
    q_err = jnp.where(q_err[0] < 0, -q_err, q_err)

    omega = x[4:7]
    omegag = xg[4:7]

    omega_err = omega - omegag

    return jnp.concatenate((q_err, omega_err))


# Code for linearization of dynamics about nominal trajectory
def state_dot(state, control, t, u_noise):

    inertia = dynamics_params["inertia"]
    inertia_inverse = dynamics_params["inertia_inv"]

    q = state[:4]
    w = state[4:7]

    tau = control + u_noise

    q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
    w_dot = inertia_inverse @ (tau - skew(w) @ inertia @ w) # cross product a x b = a_skew_symmetric @ b

    state_dot = jnp.concatenate((q_dot, w_dot))

    return state_dot




@jax.jit
def linearize_and_discretize_dynamics(x_nom_traj: jnp.ndarray, u_nom_traj: jnp.ndarray, dt: float):

    # Here the nominal state trajectory uses q_err NOT mrp
    # Converted to mrp using E
    nx = x_nom_traj.shape[1] - 1 # Shape should be 7 for q_err + omega

    def linearize_and_discretize_single(x, x_next, u):
        u_noise = jnp.zeros(u.shape)
        A = jax.jacfwd(state_dot, argnums=0)(x, u, 0.0, u_noise)
        B = jax.jacfwd(state_dot, argnums=1)(x, u, 0.0, u_noise)

        A = quaternion_jacobian(x_next).T @ A @ quaternion_jacobian(x)
        B = quaternion_jacobian(x_next).T @ B

        A = jnp.eye(nx) + A*dt
        B = B*dt

        return A, B

    Ad, Bd = jax.vmap(linearize_and_discretize_single)(x_nom_traj[:-1], x_nom_traj[1:], u_nom_traj)

    # jax.debug.print("Ad NaN: {}, Bd NaN: {}", jnp.isnan(Ad).any(), jnp.isnan(Bd).any())
    # jax.debug.print("Ad range: [{}, {}]", Ad.min(), Ad.max())
    # jax.debug.print("Bd range: [{}, {}]", Bd.min(), Bd.max())

    return Ad, Bd

# def rk4_step(state, control, t, dt):

#     k1 = state_dot(state, control, t)
#     k2 = state_dot(quaternion_projection(state + 0.5 * dt * k1), control, t + 0.5 * dt)
#     k3 = state_dot(quaternion_projection(state + 0.5 * dt * k2), control, t + 0.5 * dt)
#     k4 = state_dot(quaternion_projection(state + dt * k3), control, t + dt)

#     dx = (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

#     return quaternion_projection(state + dx)



def rk4_step(state, control, t, dt, key, noise_std):
    # Ctrl C Ctrl V from Patrick's code
    def dynamics_wrapper(s,k):
        key, noise_key = jax.random.split(k)
        u_noise = jax.random.normal(noise_key, shape=control.shape) * noise_std
        return state_dot(s, control, t, u_noise)

    # Note: quaternion projection is just identity unless defined in Dynamics class
    k1 = dynamics_wrapper(state,key)

    # has_nan = jnp.isnan(control).any() | jnp.isnan(state).any() | jnp.isnan(k1).any()
    # def print_debug():
    #     jax.debug.print("=== RK4 DEBUG ===")
    #     jax.debug.print("control: {}", control)
    #     jax.debug.print("state: {}", state)
    #     jax.debug.print("k1: {}", k1)
    # jax.lax.cond(has_nan, print_debug, lambda: None)


    k2 = dynamics_wrapper(quaternion_projection(state + 0.5 * dt * k1),key)
    k3 = dynamics_wrapper(quaternion_projection(state + 0.5 * dt * k2),key)
    k4 = dynamics_wrapper(quaternion_projection(state + dt * k3),key)

    dx = (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # dx = dt*dynamics_wrapper(state) # euler

    # key1, key = jax.random.split(key)
    # noise = jax.random.normal(key, shape=state.shape) * noise_std * jnp.sqrt(dt)
    x_next = state + dx #+ noise

    omega = x_next[4:7]
    max_omega = 500.0
    omega_norm = jnp.linalg.norm(omega)
    omega_clamped = jnp.where(omega_norm > max_omega, 
                               omega * max_omega / omega_norm, 
                               omega)

    x_next = x_next.at[4:7].set(omega_clamped)

    return x_next


def sample_episode_context(key):

    init_key, target_key = jax.random.split(key, 2)

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



    return initial_state, target_state



@jax.jit
def generate_nominal_trajectory(initial_state: jax.Array, 
                                num_steps: int,
                                control_sequence: jax.Array,
                                dt: float,
                                ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using  control sequence.

    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    initial_state = jnp.asarray(initial_state, dtype=jnp.float64)
    control_sequence = jnp.asarray(control_sequence, dtype=jnp.float64)

    def scan_step(carry, control_input):
        state, i = carry    
        # Key doesn't matter here since noise = 0
        t_current = i * dt
        next_state = rk4_step(state, control_input, t_current, dt, jax.random.PRNGKey(0), 0.0)  # No noise for nominal trajectory
        return (next_state, i+1), (state, control_input)

    # Run scan
    init_carry = (initial_state, 0)
    (final_state, _), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, control_sequence)

    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])

    # return trajectory, controls
    return trajectory, controls

@eqx.filter_jit
def generate_trajectory(initial_state: jax.Array, 
                        target_state: jax.Array,
                        dt: float,
                        key: jax.random.PRNGKey,
                        num_steps: int,
                        controller: Callable,
                        replan_freq: int = 1,
                        noise_std: float = 1e-6,
                        ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using a controller

    Performs effectively the same purpose as 
    doing an episode rollout in the environment, 
    but without the overhead of the environment class.

    Meant to be used during evaluation, since at that time
    we don't really need to use the environment


    """  

    # If I can figure out how to write mixed policy and expert policy
    # In a way that its compatible with this, that might actually save me 
    # a lot of time in long run

    initial_state = jnp.asarray(initial_state, dtype=jnp.float64)
    target_state = jnp.asarray(target_state, dtype=jnp.float64)


    def scan_step(carry, _):

        state, key, i, nominal_traj, nominal_cntrl = carry
        t_current = i * dt
        key, rk4_key, cntrl_key = jax.random.split(key, 3)

        def controller_wrapper(operand):
            return controller(*operand)
            
        def no_update(operand):
            *_ ,nominal_traj, nominal_cntrl = operand
            # Shift nominal trajectory and nominal cntrl
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]
            return action, nominal_traj, nominal_cntrl

        # Find state error to use as obs for controller
        state_error = get_state_error(state, target_state)
        action, nominal_traj, nominal_cntrl = jax.lax.cond(
            i % replan_freq == 0,
            controller_wrapper,
            no_update,
            operand=(state_error, target_state, nominal_traj, nominal_cntrl)
        )
        #control_input = nominal_cntrl[i % replan_freq]
        control_input = nominal_cntrl[0] # always 0 because no_update fn shifts
        next_state = rk4_step(state, control_input, t_current, dt, rk4_key, noise_std)
        return (next_state, key, i+1, nominal_traj, nominal_cntrl), (state, control_input)


    # Define initial stuff
    key, subkey = jax.random.split(key)

    init_nom_traj = jnp.tile(initial_state,(controller.horizon+1,1))
    init_nom_control = jnp.zeros(shape=(init_nom_traj.shape[0]-1, controller.nu))

    init_carry = (initial_state, key, 0, init_nom_traj, init_nom_control)


    (final_state, final_key, _, nominal_traj, nominal_cntrl), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)

    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])

    return trajectory, controls#, nominal_traj, nominal_cntrl



def generate_n_trajectories(initial_state: jax.Array, 
                        target_state: jax.Array,
                        dt: float,
                        key: jax.random.PRNGKey,
                        num_steps: int,
                        controller: Callable,
                        n_trajectories: int,
                        replan_freq: int = 1,
                        noise_std: float = 1e-6
                        ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using a controller

    Performs effectively the same purpose as 
    doing an episode rollout in the environment, 
    but without the overhead of the environment class.

    Meant to be used during evaluation, since at that time
    we don't really need to use the environment


    """  

    # If I can figure out how to write mixed policy and expert policy
    # In a way that its compatible with this, that might actually save me 
    # a lot of time in long run

    initial_state = jnp.asarray(initial_state, dtype=jnp.float64)
    target_state = jnp.asarray(target_state, dtype=jnp.float64)
    
