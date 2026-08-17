import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

import gymnasium as gym
from simulation_env_simpler import SpacecraftEnv

from typing import Callable, Tuple, List, Dict
from quaternion_functions import q_left, q_conj, get_rotation, q_to_mrp, skew, quaternion_projection, quaternion_jacobian
from functools import partial

MAX_TORQUE = 5e-5

@partial(jax.jit,static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def network_output_to_QR(theta, nx, nu, decomposition_type='diagonal', 
                horizon=10, qmin=1e-1, rmin=1e-1, qmax=1e5, rmax=1e5, 
                eps=1e-3):
    """
    Convert the output of the neural network (theta) into Q and R matrices for MPC.
    theta: output of the neural network, shape (nx + nu,)
    nx: number of state variables
    nu: number of control inputs
    Returns:
        Q: state cost matrix, shape (nx, nx)
        R: control cost matrix, shape (nu, nu)
    """

    # Need to consider how to handle case where const matrix is used over the horizon


    # Split theta into Q and R part
    if decomposition_type == 'diagonal':

        # Figure out better way to do this
        # Maybe check how acmpc guys did it
        n_q = nx * horizon
        n_r = nu * horizon

        Q_flat = theta[:n_q]
        R_flat = theta[n_q:n_q + n_r]

        Q_flat = qmin + 0.5 * (qmax - qmin) * (Q_flat + 1)
        R_flat = rmin + 0.5 * (rmax - rmin) * (R_flat + 1)

        # Create sequence of Q and R matrices for each timestep in the horizon

        if horizon == 1:

            # Const matrices over time
            Q = jnp.diag(Q_flat)
            R = jnp.diag(R_flat)

        else:

            Q_diag = Q_flat.reshape(horizon, nx)
            R_diag = R_flat.reshape(horizon, nu)

            Q = jax.vmap(jnp.diag)(Q_diag)
            R = jax.vmap(jnp.diag)(R_diag)

    elif decomposition_type == 'full':
        # Decomposition is of type Q = AA^T, R = BB^T

        n_q = horizon * nx * nx
        n_r = horizon * nu * nu

        A = theta[:n_q]
        B = theta[n_q:n_q + n_r]

        if horizon == 1:
            Q = A @ A.T
            R = B @ B.T

        else:

            A  = theta[:n_q].reshape((horizon, nx, nx))
            B  = theta[n_q:n_q + n_r].reshape((horizon, nu, nu))

            # No bounding for full decomposition (yet)

            Q = jax.vmap(lambda A: A @ A.T)(A)
            R = jax.vmap(lambda B: B @ B.T)(B)

    elif decomposition_type == 'cholesky':

        # Decomposition is of type Q = LL^T, R = MM^T
        n_q = (nx * (nx + 1) // 2) * horizon
        n_r = (nu * (nu + 1) // 2) * horizon

        L_flat = theta[:n_q].reshape(horizon,-1)
        M_flat = theta[n_q:n_q + n_r].reshape(horizon,-1)

        # No bounding for cholesky decomposition (yet)

        row_indices_Q, col_indices_Q = jnp.tril_indices(nx)
        row_indices_R, col_indices_R = jnp.tril_indices(nu)

        if horizon == 1:

            L = jnp.zeros((nx,nx))
            M = jnp.zeros((nu,nu))

            L = L.at[row_indices_Q, col_indices_Q].set(L_flat)
            M = M.at[row_indices_R, col_indices_R].set(M_flat)

            Q = L @ L.T
            R = M @ M.T

        else:

            L = jnp.zeros((horizon, nx, nx))
            M = jnp.zeros((horizon, nu, nu))

            L = L.at[:, row_indices_Q, col_indices_Q].set(L_flat)
            M = M.at[:, row_indices_R, col_indices_R].set(M_flat)

            Q = jax.vmap(lambda L: L @ L.T)(L)
            R = jax.vmap(lambda M: M @ M.T)(M)

    R = R / (MAX_TORQUE ** 2)  # Scale R by max torque squared

    return Q, R



def sample_initial_states(batch_size, key, state_specs):

    """
    
    Shamelessly copied from Patrick Schwartz
    
    """


    """
    Generates a batch of random initial states based on provided specs.

    Args:
        batch_size: Number of states to sample.
        key: JAX key.
        state_specs: List of dicts, e.g., 
            [{'name': 'pos', 'shape': (3,), 'dist': 'uniform', 'min': -1, 'max': 1},
             {'name': 'rot', 'shape': (4,), 'dist': 'quaternion'}]
    """
    states = []
    
    for spec in state_specs:
        key, subkey = jax.random.split(key)
        dist_type = spec.get('dist', 'uniform')
        shape = (batch_size,) + spec.get('shape', (1,))

        if dist_type == 'uniform':
            val = jax.random.uniform(subkey, shape=shape, 
                                 minval=spec['min'], maxval=spec['max'],
                                 dtype=jnp.float64)
        
        elif dist_type == 'normal':
            val = spec.get('mean', 0.0) + spec.get('std', 1.0) * jax.random.normal(subkey, shape=shape, dtype=jnp.float64)

        elif dist_type == 'quaternion':
            # Specialized sampler for unit quaternions
            q = jax.random.normal(subkey, shape=shape, dtype=jnp.float64)
            val = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

        elif dist_type == 'constant':
            return jnp.broadcast_to(jnp.array(spec['value'], jnp.float64), shape)

        states.append(val.reshape(batch_size, -1))
        #TODO: Potentially set trajectory up as pytrees (traj.pos instead of traj[:,0:3])
    return jnp.concatenate(states, axis=-1, dtype=jnp.float64)







