from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import NamedTuple
import equinox as eqx

from moreau.jax import Solver

from diff_mpc_functions import build_mpc_solver, get_A_data, get_P_csr_data
from quaternion_functions import q_mul, q_to_mrp, mrp_to_q, quaternion_projection, q_left, skew, quaternion_jacobian
# from propagate_functions import linearize_and_discretize_dynamics, dynamics_params, generate_nominal_trajectory

from configs import SpacecraftEnvConfig, ControllerConfig

import time

env_config = SpacecraftEnvConfig()
controller_config = ControllerConfig()


DYNAMICS_PARAMS = env_config.spacecraft_dynamics_parameters
MAX_TORQUE = env_config.max_torque
CONTROL_LIMITS = env_config.control_limits
STATE_LIMITS_MRP = env_config.state_limits_mrp
DT = env_config.dt

MPC_HORIZON = controller_config.mpc_horizon
REPLAN_FREQUENCY = controller_config.replan_frequency
DECOMPOSITION_TYPE = controller_config.decomposition_type
QR_OUTPUT_HORIZON = controller_config.qr_output_horizon
NETWORK_EPSILON = controller_config.network_epsilon





def get_activation(activation_name):
    if activation_name == 'relu':
        return jax.nn.relu
    elif activation_name == 'tanh':
        return jax.nn.tanh
    elif activation_name == 'softplus':
        return jax.nn.softplus
    elif activation_name == 'sigmoid':
        return jax.nn.sigmoid
    elif activation_name == 'swish':
        return jax.nn.swish
    elif activation_name == 'gelu':
        return jax.nn.gelu
    elif activation_name == 'leaky_relu':
        return jax.nn.leaky_relu
    elif activation_name == 'elu':
        return jax.nn.elu
    elif activation_name == 'selu':
        return jax.nn.selu
    else:
        raise ValueError("Invalid activation function")

class FeedForwardNetwork(eqx.Module):
    nx: int = eqx.field(static=True)
    nu: int = eqx.field(static=True)
    eps: float = eqx.field(static=True) 
    decomposition_type: str = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    qr_output_horizon: int = eqx.field(static=True)


    layers: list
    activation: callable = eqx.field(static=True)
    output_activation: callable = eqx.field(static=True)



    def __init__(self, 
                 nx, nu, key, 
                 layers, activation='relu', output_activation='tanh',
                 qr_output_horizon=10, eps=1e-3, decomposition_type='diagonal'):
        self.nx = nx
        self.nu = nu
        self.eps = eps
        self.qr_output_horizon = qr_output_horizon # Determinees to horizon Q and R matrices vary
        # output_horizon = 1 => Q,R constant over horizon = N
        # I have no idea if its possible to handle other cases but whatever
        self.decomposition_type = decomposition_type

        # For now input is just x_obs, shape (nx,)
        obs_dim = nx

        # Output dimension
        if self.decomposition_type == 'diagonal':
            # Diagonal Q and R matrices, so output dimension is nx + nu
            output_dim = nx - 1 + nu
        elif self.decomposition_type == 'full':
            # Q and R matrices decomposed to form Q = AA^T
            output_dim = (nx - 1) * (nx - 1) + nu * nu
        elif self.decomposition_type == 'cholesky':
            # Cholesky factorization of Q and R matrices, so output dimension is nx * (nx + 1) / 2 + nu * (nu + 1) / 2
            output_dim = int((nx - 1) * ((nx - 1) + 1) / 2 + nu * (nu + 1) / 2)
        else:
            raise ValueError("Invalid type. Must be 'diagonal', 'full', or 'cholesky'.")

        # Output dimension is multiplied by horizon of MPC output
        output_dim *= qr_output_horizon

        dims = [obs_dim, *layers, output_dim]

        keys = jax.random.split(key, len(dims) - 1)

        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation)

    def __call__(self, x):
        # Forward pass through the neural network to get theta
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Might need to return raw theta if I'm dealing with full
        # or cholesky decomposition
        # Cross check this
        if self.decomposition_type == 'full' or self.decomposition_type == 'cholesky':
            return self.layers[-1](x)

        x = self.output_activation(self.layers[-1](x))
        return x


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





class DiffMPCController(eqx.Module):


    network: FeedForwardNetwork

    qr_output_horizon: int = eqx.field(static=True)
    mpc_horizon: int = eqx.field(static=True)
    nx: int = eqx.field(static=True)
    nu: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    state_limits: Float[Array, "6 2"] = eqx.field(static=True)
    control_limits: Float[Array, "6 2"] = eqx.field(static=True)
    inertia: Float[Array, "3 3"] = eqx.field(static=True)
    inertia_inv: Float[Array, "3 3"] = eqx.field(static=True)
    mass: float = eqx.field(static=True)

    solver: Solver = eqx.field(static=True)
    solver_params: dict = eqx.field(static=True)




    def __init__(self, 
                 network: FeedForwardNetwork, 
                 mpc_horizon, 
                 dt,
                 state_limits, 
                 control_limits,
                 dynamics_params = DYNAMICS_PARAMS):

        self.network = network
        self.nx = network.nx
        self.nu = network.nu
        self.mpc_horizon = mpc_horizon
        self.qr_output_horizon = network.qr_output_horizon
        self.dt = dt
        self.state_limits = state_limits
        self.control_limits = control_limits
        self.inertia = dynamics_params["inertia"]
        self.inertia_inv = dynamics_params["inertia_inv"]
        self.mass = dynamics_params["mass"]


        # Build solver once
        self.solver, self.solver_params = build_mpc_solver(self.mpc_horizon, self.nx - 1, self.nu)

        # Store vals of Q, R for debugging
        self.Q = None
        self.R = None




    def state_dot_nominal(self, true_state: jnp.ndarray, control: jnp.ndarray, u_noise: jnp.ndarray) -> jnp.ndarray:

        # This is what the controller THINKS the dynamics are

        q = true_state[:4]
        w = true_state[4:7]
    
        tau = control + u_noise
    
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = self.inertia_inv @ (tau - skew(w) @ self.inertia @ w) # cross product a x b = a_skew_symmetric @ b
    
        state_dot = jnp.concatenate((q_dot, w_dot))
    
        return state_dot

    def rk4_step_nominal(self, 
                         true_state: jnp.ndarray,
                         control: jnp.ndarray,
                         u_noise: jnp.ndarray,
                         dt: float):

        k1 = self.state_dot_nominal(true_state, control, u_noise)
        k2 = self.state_dot_nominal(quaternion_projection(true_state + 0.5 * dt * k1), control, u_noise)
        k3 = self.state_dot_nominal(quaternion_projection(true_state + 0.5 * dt * k2), control, u_noise)
        k4 = self.state_dot_nominal(quaternion_projection(true_state + dt * k3), control, u_noise)

        dx = (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return quaternion_projection(true_state + dx)


    def generate_nominal_trajectory(self,
                                    initial_state: jnp.ndarray,
                                    num_steps: int,
                                    control_sequence: jnp.ndarray,
                                    dt: float):

        # Generate a nominal trajectory using the nominal dynamics
        # and a sequence of controls
        def scan_step(state, control: jnp.ndarray):
            u_noise = jnp.zeros(control.shape)  # No noise for nominal trajectory
            next_state = self.rk4_step_nominal(state, control, u_noise, dt)
            # Handle wrap around for quaternion normalization
            # if next_state[0] < 0:
            #     next_state = next_state.at[:4].set(-next_state[:4])

            next_state = jax.lax.cond(next_state[0] < 0, lambda x: x.at[:4].set(-x[:4]), lambda x: x, next_state)
            return next_state, state

        final_state, trajectory = jax.lax.scan(scan_step, initial_state, control_sequence)
        # Add final state to trajectory
        trajectory = jnp.vstack((trajectory, final_state[None, :])) # Shape (num_steps + 1, state_dim)

        return trajectory



    def linearize_and_discretize_dynamics(self, x_nom_traj: jnp.ndarray, u_nom_traj: jnp.ndarray, dt: float):

        # Here the nominal state trajectory uses q_err NOT mrp
        # Converted to mrp using E
        nx = x_nom_traj.shape[1] - 1 # Shape should be 7 for q_err + omega

        def linearize_and_discretize_single(x, x_next, u):
            u_noise = jnp.zeros(u.shape)
            A = jax.jacfwd(self.state_dot_nominal, argnums=0)(x, u, u_noise)
            B = jax.jacfwd(self.state_dot_nominal, argnums=1)(x, u, u_noise)

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


    def form_ocp_moreau(self, x0, xg, nom_traj, nom_control, Q_seq, R_seq):

        # Returns matrices that can be used by the moreau optimal control solver 

        # Get sequence of A, B matrices using linearization
        A, B = self.linearize_and_discretize_dynamics(nom_traj, nom_control, self.dt)

        # Get A data
        A_data = get_A_data(A, B, self.solver_params)

        b_eq = jnp.concatenate([
            jnp.zeros(self.mpc_horizon * (self.nx - 1)),  # dynamics
            x0,            # initial condition
        ])

        x_min = self.state_limits[:,0]
        x_max = self.state_limits[:,1]
        u_min = self.control_limits[:,0] - nom_control
        u_max = self.control_limits[:,1] - nom_control

        u_bounds = jnp.stack([-u_min, u_max], axis=1).flatten()

        b_ineq = jnp.concatenate([
            jnp.tile(-x_min, self.mpc_horizon + 1),  # -x <= -x_min
            jnp.tile(x_max, self.mpc_horizon + 1),   # x <= x_max
            u_bounds,
        ])

        b = jnp.concatenate([b_eq, b_ineq])

        P_data, P_dense = get_P_csr_data(Q_seq, R_seq, self.solver_params)

        cntrl_goal = jnp.zeros((self.mpc_horizon, self.nu))
        blocks = jnp.hstack((cntrl_goal, xg[1:] ))
        full_vector = jnp.concatenate((xg[0], blocks.ravel()))
        q = -P_dense @ full_vector


        return P_data, A_data, q, b

    def get_error_coordinates(self, x, xbar):

        dq = q_to_mrp(x[:4], xbar[:4])
        domega = x[4:7] - xbar[4:7]

        return jnp.concatenate((dq, domega))


    def get_true_coordinates(self, dx, xbar):

      
        q = mrp_to_q(dx[:3], xbar[:4])
        omega = dx[3:6] + xbar[4:7]
        return jnp.concatenate((q, omega))


    @eqx.filter_jit
    def __call__(self, 
                 obs, x_goal, x_nominal, u_nominal):


        # has_nan_input = (jnp.isnan(obs).any() | jnp.isnan(x_goal).any() | 
        #              jnp.isnan(x_nominal).any() | jnp.isnan(u_nominal).any())

        # def print_input_debug():
        #     jax.debug.print("=== NaN IN INPUTS ===")
        #     jax.debug.print("obs: {}", obs)
        #     jax.debug.print("x_goal: {}", x_goal)
        #     jax.debug.print("obs has NaN: {}", jnp.isnan(obs).any())
        #     jax.debug.print("x_goal has NaN: {}", jnp.isnan(x_goal).any())
        #     jax.debug.print("x_nominal has NaN: {}", jnp.isnan(x_nominal).any())
        #     jax.debug.print("u_nominal has NaN: {}", jnp.isnan(u_nominal).any())
        
        # jax.lax.cond(has_nan_input, print_input_debug, lambda: None)
        # # === END DEBUG ===

        obs = jnp.asarray(obs, dtype=jnp.float64)

        obs = jnp.asarray(obs, dtype=jnp.float64)
        x_goal = jnp.asarray(x_goal, dtype=jnp.float64)
        x_nominal = jnp.asarray(x_nominal, dtype=jnp.float64)
        u_nominal = jnp.asarray(u_nominal, dtype=jnp.float64)

        theta = self.network(obs)
        self.Q, self.R = network_output_to_QR(theta, self.nx - 1, self.nu, self.network.decomposition_type, self.qr_output_horizon)
        # Make prints for debugging 
        # jax.debug.print("Q : {}", Q)
        # jax.debug.print("R : {}", R)

        # Need to pass in initial error b/w state and nom_traj, call this delta_x
        # Also need to pass "error" b/w goal and nominal trajectory, call this delta_x_g
        # OCP then reduces to form of minimizing "error" b/w delta_x and delta_x_g


        # Alr have x_goal (I need to account for this in the buffer too somehow ffs)
        # Obs is error b/w current state and x_goal
        # Can use it to get current state

        # q_err = qg* x q
        # Thus q = qg x q_err
        # omega_err = omega - omega_goal

        quat = q_mul(x_goal[:4],obs[:4])
        omega = obs[4:7] + x_goal[4:7]
        x0 = jnp.concat([quat, omega]).astype(jnp.float64)
        x0 = x0.at[:4].set(x0[:4] / jnp.linalg.norm(x0[:4]))

        # Generate a nominal trajectory using previous control inputs
        # this is the "true" nominal trajectory, accounts for non-linearity (but not noise), unlike solution from ocp
        x_nominal = self.generate_nominal_trajectory(x0, self.mpc_horizon, u_nominal, self.dt)

        dx0 = self.get_error_coordinates(x0,x_nominal[0])
        dxgoal = jax.vmap(self.get_error_coordinates,in_axes=(None, 0))(x_goal, x_nominal)

        # jax.debug.print("nom_traj quat norms: {}", jnp.linalg.norm(x_nominal[:, :4], axis=1))

        # Form Optimal Control Problem
        P_data, A_data, q, b = self.form_ocp_moreau(dx0, dxgoal, x_nominal, u_nominal, self.Q, self.R)

        # jax.debug.print("A_data NaN: {}, has inf: {}", jnp.isnan(A_data).any(), jnp.isinf(A_data).any())
        # jax.debug.print("P_data range: [{}, {}]", P_data.min(), P_data.max())
        # jax.debug.print("q range: [{}, {}]", q.min(), q.max())
        # jax.debug.print("b range: [{}, {}]", b.min(), b.max())

        # Solve OCP
        solution = self.solver.solve(P_data, A_data, q, b)
        # jax.debug.print("Solver status: {}", solution.status)
        # jax.debug.print("solution.x NaN: {}, range: [{}, {}]", 
        #                 jnp.isnan(solution.x).any(),
        #                 jnp.nanmin(solution.x), 
        #                 jnp.nanmax(solution.x))

        



        reshaped = solution.x[self.nx - 1:].reshape(self.mpc_horizon, self.nx - 1 + self.nu)
        
        #### Reshaping
        du = reshaped[:, :self.nu]
        dx = reshaped[:, self.nu : self.nu + self.nx - 1]
        dx = jnp.concatenate([dx0[None,:], dx], axis=0)

        state_traj = jax.vmap(self.get_true_coordinates)(dx, x_nominal)
        control_traj = du + u_nominal

        # jax.debug.print("state_traj NaN: {}, control_traj NaN: {}", 
        #         jnp.isnan(state_traj).any(), jnp.isnan(control_traj).any())

        # state_traj = jnp.nan_to_num(state_traj)
        # control_traj = jnp.nan_to_num(control_traj)

        action = control_traj[0]

        
        return action, state_traj, control_traj




