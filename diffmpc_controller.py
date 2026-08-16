import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple
import equinox as eqx

from moreau.jax import Solver

from diff_mpc_functions import build_mpc_solver, get_A_data, get_P_csr_data
from quaternion_functions import q_mul, q_to_mrp, mrp_to_q
from mj_utils import network_output_to_QR
from propagate_functions import linearize_and_discretize_dynamics, dynamics_params, generate_nominal_trajectory


import time











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
    output_horizon: int = eqx.field(static=True)


    layers: list
    activation: callable = eqx.field(static=True)
    output_activation: callable = eqx.field(static=True)



    def __init__(self, 
                 nx, nu, key, 
                 layers, activation='relu', output_activation='tanh',
                 output_horizon=10, eps=1e-3, decomposition_type='diagonal'):
        self.nx = nx
        self.nu = nu
        self.eps = eps
        self.output_horizon = output_horizon # Determinees to horizon Q and R matrices vary
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
        output_dim *= output_horizon

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






class DiffMPCController(eqx.Module):


    network: FeedForwardNetwork

    output_horizon: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    nx: int = eqx.field(static=True)
    nu: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    state_limits: jnp.ndarray = eqx.field(static=True)
    control_limits: jnp.ndarray = eqx.field(static=True)

    solver: Solver = eqx.field(static=True)
    solver_params: dict = eqx.field(static=True)




    def __init__(self, network: FeedForwardNetwork, horizon, dt,
                 state_limits, control_limits):

        self.network = network
        self.nx = network.nx
        self.nu = network.nu
        self.horizon = horizon
        self.output_horizon = network.output_horizon
        self.dt = dt
        self.state_limits = state_limits
        self.control_limits = control_limits


        # Build solver once
        self.solver, self.solver_params = build_mpc_solver(self.horizon, self.nx - 1, self.nu)


    def form_ocp_moreau(self, x0, xg, nom_traj, nom_control, Q_seq, R_seq):

        # Returns matrices that can be used by the moreau optimal control solver 

        # Get sequence of A, B matrices using linearization
        A, B = linearize_and_discretize_dynamics(nom_traj, nom_control, self.dt)

        # Get A data
        A_data = get_A_data(A, B, self.solver_params)

        b_eq = jnp.concatenate([
            jnp.zeros(self.horizon * (self.nx - 1)),  # dynamics
            x0,            # initial condition
        ])

        x_min = self.state_limits[:,0]
        x_max = self.state_limits[:,1]
        u_min = self.control_limits[:,0] - nom_control
        u_max = self.control_limits[:,1] - nom_control

        u_bounds = jnp.stack([-u_min, u_max], axis=1).flatten()

        b_ineq = jnp.concatenate([
            jnp.tile(-x_min, self.horizon + 1),  # -x <= -x_min
            jnp.tile(x_max, self.horizon + 1),   # x <= x_max
            u_bounds,
        ])

        b = jnp.concatenate([b_eq, b_ineq])

        P_data, P_dense = get_P_csr_data(Q_seq, R_seq, self.solver_params)

        cntrl_goal = jnp.zeros((self.horizon, self.nu))
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


        has_nan_input = (jnp.isnan(obs).any() | jnp.isnan(x_goal).any() | 
                     jnp.isnan(x_nominal).any() | jnp.isnan(u_nominal).any())

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
        Q, R = network_output_to_QR(theta, self.nx - 1, self.nu, self.network.decomposition_type, self.output_horizon)
        # Make prints for debugging 
        jax.debug.print("Q : {}", Q)
        jax.debug.print("R : {}", R)

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
        x_nominal, _ = generate_nominal_trajectory(x0, self.horizon, u_nominal, self.dt)

        dx0 = self.get_error_coordinates(x0,x_nominal[0])
        dxgoal = jax.vmap(self.get_error_coordinates,in_axes=(None, 0))(x_goal, x_nominal)

        # jax.debug.print("nom_traj quat norms: {}", jnp.linalg.norm(x_nominal[:, :4], axis=1))

        # Form Optimal Control Problem
        P_data, A_data, q, b = self.form_ocp_moreau(dx0, dxgoal, x_nominal, u_nominal, Q, R)

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

        



        reshaped = solution.x[self.nx - 1:].reshape(self.horizon, self.nx - 1 + self.nu)
        
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




