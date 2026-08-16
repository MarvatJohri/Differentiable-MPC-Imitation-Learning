"""

File that contains an MPC solver written in style of HW2 trajdesign class

Help taken from https://github.com/JHU-ACEL/uranus-mpc-private/blob/main/controllers/pdip_controller.py

"""

from typing import Callable, Iterable, Tuple, Optional, Dict, Any


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


import equinox as eqx

import moreau
from moreau.jax import Solver
from jax.experimental import sparse as jsparse




# Consider:
# Solver itself only actually needs to track sparsity patter
# Can generate P_data and A_data separately at solve time
# can consider:
# Write a solver class that inputs dimensions
# Outputs a solver
# has a method to solve MPC given OCP data (A, B, Q, R, x0, xg)




def get_P_csr_idx(N: int, nx: int, nu: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    # 2. Construct P_col_indices and P_row_offsets
    # We will determine the structure of a single Q, R, and Qf block first
    
    # Local column patterns for individual blocks
    Q_col_local = jnp.tile(jnp.arange(nx), nx)
    R_col_local = jnp.tile(jnp.arange(nu), nu)
    Qf_col_local = jnp.tile(jnp.arange(nx), nx)
    
    # Local row count tracking per individual block
    Q_nnz_per_row = jnp.full(nx, nx)
    R_nnz_per_row = jnp.full(nu, nu)
    Qf_nnz_per_row = jnp.full(nx, nx)

    def scan_body(current_idx, t):
        # Calculate shifts for this specific horizon step
        q_cols = Q_col_local + current_idx
        current_idx += nx
        
        r_cols = R_col_local + current_idx
        current_idx += nu
        
        # Package the iteration's outputs
        # We concatenate Q and R for this specific step to preserve interleaving
        step_cols = jnp.concatenate([q_cols, r_cols])
        step_nnz  = jnp.concatenate([Q_nnz_per_row, R_nnz_per_row])
        
        return current_idx, (step_cols, step_nnz)

    # --- 2. Run the loop over N horizons ---
    # scan will loop N times, carrying 'current_idx' and stacking the outputs
    init_idx = 0
    dummy_xs = jnp.arange(N)
    final_idx, (stacked_cols, stacked_nnz) = jax.lax.scan(scan_body, init_idx, dummy_xs)
    
    # --- 3. Process the final Qf block ---
    qf_cols = Qf_col_local + final_idx
    
    # --- 4. Flatten the looped results and append the final Qf block ---
    # stacked_cols has shape (N, len(Q) + len(R)). .ravel() flattens it strictly in order.
    P_col_indices = jnp.concatenate([stacked_cols.ravel(), qf_cols])
    total_nnz_per_row = jnp.concatenate([stacked_nnz.ravel(), Qf_nnz_per_row])
    
    # P_row_offsets is the cumulative sum of non-zeros per row, starting at 0
    P_row_offsets = jnp.zeros(len(total_nnz_per_row) + 1)#, dtype=jnp.int32)
    P_row_offsets = P_row_offsets.at[1:].set(jnp.cumsum(total_nnz_per_row))

    return P_row_offsets.astype(jnp.int32), P_col_indices.astype(jnp.int32)


def get_A_csr_idx(N: int, nx: int, nu: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates the static CSR row offsets and column indices for the constraint matrix.
    Depends ONLY on the structural dimensions N, nx, and nu.
    """
    # ===========================
    # 1. A_dynamics Columns
    # ===========================
    # Each row contains: nx (for A) + nu (for B) + 1 (for -I) elements
    t_idx = jnp.arange(N)[:, None, None]
    x_t_start = t_idx * (nx + nu)
    u_t_start = x_t_start + nx
    x_tp1_start = jnp.where(t_idx < N - 1, (t_idx + 1) * (nx + nu), N * (nx + nu))

    range_nx = jnp.arange(nx)[None, None, :]
    range_nu = jnp.arange(nu)[None, None, :]

    col_A = jnp.tile(x_t_start + range_nx, (1, nx, 1))  # (N, nx, nx)
    col_B = jnp.tile(u_t_start + range_nu, (1, nx, 1))  # (N, nx, nu)
    
    row_i = jnp.arange(nx)[None, :, None]
    col_I = x_tp1_start + row_i                        # (N, nx, 1)
    
    dyn_col_indices = jnp.concatenate([col_A, col_B, col_I], axis=2).ravel()

    # ===========================
    # 2. A_init Columns
    # ===========================
    init_col_indices = jnp.arange(nx)

    # ===========================
    # 3. A_state_bounds Columns
    # ===========================
    t_sb = jnp.arange(N + 1)[:, None]
    x_idx_sb = jnp.where(t_sb < N, t_sb * (nx + nu), N * (nx + nu))
    state_cols = x_idx_sb + jnp.arange(nx)[None, :]   # (N+1, nx)
    sb_col_indices = jnp.concatenate([state_cols, state_cols], axis=1).ravel()

    # ===========================
    # 4. A_input_bounds Columns
    # ===========================
    t_ib = jnp.arange(N)[:, None]
    u_idx_ib = t_ib * (nx + nu) + nx
    input_cols = u_idx_ib + jnp.arange(nu)[None, :]   # (N, nu)
    ib_col_indices = jnp.concatenate([input_cols, input_cols], axis=1).ravel()

    # ===========================
    # Combine Columns & Compute Row Offsets
    # ===========================
    A_col_indices = jnp.concatenate([
        dyn_col_indices, init_col_indices, sb_col_indices, ib_col_indices
    ])
    
    # Elements per row for each section
    dyn_nnz = jnp.full(N * nx, nx + nu + 1, dtype=jnp.int32)
    init_nnz = jnp.ones(nx, dtype=jnp.int32)
    sb_nnz = jnp.ones(2 * (N + 1) * nx, dtype=jnp.int32)
    ib_nnz = jnp.ones(2 * N * nu, dtype=jnp.int32)

    total_nnz_per_row = jnp.concatenate([dyn_nnz, init_nnz, sb_nnz, ib_nnz])
    
    A_row_offsets = jnp.zeros(len(total_nnz_per_row) + 1, dtype=jnp.int32)
    A_row_offsets = A_row_offsets.at[1:].set(jnp.cumsum(total_nnz_per_row))

    return A_row_offsets.astype(jnp.int32), A_col_indices.astype(jnp.int32)



def build_mpc_solver(N: int, nx: int, nu: int) -> Tuple[Solver, Dict[str, Any]]:
    """
    Builds an MPC solver with the given horizon and dimensions.
    Returns a Moreau solver instance with precomputed sparsity patterns.
    """
    # Get CSR indices for P and A
    P_indptr, P_indices = get_P_csr_idx(N, nx, nu)
    A_indptr, A_indices = get_A_csr_idx(N, nx, nu)

    n_eq = N * nx + nx          # From A_eq
    n_ineq = 2 * (N + 1) * nx + 2 * N * nu  # From G_ineq
    cones = moreau.Cones(
        num_zero_cones=n_eq, 
        num_nonneg_cones=n_ineq
    )
    solver = Solver(
        n=(N + 1) * nx + N * nu, 
        m=n_eq + n_ineq, 
        P_row_offsets=P_indptr,
        P_col_indices=P_indices,
        A_row_offsets=A_indptr,
        A_col_indices=A_indices,
        cones=cones,
    )

    params = {
        "N": N,
        "nx": nx,
        "nu": nu,
        "P_row_offsets": P_indptr,
        "P_col_indices": P_indices,
        "A_row_offsets": A_indptr,
        "A_col_indices": A_indices,
        # "b_ineq": b_ineq
    }


    return solver, params




def get_P_csr_data(Q: jnp.ndarray, 
                   R: jnp.ndarray, 
                #    Qf: jnp.ndarray,
                   params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:

    N = params["N"]
    nx = params["nx"]
    nu = params["nu"]

    P_row_offsets = params["P_row_offsets"]
    P_col_indices = params["P_col_indices"]

    # Make sure Q_seq and R_seq are of shape (N, nx, nx) and (N, nu, nu) respectively
    if Q.ndim == 2:
        if Q.shape != (nx, nx):
            raise ValueError(f"Q must be of shape ({nx}, {nx}) or ({N}, {nx}, {nx}) or ({N+1}, {nx}, {nx}) but got {Q.shape}")
    
    elif Q.ndim == 3:
        if Q.shape != (N, nx, nx) and Q.shape != (N + 1, nx, nx):
            raise ValueError(f"Q must be of shape ({nx}, {nx}) or ({N}, {nx}, {nx}) or ({N+1}, {nx}, {nx}) but got {Q.shape}")

    if R.ndim == 2:
        if R.shape != (nu, nu):
            raise ValueError(f"R must be of shape ({nu}, {nu}) or ({N}, {nu}, {nu}) but got {R.shape}")
    elif R.ndim == 3:
        if R.shape != (N, nu, nu):
            raise ValueError(f"R must be of shape ({nu}, {nu}) or ({N}, {nu}, {nu}) but got {R.shape}")

    if Q.ndim == 2:
        Q_flat = Q.ravel()
        if R.ndim != 2:
            raise ValueError(f"If Q is 2D, R must also be 2D, but got R.ndim={R.ndim}")
        R_flat = R.ravel()
        Qf_flat = Q.ravel()
        QR_flat_sequence = jnp.concatenate([Q_flat, R_flat])
        P_data = jnp.concatenate([
            jnp.tile(QR_flat_sequence, N), 
            Qf_flat
        ])

    else:
        # Reshape Q and R to (N, nx*nx) and (N, nu*nu) respectively
        if Q.shape[0] == N + 1:
            Qf = Q[-1]
            Q = Q[:-1]
        else:
            Qf = Q[-1]
        Q_flat = Q.reshape(N, nx * nx)
        if R.ndim == 3:
            R_flat = R.reshape(N, nu * nu)
        else:
            R_flat = jnp.tile(R[None, :, :], (N, 1, 1)).reshape(N, nu * nu)
        Qf_flat = Qf.ravel()

        # Need to construct P_data in the order of the sparsity pattern
        # For each time step, we have Q and R blocks, then finally Qf
        # i.e., P_data = [Q_0, R_0, Q_1, R_1, ..., Q_{N-1}, R_{N-1}, Qf]

        P_data = jnp.zeros((N * (nx * nx + nu * nu) + nx * nx,), dtype=Q.dtype)

        def body(i, P_data):
            # P_data is the carry here
            start_idx = i * (nx * nx + nu * nu)
            P_data = P_data.at[start_idx:start_idx + nx * nx].set(Q_flat[i])
            P_data = P_data.at[start_idx + nx * nx:start_idx + nx * nx + nu * nu].set(R_flat[i])
            return P_data

        P_data = jax.lax.fori_loop(0, N, body, P_data)

        # Add Qf
        P_data = P_data.at[N * (nx * nx + nu * nu):].set(Qf_flat)

    # Q_flat = Q.ravel()
    # R_flat = R.ravel()
    # Qf_flat = Q.ravel()
    
    # # Repeat the [Q, R] blocks N times, then append Qf
    # # We use jnp.tile to repeat the sequences efficiently in JAX
    # QR_flat_sequence = jnp.concatenate([Q_flat, R_flat])
    # P_data = jnp.concatenate([
    #     jnp.tile(QR_flat_sequence, N), 
    #     Qf_flat
    # ])
    
    # Get dense version of P
    P_sparse = jsparse.BCSR((P_data, P_col_indices.astype(jnp.int32), P_row_offsets.astype(jnp.int32)), 
                            shape=((N+1)*nx + N*nu, (N+1)*nx + N*nu))
    P_dense = P_sparse.todense()

    return P_data, P_dense



def  get_A_data(A: jnp.ndarray, B: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
    """
    Extracts the numerical data vector for matrix A. 
    Maintains structural zeros for compatibility with fixed sparsity structures.
    """
    N = params["N"]
    nx = params["nx"]
    nu = params["nu"]

    # if A.ndim == 2:
    #     A = jnp.tile(A[None, :, :], (N, 1, 1))  # (N, nx, nx)
    # if B.ndim == 2:
    #     B = jnp.tile(B[None, :, :], (N, 1, 1))  # (N, nx, nu)

    if A.ndim == 2:
        A = jnp.broadcast_to(A, (N,) + A.shape)
    if B.ndim == 2:
        B = jnp.broadcast_to(B, (N,) + B.shape)


    
    # 1. Dynamics Data: Link A, B, and a column of -1.0 per row
    I_neg = jnp.full((N, nx, 1), -1.0)
    dyn_data = jnp.concatenate([A, B, I_neg], axis=2).ravel()

    # 2. Initialization Data (Identity diagonal)
    init_data = jnp.ones(nx)

    # 3. State Bounds Data (Alternates -1.0 rows and 1.0 rows per step)
    sb_data_step = jnp.concatenate([-jnp.ones(nx), jnp.ones(nx)])
    sb_data = jnp.tile(sb_data_step, N + 1)

    # 4. Input Bounds Data (Alternates -1.0 rows and 1.0 rows per step)
    ib_data_step = jnp.concatenate([-jnp.ones(nu), jnp.ones(nu)])
    ib_data = jnp.tile(ib_data_step, N)

    # Combine everything into a single flat vector matching the sparsity structure
    A_data = jnp.concatenate([dyn_data, init_data, sb_data, ib_data])
    return A_data

def get_q_vector(p_x: jnp.ndarray, p_u: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
    """
    p: linear cost vector of shape (nx + nu,)
    Returns q of shape ((N+1)*nx + N*nu,)
    """

    N = params["N"]

    q = jnp.concatenate([
        jnp.tile(jnp.concatenate([p_x, p_u]), N),  # N times [p_x, p_u]
        p_x  # Final state cost
    ])


    return q


# def solve_mpc(self, x0, xg, A, B, Q, R, p_x, p_u):
def solve_mpc(solver: Solver, x0: jnp.ndarray, xg: jnp.ndarray, 
              A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, 
              params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    

    N = params["N"]
    nx = params["nx"]
    nu = params["nu"]


    b_eq = jnp.concatenate([
        jnp.zeros(N * nx),  # dynamics
        x0,            # initial condition
    ])

    b = jnp.concatenate([b_eq, params["b_ineq"]])

    # # Extract R from Q
    # R = Q[nx:, nx:]
    # # Reduce Q to only state cost
    # Q = Q[:nx, :nx]

    # p_x = p[:nx]
    # p_u = p[nx:]


    P_data, P_dense = get_P_csr_data(Q, R, params)

    A_data = get_A_data(A, B, params)


    '''
    J = (x-xref)'Q(x-xref) + u'Ru
    => J = x'Qx - 2*xref'Qx + xref'Qxref + u'Ru
    => min x'Qx + u'Ru - 2*xref'Qx + xref'Qxref == min x'Qx + u'Ru + p'x
    where p = -2*Q*xref
    
    '''
    cntrl_goal = jnp.zeros((N, nu))
    blocks = jnp.hstack((cntrl_goal, xg[1:] ))
    full_vector = jnp.concatenate((xg[0], blocks.ravel()))
    q_vec = -P_dense @ full_vector


    # block = jnp.hstack((jnp.zeros(nu), xg))
    # full_vector = jnp.hstack((xg, jnp.tile(block, N))).T
    # q_vec = -P_dense @ full_vector

    # q_vec = get_q_vector(p_x, p_u, params)


    solution = solver.solve(P_data, A_data, q_vec, b)

    info = solver.info

    # Extract middle blocks (x_0 to x_{N-1} and u_0 to u_{N-1})
    reshaped = solution.x[:N*(nx + nu)].reshape(N, nx + nu)
    state_traj_middle = reshaped[:, :nx]      # x_0 to x_{N-1}
    cntrl_traj = reshaped[:, nx:]             # u_0 to u_{N-1}

    # Extract final state x_N
    x_N = solution.x[N*(nx + nu):]

    # Full state trajectory
    state_traj = jnp.concatenate([state_traj_middle, x_N[None, :]], axis=0)

    u0 = cntrl_traj[0]

    return u0, state_traj, cntrl_traj, info.status






