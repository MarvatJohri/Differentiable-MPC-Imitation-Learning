import jax
import jax.numpy as jnp

# Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
def skew(u):
    return jnp.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

# Left quaternion product (used in the quaternion dynamics)
def q_left(q):
  qs = q[0]
  qv = q[1:]
  qL_A = jnp.concatenate((jnp.array([[qs]]), -qv.reshape(-1, 1)), axis=0).T
  qL_B = jnp.concatenate((qv.reshape(-1,1),  jnp.array(qs*jnp.eye(3) + skew(qv))), axis=1)
  return jnp.concatenate((qL_A, qL_B), axis=0)

# Conjugate quaternion (used in the quaternion dynamics)
def q_conj(q):
    return jnp.concatenate((q[:1], -q[1:]))

# Convert the quaternion into a rotation (rotates vector r as:  r' = get_rotation(q) @ r)
# converts vector from body -> inertial frame (use q_conj(q) as input to go from inertial -> body)
def get_rotation(q):
    qw, qx, qy, qz = q
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    norm_sq = qw2 + qx2 + qy2 + qz2 

    return jnp.array(
        [
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
        ]
    ) / norm_sq

def q_to_mrp(q, qbar):
    qbar = qbar / jnp.linalg.norm(qbar)
    q_error = q_left(q_conj(qbar)) @ q

    e_scalar = q_error[0]
    e_vector = q_error[1:]

    # Standard MRP: phi = e_v / (1 + e_s)
    # Shadow MRP:   phi = -e_v / (1 - e_s)

    denom_standard = 1.0 + e_scalar
    denom_shadow = 1.0 - e_scalar

    # Avoid division by zero
    safe_denom_standard = jnp.where(jnp.abs(denom_standard) > 1e-10, denom_standard, 1e-10)
    safe_denom_shadow = jnp.where(jnp.abs(denom_shadow) > 1e-10, denom_shadow, 1e-10)

    phi_standard = e_vector / safe_denom_standard
    phi_shadow = -e_vector / safe_denom_shadow

    # Use standard when e_scalar >= 0 (keeps ||phi|| <= 1)
    phi = jnp.where(e_scalar >= 0.0, phi_standard, phi_shadow)
    return phi


def mrp_to_q(phi, qbar):
    qbar = qbar / jnp.linalg.norm(qbar)

    phi_norm_sq = jnp.sum(phi**2)

    # If shadow (||phi|| > 1), map back to standard form
    is_shadow = phi_norm_sq > 1.0
    phi_mapped = jnp.where(is_shadow, -phi / (phi_norm_sq + 1e-12), phi)
    phi_mapped_norm_sq = jnp.sum(phi_mapped**2)

    # Standard MRP to quaternion formula
    denom = 1.0 + phi_mapped_norm_sq
    e_scalar = (1.0 - phi_mapped_norm_sq) / denom
    e_vector = (2.0 * phi_mapped) / denom

    # No sign flip needed - phi_mapped is always in standard form now
    q_error = jnp.concatenate([jnp.array([e_scalar]), e_vector])

    q = q_left(qbar) @ q_error
    return q


def quaternion_projection(state: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion part of state (works with numpy or jax)."""
    q = state[:4]
    q_normalized = q / jnp.linalg.norm(q)
    return jnp.concatenate([q_normalized, state[4:]])


def quaternion_jacobian(x: jax.Array) -> jax.Array:
    nx = x.shape[0]
    quat_start = 0
    q1, q2, q3, q4 = x[0], x[1], x[2], x[3]
    G = jnp.array([
        [-q2, -q3, -q4],
        [q1, -q4, q3],
        [q4, q1, -q2],
        [-q3, q2, q1]
    ])
    E = jnp.zeros((nx, nx-1))
    # Quaternion error mapping (rows 0:3, cols quat_start:quat_start+4)
    E = E.at[quat_start:quat_start+4, quat_start:quat_start+3].set(G)
    # Angular velocity passthrough
    E = E.at[quat_start+4:quat_start+7, quat_start+3:quat_start+6].set(jnp.eye(3))
    return E


def q_mul(q1, q2):
    # scalar-first quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
