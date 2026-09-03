import jax
import jax.numpy as jnp

import numpy as np


from simulation_env_jax import SpacecraftEnvJax
from simulation_env_simpler import SpacecraftEnv

import matplotlib.pyplot as plt


DYNAMICS_PARAMS = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
DYNAMICS_PARAMS["inertia_inv"] = jnp.linalg.inv(DYNAMICS_PARAMS["inertia"])

# Environment params
DT = 0.1                         # Simulation timestep
DYN_NOISE_STD = 1e-6             # Dynamics noise
MAX_EPISODE_LENGTH = 1500         # Max episode length


# State/action limits
STATE_LIMITS = [[-1, 1]] * 4 + [[-2, 2]] * 3  # [quat, omega]
STATE_LIMITS_MRP = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
CONTROL_LIMIT_SCALE = 1        # Scales [-1, 1] control limits
MAX_TORQUE = 5e-5
CONTROL_LIMITS = jnp.array([[-CONTROL_LIMIT_SCALE, CONTROL_LIMIT_SCALE]] * 3, dtype=jnp.float64) # [normalized torque]
CONTROL_LIMITS_TORQUE = jnp.array([[-MAX_TORQUE, MAX_TORQUE]] * 3, dtype=jnp.float64) # [torque]

# Reward shaping
THETA_THRESHOLD = np.deg2rad(15.0)  # Convert to radians
OMEGA_THRESHOLD = np.deg2rad(5.0)                 # Angular velocity tolerance (rad/s)
OMEGA_PENALTY = 0.5               # Penalty weight for omega error
ACTION_PENALTY = 0.1              # Penalty weight for action magnitude
GOAL_REWARD = 50.0              # Bonus for reaching goal
THETA_THRESHOLD_REWARD = 10.0        # Bonus for staying within theta threshold



def compare_envs(seed=42):
    """Compare single-step behavior between Gym and JAX envs."""
    
    # Setup Gym env
    gym_env = SpacecraftEnv(dynamics_params=DYNAMICS_PARAMS)
    gym_obs, _ = gym_env.reset(seed=seed)
    
    # Setup JAX env
    jax_env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMS)
    jax_state, jax_obs, _ = jax_env.reset(seed=seed)
    
    print("Initial states match:", np.allclose(gym_env.state, jax_state.state, atol=1e-14))
    print("Initial goals match:", np.allclose(gym_env.goal_state, jax_state.goal_state, atol=1e-14))
    print("Initial obs match:", np.allclose(gym_obs, jax_obs, atol=1e-14))
    
    # Take same action
    action = np.array([0.5, -0.3, 0.1])
    
    gym_obs2, _, _, _, _ = gym_env.step(action)
    jax_state2, jax_obs2, _ = jax_env.step(jax_state, jnp.array(action))
    
    print("After step - states match:", np.allclose(gym_env.state, jax_state2.state, atol=1e-14))
    print("After step - obs match:", np.allclose(gym_obs2, jax_obs2, atol=1e-14))
    
    print("\nGym state:", gym_env.state)
    print("JAX state:", jax_state2.state)
    print("Diff:", np.abs(gym_env.state - np.array(jax_state2.state)))




def compare_env_rollouts(seed=42, num_steps=1500, plot=True):

    """Compare rollouts between Gym and JAX envs."""
    
    # Setup Gym env
    gym_env = SpacecraftEnv(dynamics_params=DYNAMICS_PARAMS)
    gym_observations, _ = gym_env.reset(seed=seed)
    
    # Setup JAX env
    jax_env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMS)
    jax_state, jax_observations, _ = jax_env.reset(seed=seed)

    # Generate a sequence of random actions from -1,1
    actions = np.random.uniform(-1, 1, size=(num_steps, 3))
    gym_traj = []
    jax_traj = []

    gym_observations = []
    jax_observations = []

    stopped = False
    for i in range(num_steps):
        action = actions[i]
        
        gym_obs, _, _, _, _ = gym_env.step(action)
        jax_state, jax_obs, _ = jax_env.step(jax_state, jnp.array(action))
        
        if not np.allclose(gym_env.state, jax_state.state, atol=1e-6):
            print(f"Step {i}: States do not match!")
            print("Gym state:", gym_env.state)
            print("JAX state:", jax_state.state)
            # print("Diff:", np.abs(gym_env.state - np.array(jax_state.state)))
            stopped = True
            break

        if not np.allclose(gym_observations, jax_observations, atol=1e-6):
            print(f"Step {i}: Observations do not match!")
            print("Gym obs:", gym_obs)
            print("JAX obs:", jax_obs)
            # Print the difference for debugging
            # print("Diff:", np.abs(gym_obs - np.array(jax_obs)))
            stopped = True
            break

        gym_traj.append(gym_env.state)
        jax_traj.append(jax_state.state)

        gym_observations.append(gym_obs)
        jax_observations.append(jax_obs)


    # Convert to arrays
    gym_states = np.asarray(gym_traj)
    jax_states = np.asarray(jax_traj)

    gym_obs_traj = np.asarray(gym_observations)
    jax_obs_traj = np.asarray(jax_observations)

    if not stopped:
        print("Rollouts match for all steps.")


    state_diff = gym_states - jax_states
    abs_state_diff = np.abs(state_diff)

    quat_diff = state_diff[:, :4]
    omega_diff = state_diff[:, 4:7]

    quat_abs_diff = np.abs(quat_diff)
    omega_abs_diff = np.abs(omega_diff)

    # ---------------------------------------------------------
    # Quaternion norms
    # ---------------------------------------------------------

    gym_quat_norm = np.linalg.norm(gym_states[:, :4], axis=1)
    jax_quat_norm = np.linalg.norm(jax_states[:, :4], axis=1)

    # ---------------------------------------------------------
    # Actual attitude error
    #
    # q1 and q2 represent the same attitude up to sign, so
    # use abs(dot(q1, q2)).
    # ---------------------------------------------------------

    gym_quat = gym_states[:, :4]
    jax_quat = jax_states[:, :4]

    # Normalize just for calculating attitude error
    gym_quat_normalized = gym_quat / np.linalg.norm(
        gym_quat, axis=1, keepdims=True
    )

    jax_quat_normalized = jax_quat / np.linalg.norm(
        jax_quat, axis=1, keepdims=True
    )

    quat_dot = np.sum(
        gym_quat_normalized * jax_quat_normalized,
        axis=1
    )

    # Numerical safety
    quat_dot = np.clip(np.abs(quat_dot), 0.0, 1.0)

    attitude_error_rad = 2.0 * np.arccos(quat_dot)
    attitude_error_deg = np.rad2deg(attitude_error_rad)

    # ---------------------------------------------------------
    # Maximum errors
    # ---------------------------------------------------------

    quat_error_norm = np.linalg.norm(quat_diff, axis=1)
    omega_error_norm = np.linalg.norm(omega_diff, axis=1)
    state_error_norm = np.linalg.norm(state_diff, axis=1)

    print("\nMaximum errors:")
    print(f"Quaternion component error: {np.max(quat_abs_diff):.3e}")
    print(f"Quaternion vector error:    {np.max(quat_error_norm):.3e}")
    print(f"Attitude error:              {np.max(attitude_error_deg):.3e} deg")
    print(f"Angular velocity error:      {np.max(omega_abs_diff):.3e}")
    print(f"Angular velocity norm error: {np.max(omega_error_norm):.3e}")
    print(f"Total state error:           {np.max(state_error_norm):.3e}")

    # Find where quaternion error first exceeds thresholds
    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    print("\nQuaternion error thresholds:")

    for threshold in thresholds:
        indices = np.where(quat_error_norm > threshold)[0]

        if len(indices) > 0:
            print(
                f"  > {threshold:.0e}: "
                f"step {indices[0]}"
            )
        else:
            print(
                f"  > {threshold:.0e}: "
                f"never"
            )

    if not plot:
        return {
            "gym_states": gym_states,
            "jax_states": jax_states,
            "state_diff": state_diff,
            "attitude_error_deg": attitude_error_deg,
        }

    steps = np.arange(1, num_steps + 1)

    # =========================================================
    # Plot 1: Quaternion component errors
    # =========================================================

    plt.figure(figsize=(10, 6))

    for i, label in enumerate(["qx", "qy", "qz", "qw"]):
        plt.plot(
            steps,
            quat_abs_diff[:, i],
            label=label
        )

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Absolute error")
    plt.title("Quaternion Component Error")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("quaternion_component_error.png", dpi=300)

    # =========================================================
    # Plot 2: Angular velocity component errors
    # =========================================================

    plt.figure(figsize=(10, 6))

    for i, label in enumerate(["ωx", "ωy", "ωz"]):
        plt.plot(
            steps,
            omega_abs_diff[:, i],
            label=label
        )

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Absolute error")
    plt.title("Angular Velocity Component Error")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("angular_velocity_component_error.png", dpi=300)

    # =========================================================
    # Plot 3: Quaternion norm
    # =========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        gym_quat_norm,
        label="Gym"
    )

    plt.plot(
        steps,
        jax_quat_norm,
        label="JAX"
    )

    plt.axhline(1.0, linestyle="--")

    plt.xlabel("Step")
    plt.ylabel("||q||")
    plt.title("Quaternion Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save plot
    plt.savefig("quaternion_norm.png", dpi=300)

    # =========================================================
    # Plot 4: Actual attitude difference
    # =========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        attitude_error_deg
    )

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Attitude error (degrees)")
    plt.title("Actual Attitude Difference")
    plt.grid(True, which="both")
    plt.tight_layout()

    plt.savefig("attitude_difference.png", dpi=300)

    # =========================================================
    # Plot 5: Overall error norms
    # =========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        quat_error_norm,
        label="Quaternion"
    )

    plt.savefig("quaternion_error_norm.png", dpi=300)

    plt.plot(
        steps,
        omega_error_norm,
        label="Angular velocity"
    )

    plt.savefig("angular_velocity_error_norm.png", dpi=300)

    plt.plot(
        steps,
        state_error_norm,
        label="Total state"
    )

    plt.savefig("total_state_error_norm.png", dpi=300)

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("L2 error")
    plt.title("State Divergence")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("state_divergence.png", dpi=300)
    plt.show()

    return {
        "gym_states": gym_states,
        "jax_states": jax_states,
        "gym_obs": gym_obs_traj,
        "jax_obs": jax_obs_traj,
        "state_diff": state_diff,
        "quat_error": quat_abs_diff,
        "omega_error": omega_abs_diff,
        "attitude_error_deg": attitude_error_deg,
        "quat_norm_gym": gym_quat_norm,
        "quat_norm_jax": jax_quat_norm,
    }






if __name__ == "__main__":
    compare_env_rollouts(plot=False)