from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import struct

# Load configuration from config.py


@struct.dataclass
class ReplayBuffer:
    goal_state: jnp.ndarray
    obs: jnp.ndarray
    expert_actions: jnp.ndarray
    nominal_traj: jnp.ndarray
    nominal_cntrl: jnp.ndarray
    ptr: jnp.ndarray    
    size: jnp.ndarray   
    max_size: int




def init_buffer(max_size, obs_dim, state_dim, action_dim, horizon) -> ReplayBuffer:
    return ReplayBuffer(
        goal_state=jnp.zeros((max_size, state_dim)),
        obs=jnp.zeros((max_size, obs_dim)),
        expert_actions=jnp.zeros((max_size, action_dim)),
        nominal_traj=jnp.zeros((max_size, horizon + 1, state_dim)),
        nominal_cntrl=jnp.zeros((max_size, horizon, action_dim)),
        ptr=jnp.array(0),
        size=jnp.array(0),
        max_size=max_size
    )


@jax.jit
def add_transitions_to_buffer(buffer: ReplayBuffer, 
                              obs: jnp.ndarray, 
                              goal_state: jnp.ndarray,
                              expert_actions: jnp.ndarray,
                              nominal_traj: jnp.ndarray,
                              nominal_cntrl: jnp.ndarray
                              ) -> ReplayBuffer:

    # All inputs MUST be batched, i.e., (batch_size, feature_dim)

    batch_size = obs.shape[0]
    ptr = buffer.ptr

    # Handle wrap around
    indices = (ptr + jnp.arange(batch_size)) % buffer.max_size

    new_ptr = (ptr + batch_size) % buffer.max_size
    new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)

    # replace the data at the calculated indics
    new_buffer = buffer.replace(
        obs=buffer.obs.at[indices].set(obs),
        goal_state=buffer.goal_state.at[indices].set(goal_state),
        expert_actions=buffer.expert_actions.at[indices].set(expert_actions),
        nominal_traj=buffer.nominal_traj.at[indices].set(nominal_traj),
        nominal_cntrl=buffer.nominal_cntrl.at[indices].set(nominal_cntrl),
        ptr=new_ptr,
        size=new_size
    )
    return new_buffer


def add_trajectories_to_buffer(buffer: ReplayBuffer, 
                               trajectories: List[Dict[str, jnp.ndarray]],
                               ) -> ReplayBuffer:

    obs = jnp.concatenate([traj["obs"] for traj in trajectories], axis=0)
    expert_actions = jnp.concatenate([traj["expert_actions"] for traj in trajectories], axis=0)
    nominal_traj = jnp.concatenate([traj["nominal_traj"] for traj in trajectories], axis=0)
    nominal_cntrl = jnp.concatenate([traj["nominal_cntrl"] for traj in trajectories], axis=0)

    goal_state = []
    for traj in trajectories:
        goal = traj["goal_state"]
        if goal.ndim == 1:

            T = traj["obs"].shape[0]
            goal = jnp.broadcast_to(goal, (T, goal.shape[0]))
        goal_state.append(goal)
    goal_state = jnp.concatenate(goal_state, axis=0)


    return add_transitions_to_buffer(buffer, obs, goal_state, expert_actions, nominal_traj, nominal_cntrl)





# @jax.jit
# def add_trajectories_to_buffer(buffer: ReplayBuffer, 
#                                trajectories: List[Dict[str, jnp.ndarray]],
#                                ) -> ReplayBuffer:


#     obs = [jnp.concatenate([traj["obs"] for traj in trajectories], axis=0)]
#     action = [jnp.concatenate([traj["expert_actions"] for traj in trajectories], axis=0)]
#     next_obs = [jnp.concatenate([traj["next_obs"] for traj in trajectories], axis=0)]

#     # Handle batched inputs 
#     obs = jnp.atleast_2d(obs)
#     action = jnp.atleast_2d(action)
#     next_obs = jnp.atleast_2d(next_obs)

#     batch_size = obs.shape[0]
#     ptr = buffer.ptr

#     # Handle wrap around
#     indices = (ptr + jnp.arange(batch_size)) % buffer.max_size

#     new_ptr = (ptr + batch_size) % buffer.max_size
#     new_size = jnp.minimum(buffer.size + batch_size, buffer.max_size)

#     # replace the data at the calculated indics
#     new_buffer = buffer.replace(
#         obs=buffer.obs.at[indices].set(obs),
#         actions=buffer.expert_actions.at[indices].set(action),
#         next_obs=buffer.next_obs.at[indices].set(next_obs),
#         ptr=new_ptr,
#         size=new_size
#     )
#     return new_buffer

# @jax.jit
def sample_from_buffer(buffer: ReplayBuffer, key: jax.random.PRNGKey, batch_size: int):


    # Sample random indices
    indices = jax.random.randint(key, (batch_size,), 0, buffer.size)

    # Gather the samples
    batch = {
        'obs': buffer.obs[indices],
        'goal_state': buffer.goal_state[indices],
        'expert_actions': buffer.expert_actions[indices],
        'nominal_traj': buffer.nominal_traj[indices],
        'nominal_cntrl': buffer.nominal_cntrl[indices],
    }
    return batch

def can_sample_buffer(buffer: ReplayBuffer, batch_size: int) -> bool:
    return buffer.size >= batch_size

def get_buffer_size(buffer: ReplayBuffer) -> int:
    return int(buffer.size)



