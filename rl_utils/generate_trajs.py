"""

Functions to rollout trajectories given controller/policy 
and env, including for DAgger

"""
import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, NamedTuple
import equinox as eqx
from simulation_env_jax import SpacecraftEnvJax
from diffmpc_controller import DiffMPCController, network_output_to_QR



class Trajectory(NamedTuple):
    
    observations: jnp.ndarray
    goal_state: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    Q: jnp.ndarray | None = None
    R: jnp.ndarray | None = None


def _controller_qr(controller, obs):
    """Read fixed costs or evaluate learned costs, in solver torque units."""
    if hasattr(controller, "network"):
        return network_output_to_QR(
            controller.network(obs), controller.nx - 1, controller.nu,
            controller.network.decomposition_type, controller.qr_output_horizon,
        )
    return controller.Q, controller.R




@eqx.filter_jit
def generate_dagger_trajectory(env: SpacecraftEnvJax, 
                        controller: DiffMPCController,
                        expert_policy: Callable, 
                        key: jax.random.PRNGKey, 
                        beta: float,
                        max_ep_steps: int,
                        replan_freq: int = 1,
                        horizon: int = 10,
                        nx: int = 7,
                        nu: int = 3) -> Dict[str, jnp.ndarray]:


    max_torque = env.max_torque

    def scan_step(carry, _):

        env_state, obs, i, nominal_traj, nominal_cntrl, key = carry

        # Get expert action
        expert_action = expert_policy(obs)

        def controller_wrapper(operand):
            return controller(*operand)

        def no_update_nom_traj(operand):

            _, __, nominal_traj, nominal_cntrl = operand

            # Shift nominal trajectory and control by one step
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]

            return action, nominal_traj, nominal_cntrl

        # Get controller action
        controller_action, controller_nominal_traj, controller_nominal_cntrl = jax.lax.cond(
            i%replan_freq == 0,
            controller_wrapper,
            no_update_nom_traj,
            operand=(obs, env_state.goal_state, nominal_traj, nominal_cntrl))

        # Normalize controller action
        controller_action = controller_action / max_torque

        key, subkey = jax.random.split(key)
        use_expert = jax.random.uniform(subkey) < beta

        executed_action = jnp.where(use_expert, expert_action, controller_action)

        shifted_traj  = jnp.concatenate([nominal_traj[1:],  nominal_traj[-1:]],  axis=0)
        shifted_cntrl = jnp.concatenate([nominal_cntrl[1:], nominal_cntrl[-1:]], axis=0)
        new_nominal_traj  = jnp.where(use_expert, shifted_traj,  controller_nominal_traj)
        new_nominal_cntrl = jnp.where(use_expert, shifted_cntrl, controller_nominal_cntrl)

        # Run executed action on the environment using the step key inside the env_state
        new_env_state, new_obs, reward, done, info = env.step(env_state, executed_action)

        new_carry = (new_env_state, new_obs, i+1, new_nominal_traj, new_nominal_cntrl, key)

        outputs = (obs, expert_action, nominal_traj, nominal_cntrl)

        return new_carry, outputs
            


    # Initialize stuff

    # Env
    key, subkey = jax.random.split(key)
    # seed = jax.random.randint(subkey, shape=(), minval=0, maxval=2**32 - 1)
    # init_env_state, init_obs, info = env.reset(seed)
    init_env_state, init_obs, info = env.reset(subkey)

    # Generate initial nominal trajectories
    key, subkey = jax.random.split(key)
    nominal_traj = jnp.tile(init_env_state.state, (horizon + 1, 1))
    nominal_cntrl = 1e-8 * jax.random.normal(subkey, shape=(horizon, nu), dtype=jnp.float64)

    init_carry = (init_env_state, init_obs, 0, nominal_traj, nominal_cntrl, key)

    # Do the for loop
    final_carry, outputs = jax.lax.scan(scan_step, init_carry, xs=None, length=max_ep_steps)

    final_env_state, final_obs, final_i, final_nominal_traj, final_nominal_cntrl, final_key = final_carry
    observations, expert_actions, nominal_trajs, nominal_cntrls = outputs

    observations = jnp.concatenate([observations, final_obs[None, :]], axis=0)
    final_expert_action = expert_policy(final_obs)
    expert_actions = jnp.concatenate([expert_actions, final_expert_action[None, :]], axis=0)

    nominal_trajs = jnp.concatenate([nominal_trajs, final_nominal_traj[None, :]], axis=0)
    nominal_cntrls = jnp.concatenate([nominal_cntrls, final_nominal_cntrl[None, :]], axis=0)

    trajectory = (observations, init_env_state.goal_state, expert_actions, nominal_trajs, nominal_cntrls)

    return trajectory, final_key


@eqx.filter_jit
def generate_n_dagger_trajectories(env: SpacecraftEnvJax,
                            controller: DiffMPCController,
                            expert_policy: Callable,
                            key: jax.random.PRNGKey,
                            beta: float,
                            max_ep_steps: int,
                            n_trajectories: int,
                            replan_freq: int = 1):

    key, subkey = jax.random.split(key)
    batched_keys = jax.random.split(subkey, n_trajectories)
    trajectories, _ = jax.vmap(generate_dagger_trajectory, in_axes=(None, None, None, 0, None, None, None))(env, 
                                                                                                     controller, 
                                                                                                     expert_policy, 
                                                                                                     batched_keys, 
                                                                                                     beta, 
                                                                                                     max_ep_steps, 
                                                                                                     replan_freq)
    
    return trajectories, key


def rollout_controller(env: SpacecraftEnvJax,
                       controller: DiffMPCController,
                       key: jax.random.PRNGKey,
                       horizon: int,
                       max_ep_steps: int,
                       replan_freq: int = 1,
                       nx: int = 7,
                       nu: int = 3,
                       record_qr: bool = False):


    """
    
    Function specifically for MPC type controller(s)
    of form controller(obs, goal_state, nominal_traj, nominal_cntrl) -> action, new_nominal_traj, new_nominal_cntrl
    
    """


    max_torque = env.max_torque

    def scan_step(carry, _):

        env_state, obs, i, nominal_traj, nominal_cntrl, key = carry


        def controller_wrapper(operand):
            return controller(*operand)

        def no_update(operand):

            _, __, nominal_traj, nominal_cntrl = operand

            # Shift nominal trajectory and control by one step
            nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
            nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
            action = nominal_cntrl[0]

            return action, nominal_traj, nominal_cntrl

        # Get controller action
        action, new_nominal_traj, new_nominal_cntrl = jax.lax.cond(
            i%replan_freq == 0,
            controller_wrapper,
            no_update,
            operand=(obs, env_state.goal_state, nominal_traj, nominal_cntrl))

        # Normalize controller action
        action = action / max_torque

        # Extract Q, R from controller for debugging
        if record_qr:
            Q, R = _controller_qr(controller, obs)


        # Run executed action on the environment using the step key inside the env_state
        new_env_state, new_obs, reward, done, info = env.step(env_state, action)

        new_carry = (new_env_state, new_obs, i+1, new_nominal_traj, new_nominal_cntrl, key)

        # outputs = (obs, action*max_torque, Q, R)

        outputs = {
            "obs": obs,
            "action": action*max_torque,
            "rewards": reward,
            "done": done,
        }

        if record_qr:
            outputs["Q"] = Q
            outputs["R"] = R


        # outputs = (obs, action*max_torque)

        return new_carry, outputs
            


    # Env
    key, subkey = jax.random.split(key)
    # seed = jax.random.randint(subkey, shape=(), minval=0, maxval=2**32 - 1)
    # init_env_state, init_obs, info = env.reset(seed)
    init_env_state, init_obs, info = env.reset(subkey)

    # Generate initial nominal trajectories
    key, subkey = jax.random.split(key)
    nominal_traj = jnp.tile(init_env_state.state, (horizon + 1, 1))
    nominal_cntrl = 1e-8 * jax.random.normal(subkey, shape=(horizon, nu), dtype=jnp.float64)
    init_carry = (init_env_state, init_obs, 0, nominal_traj, nominal_cntrl, key)

    # Do the for loop
    final_carry, outputs = jax.lax.scan(scan_step, init_carry, xs=None, length=max_ep_steps)

    final_env_state, final_obs, final_i, final_nominal_traj, final_nominal_cntrl, final_key = final_carry
    observations = outputs["obs"]
    actions = outputs["action"]
    rewards = outputs["rewards"]
    if record_qr:
        Q_list = outputs["Q"]
        R_list = outputs["R"]        
        final_Q, final_R = _controller_qr(controller, final_obs)
    
        Q_list = jnp.concatenate([Q_list, final_Q[None, :]], axis=0)
        R_list = jnp.concatenate([R_list, final_R[None, :]], axis=0)



    # rewards = jnp.concatenate([rewards, final_rewar[None, :]], axis=0)
    # final_action = final_nominal_cntrl[0]
    # observations, actions = outputs

    observations = jnp.concatenate([observations, final_obs[None, :]], axis=0)
    # actions = jnp.concatenate([actions, final_action[None, :]], axis=0)
    
    

    # nominal_trajs = jnp.concatenate([nominal_trajs, final_nominal_traj[None, :]], axis=0)
    # nominal_cntrls = jnp.concatenate([nominal_cntrls, final_nominal_cntrl[None, :]], axis=0)

    # trajectory = (observations, init_env_state.goal_state, actions, Q_list, R_list)
    # trajectory = (observations, init_env_state.goal_state, actions, Q_list, R_list)


    trajectory = Trajectory(
        observations=observations,
        goal_state=init_env_state.goal_state,
        actions=actions,
        rewards=rewards,
        Q=Q_list if record_qr else None,
        R=R_list if record_qr else None
    )


    return trajectory, final_key

@eqx.filter_jit
def n_rollouts_controller(env: SpacecraftEnvJax,
                      controller: DiffMPCController,
                      key: jax.random.PRNGKey,
                      horizon: int,
                      max_ep_steps: int,
                      n_rollouts: int,
                      replan_freq: int = 1,
                      record_qr: bool = False):

    key, subkey = jax.random.split(key)
    batched_keys = jax.random.split(subkey, n_rollouts)
    trajectories, _ = jax.vmap(
        lambda rollout_key: rollout_controller(
            env, controller, rollout_key,
            horizon=horizon,
            max_ep_steps=max_ep_steps,
            replan_freq=replan_freq,
            record_qr=record_qr,
        )
    )(batched_keys)

    return trajectories, key


def rollout_policy(env: SpacecraftEnvJax, 
                   policy: Callable, 
                   key: jax.random.PRNGKey, 
                   max_ep_steps: int,
                   ):


    """
    
    Function to rollout a policy of form policy(obs) -> action

    To be used for standard RL policies like PPO
    
    """

    # max_torque = env.max_torque
    
    def scan_step(carry, _):

        env_state, obs, i, key = carry

        action = policy(obs)
        key, subkey = jax.random.split(key)

        # Run executed action on the environment using the step key inside the env_state
        new_env_state, new_obs, reward, done, info = env.step(env_state, action)

        new_carry = (new_env_state, new_obs, i+1, subkey)

        # outputs = (obs, action, reward, done, info)
        outputs = {
            "obs": obs,
            "action": action,
            "rewards": reward,
            "done": done,
        }

        return new_carry, outputs
            


    # Initialize stuff

    # Env
    key, subkey = jax.random.split(key)
    init_env_state, init_obs, info = env.reset(subkey)

    init_carry = (init_env_state, init_obs, 0, key)

    # Do the for loop
    final_carry, outputs = jax.lax.scan(scan_step, init_carry, xs=None, length=max_ep_steps)

    final_env_state, final_obs, final_i, final_key = final_carry
    # observations, actions, rewards, dones, infos = outputs

    observations = outputs["obs"]
    actions = outputs["action"]
    rewards = outputs["rewards"]
    
    observations = jnp.concatenate([observations, final_obs[None, :]], axis=0)

    # trajectory = (observations, init_env_state.goal_state, actions)

    trajectory = Trajectory(
        observations=observations,
        goal_state=init_env_state.goal_state,
        actions=actions,
        rewards=rewards,
        Q=None,
        R=None
    )

    return trajectory, final_key


@eqx.filter_jit
def n_rollouts_policy(env: SpacecraftEnvJax,
                      policy: Callable,
                      key: jax.random.PRNGKey,
                      max_ep_steps: int,
                      n_rollouts: int):

    key, subkey = jax.random.split(key)
    batched_keys = jax.random.split(subkey, n_rollouts)
    trajectories, _ = jax.vmap(rollout_policy, in_axes=(None, None, 0, None))(env, 
                                                                             policy, 
                                                                             batched_keys, 
                                                                             max_ep_steps)
    return trajectories, key







