import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from typing import Sequence, NamedTuple, Any
import sys
# from wrappers import (
#     LogWrapper,
#     BraxGymnaxWrapper,
#     VecEnv,
#     NormalizeVecObservation,
#     NormalizeVecReward,
#     ClipAction,
# )
import os

# Add parent directory to sys.path to import configs and mj_utils
from pathlib import Path

HERE = Path(__file__).parent
MJ_WORK = HERE.parent
sys.path.append(str(MJ_WORK))


import equinox as eqx
from simulation_env_jax import SpacecraftEnvJax, generate_n_trajectories
from configs import ExpConfig, SpacecraftEnvConfig, PPOHyperparameters
from mj_utils import make_dummy_controller

from time import time


from ppo_model import ActorCritic, save_network, load_network

 
 
def gaussian_log_prob(mean, log_std, action):
    return jnp.sum(
        -0.5 * ((action - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )
 
 
def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

    

def sample_actions(model: ActorCritic, obs, key):
    # Assume batched observations
    # of size obs.shape = (batch_size, obs_dim)
    mean = jax.vmap(model.mean)(obs)
    value = jax.vmap(model.value)(obs)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    action = mean + jnp.exp(model.log_std) * noise
    log_prob = gaussian_log_prob(mean, model.log_std, action)
    return action, log_prob, value




class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # info: jnp.ndarray


def lr_scheduler(start_lr: float, schedule_type: str, end_lr: float, 
                 total_grad_steps: int, resume_grad_steps: int):
    # Use Optax's built-in schedulers for learning rate scheduling
    if schedule_type == "linear":
        return optax.linear_schedule(
            init_value=start_lr,
            end_value=end_lr,
            transition_steps=total_grad_steps - resume_grad_steps
        )
    elif schedule_type == "exponential":
        return optax.exponential_decay(
            init_value=start_lr,
            transition_steps=total_grad_steps - resume_grad_steps,
            decay_rate=end_lr / start_lr,
            staircase=False
        )
    elif schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=start_lr,
            decay_steps=total_grad_steps - resume_grad_steps,
            alpha=end_lr / start_lr
        )
    elif schedule_type == "constant":
        return optax.constant_schedule(start_lr)
    else:
        raise ValueError(f"Unsupported learning rate schedule type: {schedule_type}")


# class RunningStat(NamedTuple):
#     mean: jnp.ndarray
#     var: jnp.ndarray
#     count: jnp.ndarray
 
 
# def init_running_stat() -> RunningStat:
#     return RunningStat(jnp.zeros(()), jnp.ones(()), jnp.asarray(1e-4))  
 
 
# def update_running_stat(s: RunningStat, x: jnp.ndarray) -> RunningStat:
#     """Parallel (Chan et al.) update of mean/var with a batch x."""
#     b_mean, b_var, b_count = x.mean(), x.var(), x.size
#     delta = b_mean - s.mean
#     tot = s.count + b_count
#     new_mean = s.mean + delta * b_count / tot
#     m2 = s.var * s.count + b_var * b_count + delta ** 2 * s.count * b_count / tot
#     return RunningStat(new_mean, m2 / tot, tot)

class RunningMeanStd(NamedTuple):
    """Running mean/std for observations (per-dimension)."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_obs_rms(obs_dim: int) -> RunningMeanStd:
    return RunningMeanStd(
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
        count=jnp.array(1e-4)
    )


def init_ret_rms() -> RunningMeanStd:
    return RunningMeanStd(
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=jnp.array(1e-4)
    )


def update_rms(rms: RunningMeanStd, batch: jnp.ndarray) -> RunningMeanStd:
    """Update running mean/std with a batch of data. Works for any shape."""
    batch = batch.reshape(-1, *rms.mean.shape) if rms.mean.ndim > 0 else batch.flatten()
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    batch_count = batch.shape[0]
    
    delta = batch_mean - rms.mean
    tot_count = rms.count + batch_count
    
    new_mean = rms.mean + delta * batch_count / tot_count
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * rms.count * batch_count / tot_count
    new_var = M2 / tot_count
    
    return RunningMeanStd(new_mean, new_var, tot_count)


def normalize_obs(obs: jnp.ndarray, rms: RunningMeanStd, clip: float = 10.0) -> jnp.ndarray:
    """Normalize observations using running statistics."""
    return jnp.clip((obs - rms.mean) / jnp.sqrt(rms.var + 1e-8), -clip, clip)


def save_train_state(opt_state: optax.OptState, 
                     obs_rms: RunningMeanStd,
                     ret_rms: RunningMeanStd, 
                     key: jax.random.PRNGKey, 
                     iteration: int, 
                     train_state_file: str):

    model_state_checkpoint = {
            "opt_state": opt_state,
            "obs_stat": obs_rms,
            "ret_rms": ret_rms,
            "key": key,
            "iteration": iteration,
        }
    eqx.tree_serialise_leaves(train_state_file, model_state_checkpoint)


def save_model(model: ActorCritic, 
               opt_state: optax.OptState, 
               obs_rms: RunningMeanStd,
               ret_rms:RunningMeanStd, 
               key: jax.random.PRNGKey, 
               iteration: int, 
               checkpoint_path: str,
               final:bool = False):

    # Save everything
    if final:
        checkpoint_file = os.path.join(checkpoint_path, f"final_model.eqx")
        train_state_file = os.path.join(checkpoint_path, f"final_train_state.eqx")
    else:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(checkpoint_path, f"model_{iteration}_steps.eqx")
        train_state_file = os.path.join(checkpoint_path, f"train_state_{iteration}_steps.eqx")

    os.makedirs(checkpoint_path, exist_ok=True)
    save_network(model, checkpoint_file)
    save_train_state(opt_state, obs_rms, ret_rms, key, iteration, train_state_file)



def load_train_state(opt_state: optax.OptState, 
                     obs_rms: RunningMeanStd,
                     ret_rms:RunningMeanStd, 
                     train_state_file: str):
    
    if not os.path.exists(train_state_file):
        raise FileNotFoundError(f"Train state file {train_state_file} does not exist.")
    
    train_state_checkpoint = {
        "opt_state": opt_state,
        "obs_rms": obs_rms,
        "ret_rms": ret_rms,
        "key": jax.random.PRNGKey(0),  # Placeholder, will be replaced
        "iteration": 0,  # Placeholder, will be replaced
    }

    train_state = eqx.tree_deserialise_leaves(train_state_file, train_state_checkpoint)
    opt_state = train_state["opt_state"]
    obs_rms = train_state["obs_rms"]
    ret_rms = train_state["ret_rms"]
    key = train_state["key"]
    iteration = train_state["iteration"]

    print(f"Train state loaded from {train_state_file}")

    return opt_state, obs_rms, ret_rms, key, iteration




def load_model(model: ActorCritic, 
               opt_state: optax.OptState, 
               obs_rms: RunningMeanStd,
               ret_rms: RunningMeanStd, 
               checkpoint_path: str, 
               iteration: int, 
               final: bool = False):
    # Load model parameters and optimizer state using Equinox's deserialization
    
    if final:
        checkpoint_file = os.path.join(checkpoint_path, f"final_model.eqx")
        train_state_file = os.path.join(checkpoint_path, f"final_train_state.eqx")
    else:
        checkpoint_file = os.path.join(checkpoint_path, f"model_{iteration}_steps.eqx")
        train_state_file = os.path.join(checkpoint_path, f"train_state_{iteration}_steps.eqx")

    model = load_network(model, checkpoint_file)
    opt_state, obs_rms, ret_rms, key, iteration = load_train_state(opt_state, obs_rms, ret_rms, train_state_file)

    return model, opt_state, obs_rms, ret_rms, key, iteration



def make_train(config, env: SpacecraftEnvJax):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    # env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = VecEnv(env)
    
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_updates = config["NUM_UPDATES"]
    batch_size = num_envs * num_steps
    minibatch_size = config["MINIBATCH_SIZE"]
    assert batch_size % minibatch_size == 0, "Batch size must be divisible by minibatch size."
    num_minibatches = batch_size // minibatch_size



    # Get learning rate scheduler

    # Adjust total timesteps and initial timesteps
    # To account for gradient updates
    # Optax schedulers expect total update steps, 
    # not total timesteps

    total_grad_steps = config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * num_minibatches

    if config["RESUME_TRAINING"]:

        updates_alr_done = config["RESUME_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
        resume_grad_steps = updates_alr_done * config["UPDATE_EPOCHS"] * num_minibatches
    else:
        resume_grad_steps = 0


    lr = lr_scheduler(
        start_lr=config["LEARNING_RATE"],
        schedule_type=config["LEARNING_RATE_SCHEDULE"],
        end_lr=config["LEARNING_RATE_FINAL"],
        total_grad_steps=total_grad_steps,
        resume_grad_steps=resume_grad_steps
      )

    obs_dim = config["OBS_DIM"]
    act_dim = config["ACT_DIM"]

    # Vectorized reset and step functions for the environment
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step_autoreset)
    v_failed = jax.vmap(env._failed)
    v_get_obs = jax.vmap(env._get_obs)


    def calculate_gae(traj: Transition, last_val):
        def get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            not_done = 1.0 - done.astype(value.dtype)
            delta = reward + config["GAMMA"] * next_value * not_done - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * not_done * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            get_advantages, (jnp.zeros_like(last_val), last_val), traj, reverse=True,
            unroll=16
        )
        return advantages, advantages + traj.value

    def loss_fn(model: ActorCritic, batch: Transition, gae, targets):
        mean = jax.vmap(model.mean)(batch.obs)
        value = jax.vmap(model.value)(batch.obs)
        log_prob = gaussian_log_prob(mean, model.log_std, batch.action)

        # Value loss
        value_losses = jnp.square(value - targets)
        if config["CLIP_VALUE_LOSS"]:
            # v_clipped = batch.value + jnp.clip(value - batch.value, -config["VF_CLIP_EPS"], config["VF_CLIP_EPS"])
            # value_loss = 0.5 * jnp.maximum((value - targets) ** 2, (v_clipped - targets) ** 2).mean()

            value_pred_clipped = batch.value + jnp.clip(value - batch.value, -config["VF_CLIP_EPS"], config["VF_CLIP_EPS"])
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * value_losses.mean()

        # value_pred_clipped = batch.value + jnp.clip(value - batch.value, -config["CLIP_EPS"], config["CLIP_EPS"])
        # value_loss = 0.5 * jnp.maximum((value - targets) ** 2, (value_pred_clipped - targets) ** 2).mean()

        # value_losses = jnp.square(value - targets)
        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
        # value_loss = (
        #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        # )

        # Clipped surrogate objective
        log_ratio = log_prob - batch.log_prob
        ratio = jnp.exp(log_ratio)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        # jax.debug.print("log_ratio stats: min={} max={} mean={} | ratio stats: min={} max={} mean={}", 
        #             log_ratio.min(), log_ratio.max(), log_ratio.mean(),
        #             ratio.min(), ratio.max(), ratio.mean())

        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()

        entropy = gaussian_entropy(model.log_std)
        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        return total_loss, (value_loss, loss_actor, entropy, approx_kl)

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    

    def train(key):
        # INIT NETWORK
        # network = ActorCritic(
        #     env.action_space(env_params).shape[0], activation=conf    ig["ACTIVATION"]
        # )

        # Initialize actor critic network
        key, subkey = jax.random.split(key)
        model = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            layers=config["NET_ARCH"],
            log_std_init=config["LOG_STD_INIT"],
            key=subkey,
        )

        # Add this debug check before training starts (in the train function, after model init):
        filtered = eqx.filter(model, eqx.is_inexact_array)
        print("log_std in filtered model:", hasattr(filtered, 'log_std'), type(getattr(filtered, 'log_std', None)))


        params = eqx.filter(model, eqx.is_inexact_array)
        # Label as actor/critic
        labels = jax.tree_util.tree_map(lambda _: "critic", params)
        labels = eqx.tree_at(
            lambda p: (p.actor, p.log_std),
            labels,
            replace=(
                jax.tree_util.tree_map(lambda _: "actor", params.actor),
                "actor",
            ),
        )

        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #     optax.adam(learning_rate=lr, eps=1e-5),
        # )
        optimizer = optax.multi_transform(
            {
                "actor": optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["ACTOR_LEARNING_RATE"], eps=1e-5),
                ),
                "critic": optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["CRITIC_LEARNING_RATE"], eps=1e-5),
                ),
            },
            labels,   
        )
        opt_state = optimizer.init(params)


        # INIT ENV
        # rng, _rng = jax.random.split(rng)
        # reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # obsv, env_state = env.reset(reset_rng, env_params)

        key, subkey = jax.random.split(key)
        reset_keys = jax.random.split(subkey, config["NUM_ENVS"])
        env_state, obs, _ = v_reset(reset_keys)
        ep_return = jnp.zeros((config["NUM_ENVS"],),dtype=obs.dtype)
        disc_return = jnp.zeros((config["NUM_ENVS"],),dtype=obs.dtype)
        # running_stat = init_running_stat()

        obs_rms = init_obs_rms(obs_dim)
        ret_rms = init_ret_rms()

        # TRAIN LOOP
        def _update_step(carry_top, update_idx):
            # Collect stuff to be used in later jax.lax.scan calls
            # Rename carry top to smthng else later
            model, opt_state, env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key = carry_top



            # COLLECT TRAJECTORIES
            def _env_step(carry, unused):
                # train_state, env_state, last_obs, rng = runner_state
                # env_states, obs, ep_return, discounted_return, running_stat, key = carry
                env_states, obs, ep_return, discounted_return, obs_rms, ret_rms, key = carry
                key, subkey = jax.random.split(key)

                # SELECT ACTION
                # rng, _rng = jax.random.split(rng)
                # pi, value = network.apply(train_state.params, last_obs)
                # action = pi.sample(seed=_rng)
                # log_prob = pi.log_prob(action)

                obs_rms = update_rms(obs_rms, obs)

                obs_norm = normalize_obs(obs, obs_rms)
                action, log_prob, value = sample_actions(model, obs_norm, subkey)


                # STEP ENV

                # rng, _rng = jax.random.split(rng)
                # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # obsv, env_states, reward, done, info = env.step(
                #     rng_step, env_states, action, env_params
                # )

                # env manages rng internally so don't need to pass subkey again
                env_states, next_obs, raw_reward, done, info = v_step(env_states, action)
                not_done = 1.0 - done.astype(value.dtype)

                ep_return = ep_return + raw_reward
                finished_return = jnp.where(done, ep_return, 0.0)
                ep_return = ep_return * not_done

                # # Update obs running statistics
                # obs_rms = update_rms(obs_rms, next_obs)

                reward = raw_reward
                if config["NORMALIZE_REWARD"]:
                    discounted_return = discounted_return * config["GAMMA"] * not_done + raw_reward
                    # running_stat = update_running_stat(running_stat, discounted_return)
                    # reward = raw_reward / jnp.sqrt(running_stat.var + 1e-8)

                    ret_rms = update_rms(ret_rms, discounted_return)
                    reward = raw_reward / jnp.sqrt(ret_rms.var + 1e-8)
                    reward = jnp.clip(reward, -10.0, 10.0)
 
                # Time-limit truncation: bootstrap with V(terminal obs).
                # `info` holds the PRE-reset state / goal / step_count from env.step.
                if config["BOOTSTRAP_TIMEOUTS"]:
                    term_state, goal_state, term_steps, _ = info
                    timeout = done & (term_steps >= env.max_ep_steps) & ~v_failed(term_state)
                    term_obs = v_get_obs(term_state, goal_state)
                    term_obs_norm = normalize_obs(term_obs, obs_rms)
                    # term_value = jax.vmap(model.value)(term_obs)
                    term_value = jax.vmap(model.value)(term_obs_norm)
                    reward = reward + config["GAMMA"] * term_value * timeout.astype(reward.dtype)




                # transition = Transition(
                #     done, action, value, reward, log_prob, obs, info
                # )
                transition = Transition(
                    done, action, value, reward, log_prob, obs_norm
                )
                # runner_state = (train_state, env_states, obsv, rng)
                # carry = (env_states, next_obs, ep_return, discounted_return, running_stat, key)
                carry = (env_states, next_obs, ep_return, discounted_return, obs_rms, ret_rms, key)
                return carry, (transition, finished_return)

            # runner_state, traj_batch = jax.lax.scan(
            #     _env_step, runner_state, None, config["NUM_STEPS"]
            # )

            # (env_state, obs, ep_return, disc_return, rstat, key), (traj, finished_return) = jax.lax.scan(
            #     _env_step, (env_state, obs, ep_return, disc_return, rstat, key), None, length=config["NUM_STEPS"]
            # )
            (env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key), (traj, finished_return) = jax.lax.scan(
                _env_step, (env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key), None, length=config["NUM_STEPS"]
            )


            # CALCULATE ADVANTAGE
            # train_state, env_state, last_obs, rng = runner_state
            # _, last_val = network.apply(train_state.params, last_obs)
            obs_norm = normalize_obs(obs, obs_rms)
            # last_val = jax.vmap(model.value)(obs)
            last_val = jax.vmap(model.value)(obs_norm)
            advantages, targets = calculate_gae(traj, last_val)




            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                # Consider carry only containing model, opt state and a key
                model, opt_state, key = update_state

                key, subkey = jax.random.split(key)

                # Generate a permutation of indices
                permutation = jax.random.permutation(subkey, batch_size)


                # Generate minibatches, same type as purejaxRL
                batch = (traj, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch
                )



                def _update_minbatch(carry, output):
                    # traj_batch, advantages, targets = batch_info


                    model, opt_state = carry
                    traj_batch, advantages, targets = output

                    # Get loss and grads

                    old_log_std = model.log_std.copy()

                    (loss, aux), grads = grad_fn(model, traj_batch, advantages, targets)

                    # # Check grads for log_std, it _should_ be getting updated
                    # jax.debug.print("log_std grad norm: {}", jnp.linalg.norm(grads.log_std))

                    # Update using optimizer
                    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))                    

                    # Apply updates
                    model = eqx.apply_updates(model, updates)

                    # DEBUG: log_std AFTER update
                    # jax.debug.print("log_std before: {} | after: {} | diff: {}", 
                    #                 old_log_std[0], model.log_std[0], model.log_std[0] - old_log_std[0])

                    # grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    # total_loss, grads = grad_fn(
                    #     train_state.params, traj_batch, advantages, targets
                    # )
                    # train_state = train_state.apply_gradients(grads=grads)
                    
                    
                    carry = (model, opt_state)
                    output = (loss, aux)
                    
                    return carry, output

                
                # train_state, total_loss = jax.lax.scan(
                #     _update_minbatch, train_state, minibatches
                # )
                # update_state = (train_state, traj_batch, advantages, targets, rng)

                
                # Run the minibatch updates lax scan
                (model, opt_state), (losses, aux) = jax.lax.scan(
                    _update_minbatch, (model, opt_state), minibatches
                )

                # Update state
                update_state = (model, opt_state, key)
                outputs = (losses, aux)
                
                return update_state, outputs
            

            # update_state = (train_state, traj_batch, advantages, targets, rng)
            # update_state, loss_info = jax.lax.scan(
            #     _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            # )




            # train_state = update_state[0]
            # metric = traj_batch.info
            # rng = update_state[-1]


            # Run update epochs lax scan
            update_state = (model, opt_state, key)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, length=config["UPDATE_EPOCHS"]
            )

            # Unpack loss info
            losses, aux = loss_info
            value_loss, pg_loss, entropy, approx_kl = aux

            # Unpack model
            model, opt_state, key = update_state

            total_loss = losses.mean()
            value_loss = value_loss.mean()
            pg_loss = pg_loss.mean()
            entropy = entropy.mean()
            approx_kl = approx_kl.mean()

            n_eps = traj.done.sum()

            metrics = {
                "mean_reward": traj.reward.mean(),   # in normalized units if NORMALIZE_REWARD
                "reward_scale": jnp.sqrt(ret_rms.var), # what raw rewards are divided by
                "n_episodes": n_eps,
                "episode_return": jnp.where(n_eps > 0, finished_return.sum() / jnp.maximum(n_eps, 1), jnp.nan),
                "total_loss": total_loss,
                "value_loss": value_loss,
                "pg_loss": pg_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "action_std": jnp.exp(model.log_std).mean(),

            }



            if config.get("DEBUG"):

                # def callback(info):
                #     return_values = info["returned_episode_returns"][
                #         info["returned_episode"]
                #     ]
                #     timesteps = (
                #         info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                #     )
                #     for t in range(len(timesteps)):
                #         print(
                #             f"global step={timesteps[t]}, episodic return={return_values[t]}"
                #         )

                # jax.debug.callback(callback, metric)

                def print_metrics(idx, m):
                    print(
                        f"update {int(idx):5d} | step {int(idx + 1) * batch_size:>10d} | "
                        f"rew/step {float(m['mean_reward']):8.4f} (scale {float(m['reward_scale']):.3f}) | "
                        f"ep_ret {float(m['episode_return']):9.3f} ({int(m['n_episodes'])} eps) | "
                        f"vf {float(m['value_loss']):9.4f} | pg {float(m['pg_loss']):8.4f} | "
                        f"kl {float(m['approx_kl']):.5f} | std {float(m['action_std']):.3f}"
                    )
 
                jax.lax.cond(
                    update_idx % config["LOG_EVERY_UPDATES"] == 0,
                    lambda: jax.debug.callback(print_metrics, update_idx, metrics),
                    lambda: None,
                )



            # runner_state = (train_state, env_state, last_obs, rng)
            
            
            # Update runner step for update_step iteration

            # runner_state = (model, opt_state, env_state, obs, ep_return, disc_return, rstat, key)
            runner_state = (model, opt_state, env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key)
            
            
            # Return EVERYTHING
            return runner_state, metrics

        

        # rng, _rng = jax.random.split(rng)
        # runner_state = (train_state, env_state, obsv, _rng)
        # runner_state, metric = jax.lax.scan(
        #     _update_step, runner_state, None, config["NUM_UPDATES"]
        # )

        # Run update steps lax scan
        # carry_top = (model, opt_state, env_state, obs, ep_return, disc_return, running_stat, key)
        carry_top = (model, opt_state, env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key)
        carry_top, metric = jax.lax.scan(
            _update_step, carry_top, jnp.arange(config["NUM_UPDATES"]), length=config["NUM_UPDATES"]
        )   



        return {"runner_state": carry_top, "metrics": metric}

    return train


if __name__ == "__main__":

    exp_config = ExpConfig()
    hyperparams = PPOHyperparameters()
    env_config = SpacecraftEnvConfig()

    PPO_BASE_SAVE_PATH = exp_config.ppo_base_save_path
    PPO_BASE_LOG_PATH = exp_config.ppo_base_log_path


    PPO_EXPERIMENT_NAME = exp_config.ppo_experiment_name
    PPO_EXPERIMENT_NOTES = exp_config.ppo_experiment_notes

    NX = exp_config.nx
    NU = exp_config.nu

    RESUME_TRAINING = exp_config.resume_ppo_training
    RESUME_TIMESTEPS = exp_config.ppo_resume_timesteps
    RESUME_MODEL_PATH = exp_config.ppo_resume_model_path


    DYNAMICS_PARAMETERS = env_config.spacecraft_dynamics_parameters
    DT = env_config.dt
    MAX_EP_STEPS = env_config.max_ep_steps
    STATE_LIMITS = jnp.asarray(env_config.state_limits)
    CONTROL_LIMITS = jnp.asarray(env_config.control_limits)
    MAX_TORQUE = env_config.max_torque
    DYN_NOISE_STD = env_config.dyn_noise_std
    THETA_THRESHOLD = env_config.theta_threshold
    OMEGA_THRESHOLD = env_config.omega_threshold
    OMEGA_PENALTY = env_config.omega_penalty
    OMEGA_FAIL_PENALTY = env_config.omega_fail_penalty
    ACTION_PENALTY = env_config.action_penalty
    THETA_THRESHOLD_REWARD = env_config.theta_threshold_reward
    GOAL_REWARD = env_config.goal_reward
    THETA_STABILITY_TOL = env_config.theta_stability_tol
    OMEGA_STABILITY_TOL = env_config.omega_stability_tol

    # Make env
    env = SpacecraftEnvJax(dynamics_params=DYNAMICS_PARAMETERS,
                            dt=DT,
                            max_ep_steps=MAX_EP_STEPS,
                            state_limits=STATE_LIMITS,
                            control_limits=CONTROL_LIMITS,
                            max_torque=MAX_TORQUE,
                            dyn_noise_std=DYN_NOISE_STD,
                            theta_threshold=THETA_THRESHOLD,
                            omega_threshold=OMEGA_THRESHOLD,
                            theta_threshold_reward=THETA_THRESHOLD_REWARD,
                            omega_penalty=OMEGA_PENALTY,
                            omega_fail_penalty=OMEGA_FAIL_PENALTY,
                            action_penalty=ACTION_PENALTY,
                            goal_reward=GOAL_REWARD)


    config = {
        "LEARNING_RATE": hyperparams.learning_rate,
        "ACTOR_LR": hyperparams.actor_learning_rate,
        "CRITIC_LR": hyperparams.critic_learning_rate,
        "NET_ARCH": hyperparams.net_arch,
        "LEARNING_RATE_SCHEDULE": hyperparams.learning_rate_schedule,
        "LEARNING_RATE_FINAL": hyperparams.learning_rate_final,
        "NUM_ENVS": hyperparams.n_envs,
        "NUM_STEPS": hyperparams.n_steps,
        "TOTAL_TIMESTEPS": hyperparams.total_timesteps,
        "RESUME_TRAINING": exp_config.resume_ppo_training,
        "RESUME_TIMESTEPS": exp_config.ppo_resume_timesteps,
        "RESUME_MODEL_PATH": exp_config.ppo_resume_model_path,
        "UPDATE_EPOCHS": hyperparams.n_epochs,
        "NUM_MINIBATCHES": hyperparams.n_minibatches,
        "MINIBATCH_SIZE": hyperparams.minibatch_size,
        "GAMMA": hyperparams.gamma,
        "LOG_STD_INIT": hyperparams.log_std_init,
        "GAE_LAMBDA": hyperparams.gae_lambda,
        "CLIP_EPS": hyperparams.clip_range,
        "ENT_COEF": hyperparams.ent_coef,
        "VF_COEF": hyperparams.vf_coef,
        "CLIP_VALUE_LOSS": hyperparams.clip_val_loss,
        "VF_CLIP_EPS": hyperparams.vf_clip_eps,
        "MAX_GRAD_NORM": hyperparams.max_grad_norm,
        "ACTIVATION": "tanh",
        "DEBUG": True,
        "ACT_DIM": NU,
        "OBS_DIM": NX,
        "NORMALIZE_REWARD": True,
        "BOOTSTRAP_TIMEOUTS": True,
        "LOG_EVERY": hyperparams.log_every,
        "LOG_EVERY_UPDATES": hyperparams.log_every_updates,
    }
    rng = jax.random.PRNGKey(42)
    start = time()
    train_jit = jax.jit(make_train(config, env))
    out = train_jit(rng)
    print(f"Training completed successfully in {time() - start:.2f} seconds.")

    # Get stuff 
    runner_state = out["runner_state"]
    metrics = out["metrics"]

    # model, opt_state, env_state, obs, ep_return, disc_return, running_stat, key = runner_state
    model, opt_state, env_state, obs, ep_return, disc_return, obs_rms, ret_rms, key = runner_state

    # Save model and optimizer state
    SAVE_PATH = os.path.join(PPO_BASE_SAVE_PATH, PPO_EXPERIMENT_NAME)
    save_model(model, opt_state, obs_rms, ret_rms, key, config["TOTAL_TIMESTEPS"], SAVE_PATH, final=True)
    print(f"Model saved at {SAVE_PATH}/final_model.eqx")
    