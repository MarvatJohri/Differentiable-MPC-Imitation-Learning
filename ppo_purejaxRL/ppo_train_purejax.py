import jax
import jax.numpy as jnp
import optax
from typing import Sequence, NamedTuple, Any
# from wrappers import (
#     LogWrapper,
#     BraxGymnaxWrapper,
#     VecEnv,
#     NormalizeVecObservation,
#     NormalizeVecReward,
#     ClipAction,
# )


import equinox as eqx
from simulation_env_jax import SpacecraftEnvJax
from configs import ExpConfig, SpacecraftEnvConfig, PPOHyperparameters




def _ortho_linear(in_f: int, out_f: int, key, scale: float) -> eqx.nn.Linear:
    """eqx.nn.Linear with orthogonal weights and zero bias (standard PPO init)."""
    layer = eqx.nn.Linear(in_f, out_f, key=key)
    w = jax.nn.initializers.orthogonal(scale)(key, (out_f, in_f), layer.weight.dtype)
    return eqx.tree_at(lambda l: (l.weight, l.bias), layer, (w, jnp.zeros_like(layer.bias)))
 
 
class MLP(eqx.Module):
    layers: tuple
 
    def __init__(self, in_size, layers, key, out_size, out_scale):
        sizes = [in_size, *layers]
        keys = jax.random.split(key, len(sizes))
        hidden = tuple(
            _ortho_linear(sizes[i], sizes[i + 1], keys[i], jnp.sqrt(2.0))
            for i in range(len(sizes) - 1)
        )
        out = _ortho_linear(sizes[-1], out_size, keys[-1], out_scale)
        self.layers = hidden + (out,)
 
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)
 
 
class ActorCritic(eqx.Module):
    actor: MLP
    critic: MLP
    log_std: jax.Array          
 
    def __init__(self, obs_dim, act_dim, layers, key, init_log_std=-0.5):
        actor_key, critic_key = jax.random.split(key)
        # Std PPO initialization: small init for actor, larger for critic
        self.actor = MLP(obs_dim, layers, actor_key, act_dim, 0.01)   # small init -> ~0 mean actions
        self.critic = MLP(obs_dim, layers, critic_key, 1, 1.0)
        self.log_std = jnp.full((act_dim,), init_log_std)
 
    # These operate on a SINGLE obs; use jax.vmap for batches.
    def mean(self, obs):
        return self.actor(obs)
 
    def value(self, obs):
        return self.critic(obs)[0]
 
 
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
    info: jnp.ndarray


def lr_scheduler(start_lr: float, schedule_type: str, end_lr: float = 1e-5, 
                 total_timesteps: int = 1_500_000, initial_timesteps: int = 0):
    # Use Optax's built-in schedulers for learning rate scheduling
    if schedule_type == "linear":
        return optax.linear_schedule(
            init_value=start_lr,
            end_value=end_lr,
            transition_steps=total_timesteps - initial_timesteps
        )
    elif schedule_type == "exponential":
        return optax.exponential_decay(
            init_value=start_lr,
            transition_steps=total_timesteps - initial_timesteps,
            decay_rate=end_lr / start_lr,
            staircase=False
        )
    elif schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=start_lr,
            decay_steps=total_timesteps - initial_timesteps,
            alpha=end_lr / start_lr
        )
    elif schedule_type == "constant":
        return optax.constant_schedule(start_lr)
    else:
        raise ValueError(f"Unsupported learning rate schedule type: {schedule_type}")


class RunningStat(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray
 
 
def init_running_stat() -> RunningStat:
    return RunningStat(jnp.zeros(()), jnp.ones(()), jnp.asarray(1e-4))  
 
 
def update_running_stat(s: RunningStat, x: jnp.ndarray) -> RunningStat:
    """Parallel (Chan et al.) update of mean/var with a batch x."""
    b_mean, b_var, b_count = x.mean(), x.var(), x.size
    delta = b_mean - s.mean
    tot = s.count + b_count
    new_mean = s.mean + delta * b_count / tot
    m2 = s.var * s.count + b_var * b_count + delta ** 2 * s.count * b_count / tot
    return RunningStat(new_mean, m2 / tot, tot)




def make_train(config, env: SpacecraftEnvJax):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        # env = NormalizeVecObservation(env)
        # env = NormalizeVecReward(env, config["GAMMA"])

        # TODO: Add normalization for jax envs

        pass

    # Get learning rate scheduler
    lr = lr_scheduler(
        start_lr=config["LEARNING_RATE"],
        schedule_type=config["LEARNING_RATE_SCHEDULE"],
        end_lr=config["LEARNING_RATE_FINAL"],
        total_timesteps=config["TOTAL_TIMESTEPS"],
        INITIAL_TIMESTEPS=config["RESUME_TIMESTEPS"] if config["RESUME_TRAINING"] else 0
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
        # if config["CLIP_VALUE_LOSS"]:
        #     v_clipped = batch.value + jnp.clip(value - batch.value, -config["CLIP_EPS"], config["CLIP_EPS"])
        #     value_loss = 0.5 * jnp.maximum((value - targets) ** 2, (v_clipped - targets) ** 2).mean()
        # else:
        #     value_loss = 0.5 * ((value - targets) ** 2).mean()

        value_pred_clipped = batch.value + jnp.clip(value - batch.value, -config["CLIP_EPS"], config["CLIP_EPS"])
        # value_loss = 0.5 * jnp.maximum((value - targets) ** 2, (value_pred_clipped - targets) ** 2).mean()

        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # Clipped surrogate objective
        log_ratio = log_prob - batch.log_prob
        ratio = jnp.exp(log_ratio)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

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

    # grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    

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
            key=subkey,
        )


        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

        # INIT ENV
        # rng, _rng = jax.random.split(rng)
        # reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # obsv, env_state = env.reset(reset_rng, env_params)

        key, subkey = jax.random.split(key)
        reset_keys = jax.random.split(subkey, config["NUM_ENVS"])
        v_obsv, v_env_state, _ = v_reset(reset_keys)
        ep_return = jnp.zeros((config["NUM_ENVS"],),dtype=v_obsv.dtype)
        discounted_return = jnp.zeros((config["NUM_ENVS"],),dtype=v_obsv.dtype)
        running_stat = init_running_stat()

        # TRAIN LOOP
        def _update_step(carry_top, unused):
            # Collect stuff to be used in later jax.lax.scan calls
            # Rename carry top to smthng else later
            model, opt_state, env_state, obs, ep_return, disc_return, rstat, key = carry_top



            # COLLECT TRAJECTORIES
            def _env_step(carry, unused):
                # train_state, env_state, last_obs, rng = runner_state
                env_states, obs, ep_return, discounted_return, running_stat, key = carry
                key, subkey = jax.random.split(key)

                # SELECT ACTION
                # rng, _rng = jax.random.split(rng)
                # pi, value = network.apply(train_state.params, last_obs)
                # action = pi.sample(seed=_rng)
                # log_prob = pi.log_prob(action)
                action, log_prob, value = sample_actions(model, obs, subkey)

                # STEP ENV

                # rng, _rng = jax.random.split(rng)
                # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # obsv, env_states, reward, done, info = env.step(
                #     rng_step, env_states, action, env_params
                # )

                # env manages rng internally so don't need to pass subkey again
                env_state, next_obs, raw_reward, done, info = v_step(env_state, action)
                not_done = 1.0 - done.astype(value.dtype)

                ep_return = ep_return + raw_reward
                finished_return = jnp.where(done, ep_return, 0.0)
                ep_return = ep_return * not_done

                reward = raw_reward
                if config["NORMALIZE_REWARD"]:
                    disc_return = disc_return * config["GAMMA"] * not_done + raw_reward
                    rstat = update_running_stat(rstat, disc_return)
                    reward = raw_reward / jnp.sqrt(rstat.var + 1e-8)
 
                # Time-limit truncation: bootstrap with V(terminal obs).
                # `info` holds the PRE-reset state / goal / step_count from env.step.
                if config["BOOTSTRAP_TIMEOUTS"]:
                    term_state, goal_state, term_steps, _ = info
                    timeout = done & (term_steps >= env.max_ep_steps) & ~v_failed(term_state)
                    term_obs = v_get_obs(term_state, goal_state)
                    term_value = jax.vmap(model.value)(term_obs)
                    reward = reward + config["GAMMA"] * term_value * timeout.astype(reward.dtype)




                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                # runner_state = (train_state, env_states, obsv, rng)
                carry = (env_states, next_obs, ep_return, discounted_return, running_stat, key)
                return carry, (transition, finished_return)

            # runner_state, traj_batch = jax.lax.scan(
            #     _env_step, runner_state, None, config["NUM_STEPS"]
            # )

            (env_state, obs, ep_return, disc_return, rstat, key), (traj, finished_return) = jax.lax.scan(
                _env_step, (env_state, obs, ep_return, disc_return, rstat, key), None, length=config["NUM_STEPS"]
            )


            # CALCULATE ADVANTAGE
            # train_state, env_state, last_obs, rng = runner_state
            # _, last_val = network.apply(train_state.params, last_obs)

            last_val = jax.vmap(model.value)(obs)
            advantages, targets = calculate_gae(traj, last_val)




            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

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
        "NET_ARCH": hyperparams.net_arch,
        "LEARNING_RATE_SCHEDULE": hyperparams.learning_rate_schedule,
        "LEARNING_RATE_FINAL": hyperparams.learning_rate_final,
        "NUM_ENVS": hyperparams.n_rollouts,
        "NUM_STEPS": hyperparams.n_steps,
        "TOTAL_TIMESTEPS": hyperparams.total_timesteps,
        "RESUME_TRAINING": exp_config.resume_ppo_training,
        "RESUME_TIMESTEPS": exp_config.ppo_resume_timesteps,
        "RESUME_MODEL_PATH": exp_config.ppo_resume_model_path,
        "UPDATE_EPOCHS": hyperparams.n_epochs,
        "NUM_MINIBATCHES": hyperparams.batch_size,
        "GAMMA": hyperparams.gamma,
        "GAE_LAMBDA": hyperparams.gae_lambda,
        "CLIP_EPS": hyperparams.clip_range,
        "ENT_COEF": hyperparams.ent_coef,
        "VF_COEF": hyperparams.vf_coef,
        "MAX_GRAD_NORM": hyperparams.max_grad_norm,
        "ACTIVATION": "tanh",
        "NORMALIZE_ENV": True,
        "DEBUG": True,
        "ACT_DIM": NU,
        "OBS_DIM": NX,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config, env))
    out = train_jit(rng)