import jax
import jax.numpy as jnp
import optax
from typing import Sequence, NamedTuple, Any
import equinox as eqx
import os



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
    







def save_network(model: ActorCritic, checkpoint_file: str):
    # Save model parameters using Equinox's serialization
    network_checkpoint = {
        "model": eqx.filter(model, eqx.is_inexact_array),
    }
    eqx.tree_serialise_leaves(checkpoint_file, network_checkpoint)




def load_network(model: ActorCritic, checkpoint_file: str) -> ActorCritic:
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")
    
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_file, {
        "network_params": eqx.filter(model, eqx.is_array)
    })
    
    network_params = checkpoint["network_params"]
    static_network = eqx.filter(model, lambda x: not eqx.is_array(x))
    model = eqx.combine(network_params, static_network)
    
    
    print(f"Network parameters loaded from {checkpoint_file}")
    
    return model

