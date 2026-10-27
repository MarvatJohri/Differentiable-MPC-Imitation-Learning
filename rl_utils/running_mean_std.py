"""

For reward and obs normalization

"""


from typing import NamedTuple
import jax
import jax.numpy as jnp



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
