import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import equinox as eqx


def denoising_score_matching_loss(
    score_fn: Callable,
    x: jnp.ndarray,
    sigma: float,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    noise = jax.random.normal(key, x.shape)
    x_noisy = x + sigma * noise
    sigmas = jnp.full(x.shape[0], sigma)
    score = score_fn(x_noisy, sigmas)
    target_score = -noise / sigma
    return jnp.mean((score - target_score) ** 2)


def sliced_score_matching_loss(
    score_fn: Callable,
    x: jnp.ndarray,
    n_projections: int = 1,
    key: jax.random.PRNGKey = None
) -> jnp.ndarray:
    batch_size = x.shape[0]
    
    def single_sample_loss(x_i):
        score = score_fn(x_i)
        
        losses = []
        for _ in range(n_projections):
            key, subkey = jax.random.split(key)
            v = jax.random.normal(subkey, x_i.shape)
            v = v / jnp.linalg.norm(v)
            
            score_v = jnp.dot(score, v)
            grad_v = jax.grad(lambda x: jnp.dot(score_fn(x), v))(x_i)
            loss = score_v ** 2 / 2 + jnp.dot(grad_v, v)
            losses.append(loss)
        
        return jnp.mean(jnp.array(losses))
    
    return jnp.mean(jax.vmap(single_sample_loss)(x))


def get_sigma_schedule(
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    num_scales: int = 10,
    schedule_type: str = "geometric"
) -> jnp.ndarray:
    if schedule_type == "geometric":
        sigmas = jnp.geomspace(sigma_min, sigma_max, num_scales)
    elif schedule_type == "linear":
        sigmas = jnp.linspace(sigma_min, sigma_max, num_scales)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return sigmas


def annealed_score_matching_loss(
    score_fn: Callable,
    x: jnp.ndarray,
    sigmas: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    key_sigma, key_noise = jax.random.split(key)
    
    sigma_idx = jax.random.randint(key_sigma, (), 0, len(sigmas))
    sigma = sigmas[sigma_idx]
    
    return denoising_score_matching_loss(score_fn, x, sigma, key_noise)