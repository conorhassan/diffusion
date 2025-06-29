import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Generate two-moons dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        key: Random key for reproducibility
        
    Returns:
        Array of shape (n_samples, 2) containing the dataset
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key1, key2 = jax.random.split(key)
    
    n_samples_per_moon = n_samples // 2
    
    # First moon
    theta1 = jnp.linspace(0, jnp.pi, n_samples_per_moon)
    x1 = jnp.cos(theta1)
    y1 = jnp.sin(theta1)
    moon1 = jnp.stack([x1, y1], axis=1)
    
    # Second moon (shifted and flipped)
    theta2 = jnp.linspace(0, jnp.pi, n_samples - n_samples_per_moon)
    x2 = 1 - jnp.cos(theta2)
    y2 = 0.5 - jnp.sin(theta2)
    moon2 = jnp.stack([x2, y2], axis=1)
    
    # Combine moons
    data = jnp.concatenate([moon1, moon2], axis=0)
    
    # Add noise
    noise_data = noise * jax.random.normal(key1, data.shape)
    data = data + noise_data
    
    # Shuffle
    perm = jax.random.permutation(key2, n_samples)
    data = data[perm]
    
    return data


def make_swiss_roll(
    n_samples: int = 1000,
    noise: float = 0.1,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """Generate Swiss roll dataset in 2D."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key1, key2 = jax.random.split(key)
    
    t = 1.5 * jnp.pi * (1 + 2 * jax.random.uniform(key1, (n_samples,)))
    x = t * jnp.cos(t)
    y = t * jnp.sin(t)
    
    data = jnp.stack([x, y], axis=1)
    data = data + noise * jax.random.normal(key2, data.shape)
    
    # Normalize to [-2, 2]
    data = 4 * (data - data.min()) / (data.max() - data.min()) - 2
    
    return data


def make_gaussian_mixture(
    n_samples: int = 1000,
    n_components: int = 8,
    radius: float = 2.0,
    std: float = 0.2,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """Generate mixture of Gaussians arranged in a circle."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key1, key2 = jax.random.split(key)
    
    # Centers arranged in a circle
    angles = jnp.linspace(0, 2 * jnp.pi, n_components, endpoint=False)
    centers = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    
    # Sample component assignments
    components = jax.random.choice(key1, n_components, shape=(n_samples,))
    
    # Sample from each component
    data = centers[components] + std * jax.random.normal(key2, (n_samples, 2))
    
    return data


class DataLoader:
    """Simple data loader for batching."""
    
    def __init__(self, data: jnp.ndarray, batch_size: int, shuffle: bool = True, key: Optional[jax.random.PRNGKey] = None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.n_samples = data.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            perm = jax.random.permutation(subkey, self.n_samples)
            data = self.data[perm]
        else:
            data = self.data
        
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.n_samples)
            yield data[start:end]
    
    def __len__(self):
        return self.n_batches