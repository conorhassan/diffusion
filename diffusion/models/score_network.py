import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Optional
import math


class TimeEmbedding(eqx.Module):
    mlp: eqx.nn.Sequential
    
    def __init__(self, dim: int, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(dim // 4, dim, key=keys[0]),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(dim, dim, key=keys[1]),
        ])
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        half_dim = 32
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return self.mlp(emb)


class ResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    time_mlp: Optional[eqx.nn.Linear]
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    shortcut: Optional[eqx.nn.Conv2d]
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: Optional[int] = None,
        *,
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, 4)
        
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=keys[1])
        
        self.norm1 = eqx.nn.GroupNorm(8, in_channels)
        self.norm2 = eqx.nn.GroupNorm(8, out_channels)
        
        if time_dim is not None:
            self.time_mlp = eqx.nn.Linear(time_dim, out_channels, key=keys[2])
        else:
            self.time_mlp = None
        
        if in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(in_channels, out_channels, 1, key=keys[3])
        else:
            self.shortcut = None
    
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)
        
        if t is not None and self.time_mlp is not None:
            h = h + self.time_mlp(jax.nn.silu(t))[:, :, None, None]
        
        h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.conv2(h)
        
        if self.shortcut is not None:
            x = self.shortcut(x)
        
        return x + h


class ScoreNetwork(eqx.Module):
    time_embed: TimeEmbedding
    down_blocks: Sequence[ResidualBlock]
    middle_block: ResidualBlock
    up_blocks: Sequence[ResidualBlock]
    final_conv: eqx.nn.Conv2d
    channels: Sequence[int]
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
        time_dim: int = 256,
        *,
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, 2 + 2 * len(channels) + 2)
        key_idx = 0
        
        self.channels = channels
        self.time_embed = TimeEmbedding(time_dim, key=keys[key_idx])
        key_idx += 1
        
        # Downsampling blocks
        self.down_blocks = []
        ch_in = in_channels
        for ch_out in channels:
            self.down_blocks.append(
                ResidualBlock(ch_in, ch_out, time_dim, key=keys[key_idx])
            )
            ch_in = ch_out
            key_idx += 1
        
        # Middle block
        self.middle_block = ResidualBlock(
            channels[-1], channels[-1], time_dim, key=keys[key_idx]
        )
        key_idx += 1
        
        # Upsampling blocks
        self.up_blocks = []
        for ch_out in reversed(channels[:-1]):
            self.up_blocks.append(
                ResidualBlock(ch_in, ch_out, time_dim, key=keys[key_idx])
            )
            ch_in = ch_out
            key_idx += 1
        
        # Final convolution to get score
        self.final_conv = eqx.nn.Conv2d(
            channels[0], in_channels, 3, padding=1, key=keys[key_idx]
        )
    
    def __call__(self, x: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        # Time embedding from noise level
        t_emb = self.time_embed(jnp.log(sigma))
        
        # Downsampling
        h = x
        skip_connections = []
        for block in self.down_blocks:
            h = block(h, t_emb)
            skip_connections.append(h)
            h = jax.nn.avg_pool(h, (2, 2), (2, 2))
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            h = jax.image.resize(h, (h.shape[0], h.shape[1], h.shape[2] * 2, h.shape[3] * 2), method='bilinear')
            skip = skip_connections[-(i + 2)]
            h = h + skip
            h = block(h, t_emb)
        
        # Final score output
        score = self.final_conv(h)
        
        # Scale score by 1/sigma as per theory
        return score / sigma[:, None, None, None]


class SimpleScoreNetwork(eqx.Module):
    """A simpler MLP-based score network for testing and 1D/low-dim data"""
    layers: eqx.nn.Sequential
    time_embed: eqx.nn.Sequential
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 64,
        *,
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, 2)
        
        # Time embedding network
        self.time_embed = eqx.nn.Sequential([
            eqx.nn.Linear(1, time_dim, key=keys[0]),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(time_dim, time_dim, key=keys[0]),
        ])
        
        # Main network
        layers = []
        in_dim = input_dim + time_dim
        for i in range(num_layers):
            layers.extend([
                eqx.nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim, key=keys[1]),
                eqx.nn.Lambda(jax.nn.silu),
            ])
        layers.append(eqx.nn.Linear(hidden_dim, input_dim, key=keys[1]))
        
        self.layers = eqx.nn.Sequential(layers)
    
    def __call__(self, x: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        # Embed noise level
        # Handle both scalar and array sigma
        if sigma.ndim == 0:
            t_emb = self.time_embed(jnp.log(sigma)[None])
        else:
            t_emb = self.time_embed(jnp.log(sigma)[:, None])
        
        # If x is 1D (single sample) and t_emb is 2D, squeeze t_emb
        if x.ndim == 1 and t_emb.ndim == 2:
            t_emb = t_emb.squeeze(0)
        
        # Concatenate input and time embedding
        h = jnp.concatenate([x, t_emb], axis=-1)
        
        # Forward through network
        score = self.layers(h)
        
        # Scale by 1/sigma
        if sigma.ndim == 0:
            return score / sigma
        else:
            return score / sigma[:, None]