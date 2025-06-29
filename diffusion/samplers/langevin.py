import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Callable, Optional, Tuple
import numpy as np


def annealed_langevin_dynamics(
    score_fn: Callable,
    x_init: jnp.ndarray,
    sigmas: jnp.ndarray,
    n_steps_per_sigma: int = 100,
    step_size: float = 0.00001,
    key: jax.random.PRNGKey = None,
    noise_scale: float = 1.0
) -> jnp.ndarray:
    """
    Annealed Langevin dynamics sampling.
    
    Args:
        score_fn: Score function that takes (x, sigma) and returns score
        x_init: Initial samples (e.g., from N(0, sigma_max^2))
        sigmas: Decreasing sequence of noise levels
        n_steps_per_sigma: Number of Langevin steps per noise level
        step_size: Step size for Langevin dynamics
        key: Random key for noise
        noise_scale: Scale for the noise term (set to 0 for deterministic)
    
    Returns:
        Samples from the target distribution
    """
    x = x_init
    
    for i, sigma in enumerate(sigmas):
        # Adjust step size based on sigma
        alpha = step_size * (sigma / sigmas[0]) ** 2
        
        for _ in range(n_steps_per_sigma):
            key, subkey = jax.random.split(key)
            
            # Get score
            score = score_fn(x, jnp.full(x.shape[0], sigma))
            
            # Langevin update
            noise = jax.random.normal(subkey, x.shape)
            x = x + alpha * score + jnp.sqrt(2 * alpha * noise_scale) * noise
    
    return x


def predictor_corrector_sampler(
    score_fn: Callable,
    x_init: jnp.ndarray,
    sigmas: jnp.ndarray,
    predictor_steps: int = 1,
    corrector_steps: int = 1,
    snr: float = 0.16,
    key: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """
    Predictor-Corrector sampling following Song et al. (2021).
    
    Args:
        score_fn: Score function
        x_init: Initial samples
        sigmas: Noise schedule
        predictor_steps: Number of predictor steps
        corrector_steps: Number of corrector steps
        snr: Signal-to-noise ratio for corrector
        key: Random key
    
    Returns:
        Samples
    """
    x = x_init
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Predictor step (reverse diffusion)
        for _ in range(predictor_steps):
            key, subkey = jax.random.split(key)
            score = score_fn(x, jnp.full(x.shape[0], sigma))
            
            # Euler-Maruyama step
            dt = sigma_next - sigma
            diffusion = jnp.sqrt(sigma)
            noise = jax.random.normal(subkey, x.shape)
            
            x = x + dt * diffusion * score + jnp.sqrt(jnp.abs(dt)) * diffusion * noise
        
        # Corrector step (Langevin dynamics)
        for _ in range(corrector_steps):
            key, subkey = jax.random.split(key)
            score = score_fn(x, jnp.full(x.shape[0], sigma))
            
            noise = jax.random.normal(subkey, x.shape)
            step_size = 2 * (snr * jnp.linalg.norm(noise) / jnp.linalg.norm(score)) ** 2
            
            x = x + step_size * score + jnp.sqrt(2 * step_size) * noise
    
    return x


class ODESampler:
    """ODE-based sampler using diffrax for score-based models."""
    
    def __init__(
        self,
        score_fn: Callable,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        rtol: float = 1e-5,
        atol: float = 1e-5
    ):
        self.score_fn = score_fn
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
    
    def sample(
        self,
        x_init: jnp.ndarray,
        t0: float,
        t1: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Sample using probability flow ODE.
        
        Args:
            x_init: Initial condition
            t0: Initial time (typically sigma_max)
            t1: Final time (typically sigma_min)
            key: Random key (unused for ODE)
        
        Returns:
            Samples
        """
        def drift(t, x, args):
            sigma = t
            score = self.score_fn(x, jnp.full(x.shape[0], sigma))
            return -0.5 * sigma * score
        
        term = diffrax.ODETerm(drift)
        
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=x_init,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
        )
        
        return sol.ys[-1]


class SDESampler:
    """SDE-based sampler using diffrax."""
    
    def __init__(
        self,
        score_fn: Callable,
        solver: diffrax.AbstractSolver = diffrax.Euler(),
        dt: float = 0.01
    ):
        self.score_fn = score_fn
        self.solver = solver
        self.dt = dt
    
    def sample(
        self,
        x_init: jnp.ndarray,
        t0: float,
        t1: float,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample using reverse-time SDE.
        
        Args:
            x_init: Initial condition
            t0: Initial time
            t1: Final time
            key: Random key for noise
        
        Returns:
            Samples
        """
        def drift(t, x, args):
            sigma = t
            score = self.score_fn(x, jnp.full(x.shape[0], sigma))
            return -0.5 * sigma * score
        
        def diffusion(t, x, args):
            return jnp.sqrt(t) * jnp.ones_like(x)
        
        brownian_motion = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=1e-3, shape=x_init.shape, key=key
        )
        
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian_motion)
        )
        
        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt,
            y0=x_init,
            saveat=diffrax.SaveAt(ts=[t1])
        )
        
        return sol.ys[-1]