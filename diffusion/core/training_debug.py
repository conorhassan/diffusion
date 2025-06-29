import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, Any, Tuple, Optional, Callable, List
from tqdm import tqdm
import numpy as np

from .score_matching import get_sigma_schedule


class DebugScoreMatchingTrainer:
    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        sigma_schedule: jnp.ndarray,
        loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        self.sigma_schedule = sigma_schedule
        self.loss_fn = loss_fn
        
        # Statistics tracking
        self.stats = {
            'losses': [],
            'grad_norms': [],
            'score_norms': [],
            'per_sigma_losses': {float(sigma): [] for sigma in sigma_schedule},
            'param_norms': [],
            'update_norms': []
        }
    
    @eqx.filter_jit
    def compute_per_sigma_losses(
        self,
        model: eqx.Module,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Compute loss for each sigma value."""
        losses = []
        
        for i, sigma in enumerate(self.sigma_schedule):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, batch.shape)
            x_noisy = batch + sigma * noise
            sigmas = jnp.full(batch.shape[0], sigma)
            
            # Get scores
            scores = jax.vmap(model)(x_noisy, sigmas)
            target_scores = -noise / sigma
            
            # Compute loss for this sigma
            loss = jnp.mean((scores - target_scores) ** 2)
            losses.append(loss)
        
        return jnp.array(losses)
    
    @eqx.filter_jit
    def debug_train_step(
        self,
        model: eqx.Module,
        opt_state: Any,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[eqx.Module, Any, Dict[str, Any]]:
        """Training step with detailed statistics."""
        
        # Split keys
        key1, key2, key3 = jax.random.split(key, 3)
        
        def loss_fn(model, batch, key):
            # Sample random sigma
            sigma_idx = jax.random.randint(key, (), 0, len(self.sigma_schedule))
            sigma = self.sigma_schedule[sigma_idx]
            
            # Add noise
            noise = jax.random.normal(key, batch.shape)
            x_noisy = batch + sigma * noise
            sigmas = jnp.full(batch.shape[0], sigma)
            
            # Get scores
            scores = jax.vmap(model)(x_noisy, sigmas)
            target_scores = -noise / sigma
            
            # Compute loss
            loss = jnp.mean((scores - target_scores) ** 2)
            
            # Additional metrics
            score_norm = jnp.mean(jnp.linalg.norm(scores, axis=1))
            
            return loss, (score_norm, scores, target_scores)
        
        # Compute loss and gradients
        (loss, (score_norm, scores, target_scores)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key1)
        
        # Compute gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads) if isinstance(g, jnp.ndarray)))
        
        # Compute parameter norm before update
        param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(model) if isinstance(p, jnp.ndarray)))
        
        # Apply updates
        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        # Compute update norm
        update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree_util.tree_leaves(updates) if isinstance(u, jnp.ndarray)))
        
        # Compute per-sigma losses
        per_sigma_losses = self.compute_per_sigma_losses(model, batch, key2)
        
        return model, opt_state, (loss, grad_norm, score_norm, param_norm, update_norm, per_sigma_losses, jnp.std(scores), jnp.std(target_scores))
    
    def train(
        self,
        data_loader,
        num_epochs: int,
        key: jax.random.PRNGKey,
        log_interval: int = 10,
        debug_interval: int = 100
    ) -> Dict[str, Any]:
        train_key = key
        step_count = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, batch in enumerate(pbar):
                train_key, step_key = jax.random.split(train_key)
                
                self.model, self.opt_state, step_results = self.debug_train_step(
                    self.model, self.opt_state, batch, step_key
                )
                
                # Unpack results
                loss, grad_norm, score_norm, param_norm, update_norm, per_sigma_losses, score_std, target_score_std = step_results
                
                # Record statistics
                self.stats['losses'].append(float(loss))
                self.stats['grad_norms'].append(float(grad_norm))
                self.stats['score_norms'].append(float(score_norm))
                self.stats['param_norms'].append(float(param_norm))
                self.stats['update_norms'].append(float(update_norm))
                
                # Record per-sigma losses
                for j, sigma in enumerate(self.sigma_schedule):
                    self.stats['per_sigma_losses'][float(sigma)].append(float(per_sigma_losses[j]))
                
                epoch_losses.append(float(loss))
                
                if i % log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "grad_norm": f"{float(grad_norm):.4f}",
                        "score_norm": f"{float(score_norm):.4f}"
                    })
                
                if step_count % debug_interval == 0:
                    print(f"\nStep {step_count} debug info:")
                    print(f"  Loss: {float(loss):.6f}")
                    print(f"  Gradient norm: {float(grad_norm):.6f}")
                    print(f"  Score norm: {float(score_norm):.6f}")
                    print(f"  Score std: {float(score_std):.6f}")
                    print(f"  Target score std: {float(target_score_std):.6f}")
                    print(f"  Param norm: {float(param_norm):.6f}")
                    print(f"  Update norm: {float(update_norm):.6f}")
                    print("  Per-sigma losses:")
                    for j, sigma in enumerate(self.sigma_schedule):
                        print(f"    σ={float(sigma):.2f}: {float(per_sigma_losses[j]):.6f}")
                
                step_count += 1
        
        return self.stats


def create_debug_trainer(
    model: eqx.Module,
    learning_rate: float = 1e-4,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    num_scales: int = 10,
    optimizer_config: Optional[Dict[str, Any]] = None
) -> DebugScoreMatchingTrainer:
    # Default optimizer config with gradient clipping
    if optimizer_config is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adam(learning_rate)
        )
    else:
        opt_name = optimizer_config.pop("name", "adam")
        if opt_name == "adam":
            optimizer = optax.adam(**optimizer_config)
        elif opt_name == "sgd":
            optimizer = optax.sgd(**optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    # Create sigma schedule
    sigma_schedule = get_sigma_schedule(sigma_min, sigma_max, num_scales)
    
    return DebugScoreMatchingTrainer(model, optimizer, sigma_schedule)