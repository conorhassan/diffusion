import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm
import numpy as np

from .score_matching import annealed_score_matching_loss, get_sigma_schedule


class ScoreMatchingTrainer:
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
        self.loss_fn = loss_fn or annealed_score_matching_loss
    
    @eqx.filter_jit
    def train_step(
        self,
        model: eqx.Module,
        opt_state: Any,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[eqx.Module, Any, float]:
        def loss_fn(model, batch, key):
            score_fn = lambda x, sigma: jax.vmap(model)(x, sigma)
            return self.loss_fn(score_fn, batch, self.sigma_schedule, key)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
        
        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    def train(
        self,
        data_loader,
        num_epochs: int,
        key: jax.random.PRNGKey,
        log_interval: int = 100
    ) -> Dict[str, list]:
        train_key = key
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, batch in enumerate(pbar):
                train_key, step_key = jax.random.split(train_key)
                
                self.model, self.opt_state, loss = self.train_step(
                    self.model, self.opt_state, batch, step_key
                )
                
                epoch_losses.append(float(loss))
                
                if i % log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    losses.append(avg_loss)
        
        return {"losses": losses}


def create_trainer(
    model: eqx.Module,
    learning_rate: float = 1e-4,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    num_scales: int = 10,
    optimizer_config: Optional[Dict[str, Any]] = None
) -> ScoreMatchingTrainer:
    # Default optimizer config
    if optimizer_config is None:
        optimizer = optax.adam(learning_rate)
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
    
    return ScoreMatchingTrainer(model, optimizer, sigma_schedule)