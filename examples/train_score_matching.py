import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import optax

import sys
sys.path.append('..')

from diffusion.utils.datasets import make_moons


def create_mlp(key, in_dim=128, hidden_dim=256, out_dim=2):
    """Create a simple MLP with better initialization."""
    keys = jax.random.split(key, 6)
    
    # He initialization for ReLU networks
    W1 = jax.random.normal(keys[0], (in_dim, hidden_dim)) * jnp.sqrt(2.0 / in_dim)
    b1 = jnp.zeros(hidden_dim)
    
    W2 = jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros(hidden_dim)
    
    W3 = jax.random.normal(keys[2], (hidden_dim, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim)
    b3 = jnp.zeros(hidden_dim)
    
    # Output layer - smaller initialization
    W4 = jax.random.normal(keys[3], (hidden_dim, out_dim)) * 0.01
    b4 = jnp.zeros(out_dim)
    
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}
    return params


def time_embedding(t, dim=128):
    """Sinusoidal time embeddings."""
    half_dim = dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb


def mlp_forward(params, x, sigma):
    """Forward pass with proper time embedding."""
    batch_size = x.shape[0]
    
    # Create time embedding from sigma
    t = jnp.log(sigma) / 4.0  # Normalize log(sigma) to reasonable range
    t_vec = jnp.full(batch_size, t)
    t_emb = time_embedding(t_vec, 128)
    
    # Concatenate x with full time embedding
    h = jnp.concatenate([x, t_emb], axis=1)
    
    # Network forward pass
    h = jnp.dot(h, params['W1']) + params['b1']
    h = jax.nn.gelu(h)  # GELU often works better than ReLU
    
    h = jnp.dot(h, params['W2']) + params['b2']
    h = jax.nn.gelu(h)
    
    h = jnp.dot(h, params['W3']) + params['b3']
    h = jax.nn.gelu(h)
    
    # Output - no activation
    score_times_sigma = jnp.dot(h, params['W4']) + params['b4']
    
    # Important: Output is score * sigma, so divide by sigma to get score
    return score_times_sigma / sigma


def compute_dsm_loss(params, x_batch, sigma, key):
    """Denoising score matching loss."""
    # Add noise
    eps = jax.random.normal(key, x_batch.shape)
    x_noisy = x_batch + sigma * eps
    
    # Predict score
    score_pred = mlp_forward(params, x_noisy, sigma)
    
    # True score
    score_true = -eps / sigma
    
    # Loss - no weighting needed since we're predicting score directly
    loss = jnp.mean((score_pred - score_true) ** 2)
    
    return loss


def get_sigma_schedule(num_levels=10, sigma_min=0.01, sigma_max=1.0):
    """Get geometric sequence of sigmas."""
    return jnp.geomspace(sigma_min, sigma_max, num_levels)


# Create loss function that samples random sigmas
def loss_fn(params, x_batch, sigmas, key):
    """Loss with random sigma sampling."""
    key1, key2 = jax.random.split(key)
    
    # Sample sigma with importance weights (optional - weight smaller sigmas more)
    # probs = 1.0 / sigmas
    # probs = probs / jnp.sum(probs)
    # idx = jax.random.choice(key1, len(sigmas), p=probs)
    
    # For now, uniform sampling
    idx = jax.random.randint(key1, (), 0, len(sigmas))
    sigma = sigmas[idx]
    
    # Compute loss
    loss = compute_dsm_loss(params, x_batch, sigma, key2)
    
    # Return both raw loss and sigma for tracking
    return loss, sigma


# JIT and create gradient function
loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def clip_grads(grads, max_norm):
    """Clip gradients by global norm."""
    grad_leaves, tree_def = jax.tree_util.tree_flatten(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
    clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    clipped_leaves = [g * clip_factor for g in grad_leaves]
    return jax.tree_util.tree_unflatten(tree_def, clipped_leaves), grad_norm


def train(data, args):
    """Main training loop."""
    # Initialize
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = create_mlp(init_key, in_dim=2+128)  # 2 for x, 128 for time embedding
    
    # Create sigma schedule
    sigmas = get_sigma_schedule(args.n_sigmas, args.sigma_min, args.sigma_max)
    print(f"Sigma schedule: {sigmas}")
    
    # Initialize optimizer with exponential decay
    schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=2000,
        decay_rate=0.9
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    losses = []
    weighted_losses = []
    grad_norms = []
    sigma_losses = {float(s): [] for s in sigmas}  # Track per-sigma losses
    
    for step in tqdm(range(args.steps)):
        # Sample batch
        key, batch_key, loss_key = jax.random.split(key, 3)
        idx = jax.random.choice(batch_key, len(data), (args.batch_size,))
        batch = data[idx]
        
        # Compute loss and gradients
        (loss_val, sigma_used), grads = loss_and_grad_fn(params, batch, sigmas, loss_key)
        
        # Get gradient norm for logging
        grad_leaves, _ = jax.tree_util.tree_flatten(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        
        # Update (optimizer already has gradient clipping)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        
        # Log raw loss
        losses.append(float(loss_val))
        grad_norms.append(float(grad_norm))
        
        # Track per-sigma losses
        sigma_losses[float(sigma_used)].append(float(loss_val))
        
        # Compute weighted loss (normalize by sigma^2 since score ~ 1/sigma)
        weighted_loss = float(loss_val) * float(sigma_used)**2
        weighted_losses.append(weighted_loss)
        
        if step % 500 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else float(loss_val)
            avg_weighted = np.mean(weighted_losses[-100:]) if len(weighted_losses) >= 100 else weighted_loss
            print(f"Step {step}: loss={avg_loss:.4f}, weighted_loss={avg_weighted:.4f}, σ={float(sigma_used):.3f}, grad_norm={float(grad_norm):.4f}")
    
    # Print per-sigma statistics
    print("\nPer-sigma loss statistics:")
    for sigma in sorted(sigma_losses.keys()):
        if sigma_losses[sigma]:
            mean_loss = np.mean(sigma_losses[sigma])
            print(f"  σ={sigma:.3f}: mean_loss={mean_loss:.4f} (n={len(sigma_losses[sigma])})")
    
    return params, losses, weighted_losses, grad_norms, sigmas


def sample_langevin(params, sigmas, n_samples=1000, n_steps=100, args=None):
    """Langevin dynamics sampling with decreasing noise."""
    key = jax.random.PRNGKey(999)
    
    # Initialize from pure noise matching largest sigma
    key, init_key = jax.random.split(key)
    x = jax.random.normal(init_key, (n_samples, 2)) * sigmas[0]
    
    # Sample with decreasing sigmas
    for i, sigma in enumerate(tqdm(sigmas, desc="Sampling")):
        # Adaptive step size based on sigma
        eps = args.langevin_step * (sigma / sigmas[0]) ** 2
        
        for _ in range(n_steps):
            key, noise_key = jax.random.split(key)
            
            # Compute score
            score = mlp_forward(params, x, sigma)
            
            # Langevin dynamics update
            noise = jax.random.normal(noise_key, x.shape)
            x = x + eps * score + jnp.sqrt(2 * eps) * noise
    
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--n-sigmas', type=int, default=10)
    parser.add_argument('--sigma-min', type=float, default=0.01)
    parser.add_argument('--sigma-max', type=float, default=1.0)
    parser.add_argument('--langevin-step', type=float, default=0.01)
    parser.add_argument('--langevin-steps-per-sigma', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Generate data
    print("Generating data...")
    key = jax.random.PRNGKey(args.seed)
    data = make_moons(n_samples=10000, noise=0.05, key=key)
    
    # Normalize to zero mean and unit variance
    data_mean = data.mean(0)
    data_std = data.std(0)
    data = (data - data_mean) / data_std
    
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {data.mean(0)}, std: {data.std(0)}")
    
    # Train
    print(f"\nTraining configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gradient clipping: {args.grad_clip}")
    print(f"  Sigma range: [{args.sigma_min}, {args.sigma_max}] with {args.n_sigmas} levels")
    
    params, losses, weighted_losses, grad_norms, sigmas = train(data, args)
    
    # Plot training
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Raw loss
    ax1.plot(losses, alpha=0.3, label='Raw')
    if len(losses) > 100:
        smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
        ax1.plot(range(50, len(smoothed)+50), smoothed, label='Smoothed (100)', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Raw Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # Weighted loss
    ax2.plot(weighted_losses, alpha=0.3, label='Weighted')
    if len(weighted_losses) > 100:
        smoothed = np.convolve(weighted_losses, np.ones(100)/100, mode='valid')
        ax2.plot(range(50, len(smoothed)+50), smoothed, label='Smoothed (100)', linewidth=2, color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Weighted Loss (Loss × σ²)')
    ax2.set_title('Sigma-Normalized Training Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Gradient norms
    ax3.plot(grad_norms)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norms')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Loss comparison
    ax4.semilogy(losses, alpha=0.5, label='Raw Loss')
    ax4.semilogy(weighted_losses, alpha=0.5, label='Weighted Loss')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Comparison (Log Scale)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Initial raw loss: {losses[0]:.4f}")
    print(f"Final raw loss: {losses[-1]:.4f}")
    print(f"Raw loss reduction: {losses[0]/losses[-1]:.1f}x")
    print(f"Initial weighted loss: {weighted_losses[0]:.4f}")
    print(f"Final weighted loss: {weighted_losses[-1]:.4f}")
    print(f"Weighted loss reduction: {weighted_losses[0]/weighted_losses[-1]:.1f}x")
    
    # Test score matching at different sigmas
    print("\nTesting score matching quality:")
    test_key = jax.random.PRNGKey(123)
    for sigma in [0.01, 0.1, 0.5, 1.0]:
        test_data = data[:100]
        test_key, noise_key = jax.random.split(test_key)
        eps = jax.random.normal(noise_key, test_data.shape)
        x_noisy = test_data + sigma * eps
        
        score_pred = mlp_forward(params, x_noisy, sigma)
        score_true = -eps / sigma
        
        mse = jnp.mean((score_pred - score_true) ** 2)
        correlation = jnp.corrcoef(score_pred.flatten(), score_true.flatten())[0, 1]
        
        print(f"  σ={sigma:.2f}: MSE = {mse:.4f}, Corr = {correlation:.4f}")
    
    # Generate samples
    print(f"\nGenerating samples...")
    print(f"  Using {len(sigmas)} sigmas with {args.langevin_steps_per_sigma} steps each")
    
    # Use more sigmas for sampling for smoother transitions
    sampling_sigmas = get_sigma_schedule(50, args.sigma_min, args.sigma_max)[::-1]  # Reverse: high to low
    samples = sample_langevin(params, sampling_sigmas, n_samples=2000, 
                            n_steps=args.langevin_steps_per_sigma, args=args)
    
    # Denormalize
    samples_original = samples * data_std + data_mean
    data_original = data * data_std + data_mean
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original data
    ax = axes[0, 0]
    ax.scatter(data_original[:1000, 0], data_original[:1000, 1], alpha=0.5, s=10)
    ax.set_title("Original Data")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Generated samples
    ax = axes[0, 1]
    ax.scatter(samples_original[:, 0], samples_original[:, 1], alpha=0.5, s=10)
    ax.set_title("Generated Samples")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Score field at sigma=0.3
    ax = axes[1, 0]
    xx = np.linspace(-2, 2, 20)
    yy = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(xx, yy)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    
    scores = mlp_forward(params, points, 0.3)
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)
    
    ax.quiver(X, Y, U, V, alpha=0.6)
    ax.scatter(data[:200, 0], data[:200, 1], alpha=0.3, s=5, c='red')
    ax.set_title("Score Field (σ=0.3)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Score field at sigma=0.01
    ax = axes[1, 1]
    scores_fine = mlp_forward(params, points, 0.01)
    U_fine = scores_fine[:, 0].reshape(X.shape)
    V_fine = scores_fine[:, 1].reshape(X.shape)
    
    ax.quiver(X, Y, U_fine, V_fine, alpha=0.6)
    ax.scatter(data[:200, 0], data[:200, 1], alpha=0.3, s=5, c='red')
    ax.set_title("Score Field (σ=0.01)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.close()
    
    print("\nDone! Check training_metrics.png and results.png")


if __name__ == "__main__":
    main()