#!/usr/bin/env python3
"""Create minimal aesthetic denoising visualization."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
from tqdm import tqdm
import argparse

import sys
sys.path.append('..')

from diffusion.utils.datasets import make_moons
from train_score_matching import train, mlp_forward, get_sigma_schedule


def create_minimal_animation(params, args):
    """Create minimal animation with no text or excess space."""
    
    # Generate trajectory
    key = jax.random.PRNGKey(args.seed)
    sigmas = get_sigma_schedule(args.n_sigmas, 0.01, 1.0)[::-1]
    
    # Initialize from noise
    key, init_key = jax.random.split(key)
    x = jax.random.normal(init_key, (args.n_samples, 2)) * sigmas[0]
    
    trajectory = [x.copy()]
    
    # Denoise
    print("Generating denoising trajectory...")
    for i, sigma in enumerate(tqdm(sigmas)):
        step_size = 0.1 * (sigma / sigmas[0]) ** 2
        
        for j in range(args.steps_per_sigma):
            key, noise_key = jax.random.split(key)
            score = mlp_forward(params, x, sigma)
            noise = jax.random.normal(noise_key, x.shape)
            x = x + step_size * score + jnp.sqrt(2 * step_size) * noise
            
            if j % max(1, args.steps_per_sigma // 3) == 0:
                trajectory.append(x.copy())
    
    # Normalize for display
    key = jax.random.PRNGKey(args.seed)
    ref_data = make_moons(n_samples=1000, noise=0.05, key=key)
    data_mean = ref_data.mean(0)
    data_std = ref_data.std(0)
    trajectory = [(f * data_std + data_mean) for f in trajectory]
    
    # Create figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 6), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.axis('off')
    
    # Initialize scatter
    scatter = ax.scatter([], [], alpha=0.8, s=20, c=[], cmap='plasma', vmin=0, vmax=1)
    
    def animate(i):
        points = trajectory[i]
        colors = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
        scatter.set_offsets(points)
        scatter.set_array(colors)
        return scatter,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(trajectory),
        interval=args.interval, blit=True
    )
    
    # Save with minimal padding
    print(f"Saving to {args.output}...")
    writer = PillowWriter(fps=args.fps)
    anim.save(args.output, writer=writer, savefig_kwargs={'facecolor': 'black', 'pad_inches': 0})
    plt.close()
    print(f"Animation saved!")
    
    # Create keyframes
    create_minimal_keyframes(trajectory)


def create_minimal_keyframes(trajectory):
    """Create minimal keyframes strip."""
    plt.style.use('dark_background')
    
    n_frames = len(trajectory)
    indices = [0, n_frames//6, 2*n_frames//6, 3*n_frames//6, 4*n_frames//6, 5*n_frames//6, -1]
    
    fig = plt.figure(figsize=(14, 2), facecolor='black')
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, 7, i+1)
        points = trajectory[idx]
        
        colors = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
        
        ax.scatter(points[:, 0], points[:, 1], 
                  alpha=0.8, s=6, c=colors, cmap='plasma', vmin=0, vmax=1)
        
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0)
    plt.savefig('minimal_frames.png', dpi=300, facecolor='black', 
                bbox_inches='tight', pad_inches=0)
    print("Keyframes saved to minimal_frames.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=800)
    parser.add_argument('--n-sigmas', type=int, default=150)
    parser.add_argument('--steps-per-sigma', type=int, default=15)
    parser.add_argument('--output', type=str, default='minimal.gif')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--interval', type=int, default=33)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Train model
    print("Training model...")
    key = jax.random.PRNGKey(args.seed)
    data = make_moons(n_samples=10000, noise=0.05, key=key)
    data = (data - data.mean(0)) / data.std(0)
    
    class TrainArgs:
        steps = 10000
        batch_size = 256
        lr = 0.0005
        grad_clip = 5.0
        n_sigmas = 50
        sigma_min = 0.01
        sigma_max = 1.0
        seed = args.seed
    
    params, _, _, _, _ = train(data, TrainArgs())
    
    # Create animation
    create_minimal_animation(params, args)


if __name__ == "__main__":
    main()