# Score Matching Examples

This directory contains example implementations of score-based generative models.

## Files

- `train_score_matching.py` - Main training script for denoising score matching
- `quick_demo.py` - Quick demo with reduced training steps

## Usage

### Quick Demo (5k steps, ~30 seconds)
```bash
python quick_demo.py
```

### Full Training (30k steps, ~2 minutes)
```bash
python train_score_matching.py --steps 30000 --lr 0.001 --grad-clip 5.0 --n-sigmas 20
```

### Command Line Arguments

- `--steps`: Number of training steps (default: 20000)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--grad-clip`: Gradient clipping threshold (default: 5.0)
- `--n-sigmas`: Number of noise levels (default: 10)
- `--sigma-min`: Minimum noise level (default: 0.01)
- `--sigma-max`: Maximum noise level (default: 1.0)
- `--langevin-steps-per-sigma`: Sampling steps per noise level (default: 100)
- `--seed`: Random seed (default: 42)

## Key Features

1. **Multi-scale Training**: Trains across multiple noise levels (σ) for better generation
2. **Weighted Loss Tracking**: Shows both raw and σ-normalized losses for better monitoring
3. **Sinusoidal Time Embeddings**: 128-dimensional embeddings for noise level conditioning
4. **Proper Score Scaling**: Network outputs score × σ, then divides by σ for correct scaling
5. **Adaptive Sampling**: Langevin dynamics with σ-dependent step sizes

## Output

The script generates two files:
- `training_metrics.png`: Training curves showing raw loss, weighted loss, and gradient norms
- `results.png`: Visualization of original data, generated samples, and score fields

## Implementation Details

The implementation uses:
- JAX for automatic differentiation
- Optax for optimization with gradient clipping
- Proper weight initialization (He init for hidden layers)
- GELU activation functions
- Exponential learning rate decay