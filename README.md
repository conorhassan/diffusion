# Diffusion Models in JAX

![Denoising Process](examples/minimal.gif)

An implementation of diffusion models in JAX, building towards implementing various diffusion-related papers.
## Overview

This repository provides a clean, modular implementation of diffusion models in JAX [TODO]. The goal is to create reusable components for implementing and experimenting with different diffusion model variants from the literature.

### What are Diffusion Models?

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. There are two main perspectives:

1. **Denoising Diffusion Probabilistic Models (DDPMs)**: Learn to reverse a fixed forward noising process
2. **Score-Based Generative Models**: Learn the score function (gradient of log density) at various noise levels

These perspectives are mathematically equivalent and connected through Tweedie's formula [TODO: show the basics].

## Current Implementation

### Denoising Score Matching

We have currently implemented denoising score matching, which trains a neural network to predict the score function:

```
∇_x log p(x) ≈ -1/σ² (x - x_clean)
```

See `examples/train_score_matching.py` for an example on the two-moons dataset.

## Project Structure

```
diffusion/
├── core/
│   └── score_matching.py      # Core loss functions
├── models/
│   └── score_network.py       # Neural network architectures
├── samplers/
│   └── langevin.py           # Sampling algorithms
├── utils/
│   └── datasets.py           # Dataset utilities
└── examples/
    ├── train_score_matching.py # Main training script
    └── quick_demo.py          # Quick demo script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion.git
cd diffusion

# Install dependencies
uv pip install -e .
```

## Quick Start

```bash
# Quick demo (5k steps, ~30 seconds)
cd examples
python quick_demo.py

# Full training (30k steps, ~2 minutes)
python train_score_matching.py --steps 30000 --lr 0.001
```

## Roadmap

### Near-term Goals

1. **DDPM Implementation**
   - Implement the DDPM framework alongside score matching
   - Show equivalence between the two approaches
   - Add DDIM sampling for faster generation

2. **SDE/ODE Framework**
   - Integrate with diffrax for SDE/ODE solving
   - Implement VP-SDE, VE-SDE, and sub-VP variants (what are these?)
   - Add predictor-corrector sampling (what are these?)

3. **Better Architectures**
   - U-Net implementation with attention
   - Time-dependent layer normalization (see Karras paper I guess)
   - Self-attention at multiple resolutions

4. **Conditional Generation**
   - Class-conditional diffusion
   - Classifier-free guidance
   - Text conditioning infrastructure

### Future Implementations

- **Latent Diffusion Models**: Diffusion in learned latent spaces
- **Consistency Models**: Direct single-step generation
- **Flow Matching**: Optimal transport perspective
- **Diffusion Transformers (DiT)**: Transformer-based diffusion models
- **ControlNet**: Adding spatial control to diffusion models
- **DreamBooth**: Few-shot personalization

**The things that I actually would like to do:** 
- diffusion transformers
- SMC sampling/diffusion paper (oxford group june 2025)
- DDIM
- Speculztive decoding with diffusion models

## References

Key papers that inspire this implementation:

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
2. [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) (Song et al., 2021)
3. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
4. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2021)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation draws inspiration from:
- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) by Yang Song
- [diffusion](https://github.com/hojonathanho/diffusion) by Jonathan Ho
- The JAX and Equinox communities for excellent tools
