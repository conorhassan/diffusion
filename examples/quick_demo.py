#!/usr/bin/env python3
"""Quick demo of score matching on two-moons dataset."""

import subprocess
import sys

# Default parameters for a quick demo
default_args = [
    "python", "train_score_matching.py",
    "--steps", "5000",
    "--lr", "0.001",
    "--grad-clip", "5.0",
    "--n-sigmas", "10",
    "--sigma-min", "0.01",
    "--sigma-max", "1.0",
    "--langevin-steps-per-sigma", "50"
]

if __name__ == "__main__":
    print("Running quick score matching demo...")
    print("This will train for 5000 steps (about 30 seconds)")
    print("\nFor full training, use:")
    print("  python train_score_matching.py --steps 30000")
    print()
    
    # Run with any additional arguments passed
    subprocess.run(default_args + sys.argv[1:])