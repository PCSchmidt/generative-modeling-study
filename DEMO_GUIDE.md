# Demo Guide: β-VAE Generative Modeling Study

## What This Project Demonstrates

This study implements a β-Variational Autoencoder from scratch in PyTorch and systematically explores the reconstruction–disentanglement trade-off by sweeping β ∈ {0.5, 1.0, 4.0, 10.0} on MNIST.

## Running the Demo

### 1. Setup (~2 min)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### 2. Run the Notebook (~5 min on CPU)
```bash
jupyter notebook notebooks/beta_vae_study.ipynb
```
Execute all cells. MNIST downloads automatically (~12 MB). Training 4 models × 15 epochs takes ~5 min on CPU.

### 3. Key Outputs to Inspect

| Section | What to Look For |
|---------|-----------------|
| §4 Loss Curves | Smooth convergence of total/recon/KL loss |
| §5 β Comparison | How reconstruction degrades as β increases |
| §5 Recon Grid | Side-by-side originals vs. reconstructions at each β |
| §6 Latent Space | Scatter plots showing cluster tightening with higher β |
| §7 Interpolation | Smooth 3→7 digit morphing across β values |
| §8 Generation | Random samples from the prior at each β |
| §8 Summary Table | Quantitative test metrics across all β values |

### 4. Run Tests
```bash
python -m pytest tests/ -v
```
19 tests covering model architecture, loss computation, and training mechanics.

## Talking Points

1. **Why β-VAE over standard VAE?** β > 1 encourages disentangled latent representations — each latent dimension captures an independent factor of variation. This is crucial for interpretability and controllable generation.

2. **The trade-off is real.** β=10 makes the latent space beautifully structured (great for downstream tasks) but produces blurry reconstructions. β=0.5 gives sharp images but a messy latent space.

3. **Mathematical foundation.** The ELBO decomposes into reconstruction accuracy vs. posterior regularization. β directly controls this balance.

4. **Engineering quality.** Modular source code, 19 pytest tests, CI pipeline, evidence PNGs — production-grade project structure.

## Evidence Files

After running the notebook, `evidence/` contains:
- `loss_curves_beta1.png` — Training convergence
- `beta_comparison.png` — Loss curves across β values
- `reconstruction_comparison.png` — Visual quality comparison
- `latent_space_comparison.png` — 2D latent projections
- `latent_interpolation.png` — Digit morphing
- `generated_samples.png` — Samples from the prior
