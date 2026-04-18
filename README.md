# generative-modeling-study

A from-scratch study of **β-Variational Autoencoders (β-VAE)** on MNIST, exploring how the β hyperparameter controls the trade-off between reconstruction fidelity and latent disentanglement.

![CI](https://github.com/PCSchmidt/generative-modeling-study/actions/workflows/ci.yml/badge.svg)

## Highlights

| β Value | Reconstruction Loss | KL Divergence | Behavior |
|---------|-------------------|---------------|----------|
| 0.5 | Lowest | Highest | Sharp reconstructions, unstructured latent space |
| 1.0 | Baseline | Baseline | Standard VAE |
| 4.0 | Moderate | Low | Good disentanglement–quality balance |
| 10.0 | Highest | Lowest | Maximally disentangled, blurry outputs |

## Project Structure

```
generative-modeling-study/
├── src/
│   ├── models.py          # β-VAE architecture (Encoder, Decoder, BetaVAE)
│   ├── training.py         # Loss function, training loop, evaluation
│   └── visualization.py    # Loss curves, reconstructions, latent plots
├── notebooks/
│   └── beta_vae_study.ipynb  # Full study notebook with 9 sections
├── tests/
│   ├── test_models.py       # 11 tests for model architecture
│   └── test_training.py     # 8 tests for training utilities
├── evidence/                # Exported PNG evidence from notebook runs
└── pyproject.toml
```

## Key Concepts

- **Reparameterization trick**: Enables backprop through stochastic sampling by expressing z = μ + σ·ε where ε ~ N(0,I)
- **ELBO loss**: L = BCE(recon, target) + β · KL(q(z|x) || p(z))
- **Disentanglement**: Higher β pushes the latent posterior toward N(0,I), encouraging independent latent factors
- **Convolutional VAE**: Stride-2 convolutions for spatial downsampling; transposed convolutions for upsampling

## PDF Report

A fully executed notebook with all outputs is available as a PDF:  
[`notebooks/beta_vae_study.pdf`](notebooks/beta_vae_study.pdf)

## Quick Start

```bash
# Install (CPU-only PyTorch recommended for local dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Open the study notebook
jupyter notebook notebooks/beta_vae_study.ipynb
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision, matplotlib, numpy, pandas

## Author

Chris Schmidt — MS Applied Mathematics | AI Engineering MSE (JHU)

## License

MIT
