# Sign Language World Model — Continuous Latent World Model (CLWM)

## Overview

A world model approach to sign language recognition and translation. Instead of
frame-by-frame classification, the model maintains a **continuous latent state**
z_t representing the evolving "world state" of the signing scene.

Key innovation (SCI novelty):
- **Continuous latent world model** for sign language (not discrete tokens)
- State transition: p(z_t | z_{<t}, x_{<t}) = N(mu_theta, Sigma_theta)
- Enables physical prediction, occlusion handling, and temporal coherence

## Architecture

```
RGB-D Video -> I3D Encoder -> VQ-VAE Tokenizer -> World Model -> LLM Translator -> Text
                      |
            Temporal Transformer (latent dynamics)
```

## Components

- **ContinuousWorldModel**: Gaussian latent state with dynamics prior
- **VQVAETokenizer**: Discretizes continuous latent into discrete tokens
- **TemporalTransformer**: Causal sequence modeling of latent trajectories
- **SignLanguageWorldModel**: Full pipeline with gloss prediction head

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
./start.sh train
python main.py --config configs/default.yaml

# Generation demo
./start.sh generate
```

## Reference

- Ha and Schmidhuber, "World Models" (2018) arXiv:1803.10122
- DreamerV3 (2023)
- CNNsa-LSTM sign language recognition, Scientific Reports (2024)
