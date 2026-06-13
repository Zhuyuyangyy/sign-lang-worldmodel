# Sign Language World Model

**Continuous Latent World Model (CLWM) for Sign Language Recognition and Translation**

A research prototype that applies world model methodology to sign language translation. Instead of frame-by-frame classification, the model maintains a continuous latent state representing the evolving "world state" of the signing scene, enabling physical prediction, occlusion handling, and temporal coherence.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Benchmarks](#benchmarks)
- [Research](#research)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)

---

## Overview

Sign language translation is challenging because signing is continuous, spatial, and context-dependent. Traditional approaches classify individual frames or short clips, losing the temporal dynamics that carry grammatical and semantic information. Hand occlusion (hands covering the face or each other) further degrades frame-level methods.

This project introduces a **Continuous Latent World Model (CLWM)** that addresses these challenges by modeling sign language as a dynamical system:

- The model maintains a continuous Gaussian latent state `z_t` that captures the evolving "world state" of the signing scene.
- State transitions follow `p(z_t | z_{<t}, x_{<t}) = N(mu_theta, Sigma_theta)`, giving the model uncertainty estimation crucial for handling occlusion.
- A VQ-VAE tokenizer bridges the continuous world model with discrete token-based LLM translation.

This approach enables three capabilities absent in frame-by-frame methods:
1. **Physical Prediction**: The model can "imagine" future frames by rolling out its dynamics model.
2. **Occlusion Robustness**: The denoised latent state is robust to missing visual data.
3. **Temporal Coherence**: Smooth state transitions enforce physically plausible sign trajectories.

---

## Key Features

### Continuous Latent World Model

- Gaussian latent state with learned dynamics prior
- Posterior update combining prior predictions with observations
- Reparameterization trick for end-to-end gradient flow
- Free bits mechanism to prevent KL collapse (Kingma & Welling, 2015)

### VQ-VAE Tokenizer

- Discretizes continuous latent into 8192 discrete codes
- Exponential moving average (EMA) codebook updates for training stability
- Straight-through estimator for gradient propagation
- Bridges continuous world model with autoregressive LLM translation

### Temporal Transformer

- Causal self-attention (can only attend to past states)
- Sinusoidal positional encoding for temporal structure
- Residual feed-forward blocks with LayerNorm and GELU
- Configurable depth (default 8 layers) and width (default 256 dim)

### Sign Language Translation Pipeline

```
RGB-D Video -> I3D Encoder -> VQ-VAE Tokenizer -> World Model -> LLM Translator -> Text
                      |
            Temporal Transformer (latent dynamics)
```

### Training Infrastructure

- PyTorch-based training with AdamW optimizer
- OneCycleLR learning rate scheduling
- Gradient accumulation support
- Gradient norm clipping
- Checkpoint saving with best-model tracking
- YAML-based configuration

---

## Architecture

```
+------------------------------------------------------------------+
|                   SignLanguageWorldModel                          |
|                                                                   |
|  +-------------------+     +-------------------+                 |
|  | Visual Encoder    |     | VQ-VAE Tokenizer  |                 |
|  | (Linear 2048->D)  |     | (8192 codes)      |                 |
|  +--------+----------+     +--------+----------+                 |
|           |                          |                            |
|           v                          v                            |
|  +--------------------------------------------------------------+|
|  |              ContinuousWorldModel                             ||
|  |                                                               ||
|  |  +------------------+   +------------------+                 ||
|  |  | Obs Encoder      |   | State Parameters |                 ||
|  |  | (Linear+LN+GELU) |   | mu_theta, sigma  |                 ||
|  |  +--------+---------+   +--------+---------+                 ||
|  |           |                      |                            ||
|  |           v                      v                            ||
|  |  +------------------+   +------------------+                 ||
|  |  | Dynamics Model   |   | Posterior Model  |                 ||
|  |  | (TemporalTrans)  |   | (TemporalTrans)  |                 ||
|  |  | 8 layers, causal |   | 2 layers         |                 ||
|  |  +--------+---------+   +--------+---------+                 ||
|  |           |                      |                            ||
|  |           +----------+-----------+                            ||
|  |                      |                                        ||
|  |                      v                                        ||
|  |  +------------------+   +------------------+                 ||
|  |  | Decoder          |   | Sign Head        |                 ||
|  |  | (obs prediction) |   | (gloss classify) |                 ||
|  |  +------------------+   +------------------+                 ||
|  +--------------------------------------------------------------+|
+------------------------------------------------------------------+
```

**World Model Loss:**

```
L_total = L_reconstruction + beta * L_KL

Where:
  L_reconstruction = ||obs_seq - obs_pred||^2
  L_KL = KL(q(z|x) || p(z|z_{<t}))
       = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q - mu_p)^2)/var_p - 1)
  q(z|x)     = N(z_mean, exp(z_logvar))      -- posterior
  p(z|z_{<t}) = N(z_prior_mean, exp(z_prior_logvar)) -- dynamics prior
  beta = 0.1 (beta-VAE weighting)
  Free bits = 0.5 * latent_dim (prevents KL collapse)
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | PyTorch 2.0+ | Deep learning framework |
| Model | Custom Transformer + VQ-VAE | World model architecture |
| Language | Python 3.10+ | Implementation |
| Config | PyYAML | Configuration management |
| Numerical | NumPy | Data processing |
| GPU | CUDA (optional) | Accelerated training |

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
cd sign-lang-worldmodel
pip install -r requirements.txt
```

### Training

```bash
# Using the start script
./start.sh train

# Or directly
python main.py --config configs/default.yaml
```

### Generation Demo

```bash
# Using the start script
./start.sh generate

# Or directly
python main.py --config configs/default.yaml --exp-dir experiments
```

### Smoke Test

```bash
python -c "
from src.signlang.world_model.signer_world_model import SignLanguageWorldModel
import torch

model = SignLanguageWorldModel(obs_dim=2048, latent_dim=256)
video = torch.randn(2, 30, 2048)
out = model(video)
print(f'Gloss logits: {out[\"gloss_logits\"].shape}')
print(f'Tokens: {out[\"tokens\"].shape}')
print(f'Latent: {out[\"latent\"].shape}')
"
```

### Configuration

Edit `configs/default.yaml` to customize model and training parameters:

```yaml
model:
  obs_dim: 2048          # I3D visual feature dimension
  latent_dim: 256        # Latent state dimension
  hidden_dim: 512        # Hidden layer dimension
  num_heads: 8           # Attention heads
  num_layers: 8          # Transformer layers
  num_codes: 8192        # VQ-VAE codebook size
  num_glosses: 100       # Sign language gloss classes

training:
  epochs: 100
  batch_size: 16
  lr: 1.0e-4
  beta_kl: 0.1           # KL divergence weight
  commitment_cost: 0.25   # VQ-VAE commitment penalty
  device: "cuda"          # or "cpu"
```

---

## Project Structure

```
sign-lang-worldmodel/
├── src/
│   └── signlang/
│       └── world_model/
│           └── signer_world_model.py    # Core model implementation
│                                        # - ContinuousWorldModel
│                                        # - VQVAETokenizer
│                                        # - TemporalTransformer
│                                        # - SignLanguageWorldModel
├── configs/
│   └── default.yaml                     # Model & training configuration
├── main.py                              # Training entry point
├── scripts/                             # Utility scripts
├── paper/                               # Research paper materials
├── tests/
│   └── test_smoke.py                    # Smoke tests
├── requirements.txt                     # Python dependencies
├── start.sh                             # Startup script
├── REPRODUCE.md                         # Reproduction guide
└── README.md
```

---

## Benchmarks

| Metric | Value |
|--------|-------|
| Model Parameters | ~15M (configurable) |
| Latent Dimension | 256 |
| VQ-VAE Codebook Size | 8192 codes |
| Temporal Transformer Layers | 8 |
| Attention Heads | 8 |
| Input Feature Dimension | 2048 (I3D) |
| Sequence Length | 30 frames (default) |
| Gloss Classes | 100 (configurable) |
| KL Free Bits | 0.5 * latent_dim |
| Posterior Blend Weight | 0.7 (dynamics vs observation) |
| Default Learning Rate | 1e-4 |
| Default Batch Size | 16 |

**IMPORTANT DISCLAIMER:** This is a research prototype. The current implementation uses **synthetic random noise** for validation only -- all training data is randomly generated, not real sign language features. Any reported training metrics (loss, accuracy) are meaningless and should NOT be interpreted as model performance. The model has NOT been evaluated on any real sign language dataset (AUTSL, WLASL, Phoenix). Performance on real data requires integration with actual video feature extraction pipelines and is currently unknown.

---

## Research

### Core Innovation: Continuous Latent World Models for Sign Language

The key contribution is applying world model methodology -- originally developed for reinforcement learning (Ha & Schmidhuber, 2018; Hafner et al., 2023) -- to sign language translation.

**Why World Models for Sign Language?**

Sign language is inherently a dynamical system:
- Signs evolve continuously in time (not discrete tokens)
- Hand trajectories follow physically constrained paths
- Self-occlusion creates missing data that must be inferred
- Grammatical information is encoded in motion dynamics, not static poses

A continuous latent world model naturally handles all four properties:
- **Continuous state**: The Gaussian latent `z_t` captures smooth temporal evolution
- **Dynamics prior**: The learned transition model `p(z_t | z_{<t})` encodes physical constraints
- **Uncertainty**: The variance of `z_t` quantifies confidence, enabling robust inference under occlusion
- **Prediction**: The model can "imagine" future states, supporting temporal reasoning

**Architecture Design Choices:**

1. **Beta-VAE weighting** (beta=0.1): Prevents the KL term from dominating, allowing the latent space to capture useful structure rather than collapsing to the prior.
2. **Free bits** (0.5 * latent_dim): Ensures minimum information flow through the latent, preventing posterior collapse.
3. **Posterior blending** (0.7:0.3): When predicting the next state, the model blends 70% posterior (observation-informed) with 30% dynamics (prediction-only), balancing responsiveness with stability.
4. **VQ-VAE bridge**: The discrete tokenizer enables integration with autoregressive LLM decoders for text generation.

### References

**World Models and Latent Dynamics:**
- Ha, D. & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122*.
- Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104* (DreamerV3).
- Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination (Dreamer). *ICLR 2020*.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature* (MuZero).

**Variational Autoencoders and VQ-VAE:**
- Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *arXiv:1312.6114*.
- van den Oord, A., et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *arXiv:1711.00937*.
- Razavi, A., et al. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. *NeurIPS 2019*.

**Sign Language Recognition and Translation:**
- Hu, H., et al. (2023). SignBERT+: Pre-Training of Hand-Object Transformer for Sign Language Recognition. *IEEE TPAMI*.
- Camgoz, N. C., et al. (2020). Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation. *CVPR 2020*.
- Li, D., et al. (2020). Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. *WACV 2020*. (WLASL dataset)
- Sincan, O. M. & Keles, H. Y. (2020). AUTSL: A Large-Scale Multi-Modal Turkish Sign Language Dataset and Baseline Methods. *IEEE Access*. (AUTSL dataset)
- Ko, S. K., et al. (2019). A Survey on Sign Language Recognition, Translation, and Generation. *IEEE Access*.

---

## Roadmap

- [ ] Integration with real sign language datasets (AUTSL, WLASL, Phoenix)
- [ ] I3D / SlowFast visual feature extraction pipeline
- [ ] Diffusion-based decoder for frame generation
- [ ] LLM integration for grammar-aware sign-to-text translation
- [ ] Fingerspelling detection and handling
- [ ] Multi-sign-language support (ASL, BSL, CSL, Turkish Sign Language)
- [ ] Real-time inference optimization (ONNX export, TensorRT)
- [ ] Evaluation benchmarks (BLEU, ROUGE, WER)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/sign-lang-worldmodel/issues)
- Discussions: Open an issue with the `discussion` tag

---

*Bridging continuous perception and discrete language through world models.*
