# REPRODUCE.md - sign-lang-worldmodel

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Recommended for training
- **PyTorch**: 2.0+

## Install

```bash
cd sign-lang-worldmodel
pip install -r requirements.txt
```

Dependencies: torch, numpy, pyyaml

## Run

```bash
python main.py
```

## Architecture

```
RGB-D Video -> I3D Encoder -> VQ-VAE Tokenizer -> World Model -> LLM Translator -> Text
                      |
            Temporal Transformer (latent dynamics)
```

Key innovation: Continuous latent world model for sign language (not discrete tokens)
- State transition: p(z_t | z_{<t}, x_{<t}) = N(mu_theta, Sigma_theta)

## Known Issues

- No dataset included (requires sign language video data)
- No test suite
- Minimal dependencies (lightweight)
- Research prototype stage
