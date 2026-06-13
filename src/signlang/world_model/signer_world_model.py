"""
World Model for Sign Language Translation
========================================
Implements: Continuous Latent World Model (CLWM) for sign language.

DISCLAIMER: This is a research prototype validated on synthetic random data
only. It has NOT been evaluated on any real sign language dataset. Claims
about architecture capabilities (occlusion handling, physical prediction)
are theoretical and unverified.

Core idea:
  The model doesn't just classify signs frame-by-frame.
  It maintains a continuous latent state z_t that represents
  the evolving "world state" of the signing scene.

  State transition: p(z_t | z_{<t}, x_{<t}) = N(mu_theta(z_{<t}), Sigma_theta(z_{<t}))

  This enables (theoretically, not yet validated):
  - Physical prediction: model can "imagine" next frames
  - Handling occlusion: z_t is denoised, robust to missing data
  - Temporal coherence: smooth transitions between signs

Architecture:
  1. Visual Encoder: Linear projection (placeholder for I3D/SlowFast)
  2. Tokenizer: VQ-VAE with EMA codebook updates
  3. World Model: Continuous latent dynamics with Temporal Transformer
  4. LLM Translator: Placeholder (NOT functional)

Published reference basis:
  - Ha World Models (2018) arXiv:1803.10122
  - DreamerV3 (2023) - latent world models for reinforcement learning
  - SignBERT+ (2024) - pre-trained sign language model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for time/timestep encoding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for world state sequence modeling.
    
    Uses causal masking (can only see past) and relative positional encoding.
    """
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.pos_emb = SinusoidalPosEmb(dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Time embeddings
        if t is None:
            t = torch.arange(T, device=x.device)
        t_emb = self.pos_emb(t)  # [T, D]
        x = x + t_emb.unsqueeze(0)  # broadcast
        
        # Causal self-attention
        x = self.norm(x)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + h
        
        # Feed-forward blocks
        for block in self.blocks:
            x = block(x)
        
        return x  # [B, T, D]


class ContinuousWorldModel(nn.Module):
    """
    Continuous Latent World Model (CLWM) for sign language.
    
    Key innovation: state is continuous Gaussian, not discrete tokens.
    This gives the model uncertainty estimation (crucial for occlusion).
    
    Components:
      - encoder: observation → latent mean/logvar
      - dynamics: world state transition (the physics model)
      - predictor: world state → next observation
      - decoder: world state → output (sign translation)
    """
    
    def __init__(
        self,
        obs_dim: int = 512,
        latent_dim: int = 256,
        action_dim: int = 0,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_glosses: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Observation encoder: maps video frames to latent
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout),
        )
        
        # World state: continuous latent (mean + logvar)
        self.state_mean = nn.Linear(hidden_dim, latent_dim)
        self.state_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Prior dynamics: p(z_t | z_{t-1}) - learned physics model
        self.dynamics = TemporalTransformer(
            dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Dynamics prior distribution parameters
        self.dynamics_mean = nn.Linear(latent_dim, latent_dim)
        self.dynamics_logvar = nn.Linear(latent_dim, latent_dim)
        
        # Posterior encoder: q(z_t | z_{t-1}, x_t) - observation update
        self.posterior = TemporalTransformer(
            dim=latent_dim + obs_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
        )
        
        # Output: z_t → observation prediction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(3)],
            nn.Linear(hidden_dim, obs_dim),
        )
        
        # For sign language: z_t -> gloss/text
        self.num_glosses = num_glosses
        self.sign_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_glosses),
        )
    
    def encode(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode observation to latent distribution parameters."""
        h = self.obs_encoder(obs)
        mean = self.state_mean(h)
        logvar = torch.clamp(self.state_logvar(h), -10, 10)
        return mean, logvar
    
    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Sample from the latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, obs_seq: Tensor, latent_prior: Optional[Tensor] = None) -> dict:
        """
        Full forward pass through the world model.
        
        Args:
            obs_seq: [B, T, obs_dim] - sequence of observations
            latent_prior: [B, T, latent_dim] - prior latent (optional, for generation)
        
        Returns:
            dict with:
                z_mean, z_logvar, z (latent samples)
                obs_pred (predicted observations)
                gloss_logits
        """
        B, T, _ = obs_seq.shape

        # Encode each observation
        z_mean, z_logvar = self.encode(obs_seq)  # [B, T, L]

        # Prior dynamics: what does the model expect?
        z_dynamics = self.dynamics(z_mean)  # [B, T, L]
        z_prior_mean = self.dynamics_mean(z_dynamics)     # [B, T, L]
        z_prior_logvar = torch.clamp(
            self.dynamics_logvar(z_dynamics), -10, 10
        )  # [B, T, L]

        # Posterior: combine dynamics prior with observation
        posterior_input = torch.cat([z_dynamics, obs_seq], dim=-1)  # [B, T, L+obs]
        z_post = self.posterior(posterior_input)[:, :, :self.latent_dim]  # [B, T, L]

        # Blend posterior (observation-informed) with encoder output
        # This ensures the posterior module contributes to the computation graph
        z_refined = 0.7 * z_post + 0.3 * z_mean  # [B, T, L]

        # Sample from refined posterior (for training)
        z = self.reparameterize(z_refined, z_logvar)  # [B, T, L]

        # Predict observations from latent
        obs_pred = self.decoder(z)  # [B, T, obs_dim]

        # Gloss prediction (for sign language)
        gloss_logits = self.sign_head(z)  # [B, T, num_glosses]

        return {
            "z_mean": z_refined,
            "z_logvar": z_logvar,
            "z": z,
            "z_prior_mean": z_prior_mean,
            "z_prior_logvar": z_prior_logvar,
            "obs_pred": obs_pred,
            "gloss_logits": gloss_logits,
        }
    
    def world_loss(
        self,
        obs_seq: Tensor,
        latent: Tensor,
        obs_pred: Tensor,
        z_mean: Tensor,
        z_logvar: Tensor,
        z_prior_mean: Tensor,
        z_prior_logvar: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        World model loss = reconstruction + KL divergence.

        L_total = L_recon + beta * L_KL

        Where:
          L_recon = ||obs_seq - obs_pred||^2
          L_KL = KL(q(z|x) || p(z|z_{<t}))
               = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q - mu_p)^2)/var_p - 1)

        The KL divergence is between the posterior q(z|x) and the
        dynamics prior p(z|z_{<t}), not a standard normal. This ensures
        the latent space is shaped by the learned dynamics model.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(obs_pred, obs_seq, reduction="sum")

        # KL divergence: KL(q(z|x) || p(z|z_{<t}))
        # q ~ N(z_mean, exp(z_logvar)), p ~ N(z_prior_mean, exp(z_prior_logvar))
        kl_loss = 0.5 * torch.sum(
            z_prior_logvar - z_logvar
            + z_logvar.exp() / z_prior_logvar.exp()
            + (z_mean - z_prior_mean).pow(2) / z_prior_logvar.exp()
            - 1.0
        )

        # Free bits to prevent KL collapse (Kingma & Welling 2015)
        free_bits = 0.5 * self.latent_dim
        kl_loss = torch.clamp_min(kl_loss, free_bits)

        beta = 0.1  # KL weight (beta-VAE style)
        total_loss = recon_loss + beta * kl_loss

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    def predict_next(self, z_prev: Tensor, obs_current: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next world state (imagination step).

        This is the key to the world model's generative capability:
        given the current world state, predict what happens next.
        """
        # Dynamics prediction
        z_raw = self.dynamics(z_prev.unsqueeze(1)).squeeze(1)  # [B, L]
        z_dynamics = self.dynamics_mean(z_raw)  # [B, L]

        # If we have an observation, update posterior
        if obs_current is not None:
            # Posterior expects [z_prior, obs] with dim = latent_dim + obs_dim
            combined = torch.cat([z_raw, obs_current], dim=-1)
            z_post = self.posterior(combined.unsqueeze(1)).squeeze(1)[:, :self.latent_dim]
            z_dynamics = 0.7 * z_post + 0.3 * z_dynamics  # blend

        return z_dynamics


class VQVAETokenizer(nn.Module):
    """
    VQ-VAE Tokenizer: discretizes continuous latent into discrete tokens.

    This bridges the continuous world model with autoregressive LLM translation.

    Uses exponential moving average (EMA) for codebook updates,
    which is more stable than naive vector quantization (van den Oord et al., 2017).
    Dead codes (unused for `dead_code_threshold` steps) are re-initialized
    to random encoder outputs to maintain codebook utilization.

    NOTE: EMA updates are applied during training only (model.training == True).
    During inference, the codebook is frozen.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_codes: int = 8192,
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.commitment = commitment
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook: learnable discrete latent space
        self.codebook = nn.Parameter(
            torch.randn(num_codes, latent_dim) * 0.02
        )

        # EMA tracking buffers (not parameters -- updated manually)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding_sum", torch.zeros(num_codes, latent_dim))
        self.register_buffer("steps_since_used", torch.zeros(num_codes))

        # Pre-quantizer projection
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize continuous latent to discrete code.

        Returns:
            z_q: quantized latent [B, T, D]
            tokens: code indices [B, T]
            commitment_loss: codebook commitment penalty
        """
        # Project to latent space
        z_e = self.proj(z)
        z_e = self.norm(z_e)

        # Compute distances to codebook
        B, T, D = z_e.shape
        z_e_flat = z_e.view(-1, D)  # [B*T, D]

        d = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook ** 2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.codebook.t())
        )  # [B*T, num_codes]

        # Nearest codebook entry
        indices = torch.argmin(d, dim=1)  # [B*T]
        tokens = indices.view(B, T)

        # Quantized representation
        z_q = self.codebook[indices].view(B, T, D)

        # --- EMA codebook update (training only) ---
        if self.training:
            self._ema_update(z_e_flat, indices)

        # Commitment loss: encourage encoder output to stay close to codebook.
        # Detach z_q so gradients flow to the encoder (z_e), not the codebook.
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Use straight-through estimator: forward=z_q, backward=gradient from z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, tokens, commitment_loss

    @torch.no_grad()
    def _ema_update(self, z_e_flat: Tensor, indices: Tensor) -> None:
        """
        Apply exponential moving average update to the codebook.

        This tracks the running average of encoder outputs assigned to each
        code, providing a smoother and more stable codebook update than
        gradient-based optimization alone (van den Oord et al., 2017).

        Dead codes (not selected for `dead_code_threshold` consecutive steps)
        are re-initialized to random encoder outputs.
        """
        # One-hot encoding of assignments
        encodings = F.one_hot(indices, self.num_codes).float()  # [B*T, K]

        # Count how many vectors were assigned to each code
        cluster_size = encodings.sum(dim=0)  # [K]

        # Sum of encoder outputs assigned to each code
        embedding_sum = encodings.t() @ z_e_flat  # [K, D]

        # Update EMA running averages
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
        )

        # Update codebook to the EMA centroid
        updated_codebook = self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
        self.codebook.copy_(updated_codebook)

        # Track dead codes: increment counters, reset used codes
        self.steps_since_used.add_(1)
        self.steps_since_used.scatter_(0, indices, torch.zeros_like(indices, dtype=torch.float32))

        # Re-initialize dead codes
        dead_mask = self.steps_since_used > self.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead > 0 and z_e_flat.size(0) > 0:
            # Pick random encoder outputs to replace dead codes
            random_indices = torch.randint(0, z_e_flat.size(0), (num_dead,), device=z_e_flat.device)
            self.codebook.data[dead_mask] = z_e_flat[random_indices]
            self.ema_embedding_sum[dead_mask] = z_e_flat[random_indices]
            self.ema_cluster_size[dead_mask] = 1.0
            self.steps_since_used[dead_mask] = 0

    def decode_tokens(self, tokens: Tensor) -> Tensor:
        """Convert discrete tokens back to continuous latent."""
        return self.codebook[tokens]  # [B, T, D]


class LLMTranslatorPlaceholder(nn.Module):
    """
    Placeholder for LLM-based sign-to-text translation.

    DISCLAIMER: This module is NOT implemented. The architecture paper and
    README describe a pipeline where discrete VQ-VAE tokens are fed to a
    large language model for grammar-aware sign-to-text translation. This
    placeholder exists to make the architecture gap explicit. A real
    implementation would integrate a pre-trained LLM (e.g., LLaMA, GPT-2)
    fine-tuned on sign language gloss-to-text pairs.

    Current status: NOT FUNCTIONAL. Gloss classification is performed by
    the world model's sign_head instead.
    """

    def __init__(self, num_codes: int = 8192, vocab_size: int = 32000, hidden_dim: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(num_codes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        # Initialize to near-zero so outputs are roughly uniform (no-op behavior)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Translate discrete sign tokens to text logits.

        Args:
            tokens: [B, T] discrete token indices from VQ-VAE

        Returns:
            text_logits: [B, T, vocab_size] -- NOTE: outputs are near-uniform
                because this is a non-functional placeholder.
        """
        h = self.token_embedding(tokens)
        return self.output_proj(h)


class SignLanguageWorldModel(nn.Module):
    """
    Full sign language translation system using World Model approach.

    Pipeline:
      RGB-D Video -> I3D Encoder -> VQ-VAE Tokenizer -> World Model -> LLM Translator -> Text

    The World Model handles:
      - Temporal modeling (smooth sign transitions)
      - Handling of self-occlusion (hands covering face)
      - Physical plausibility (reasonable hand trajectories)

    DISCLAIMER: The LLM translator is a non-functional placeholder.
    Current output uses gloss classification via the world model's sign_head,
    not actual text translation. See LLMTranslatorPlaceholder for details.
    """

    def __init__(
        self,
        obs_dim: int = 2048,   # I3D feature dim
        latent_dim: int = 256,
        feature_dim: int = None,  # raw input feature dimension (e.g., I3D). Defaults to obs_dim.
        num_codes: int = 8192,
        num_glosses: int = 100,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        if feature_dim is None:
            feature_dim = obs_dim
        self.feature_dim = feature_dim

        # Continuous world model
        self.world_model = ContinuousWorldModel(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_glosses=num_glosses,
        )

        # VQ-VAE tokenizer
        self.tokenizer = VQVAETokenizer(
            latent_dim=latent_dim,
            num_codes=num_codes,
        )

        # LLM translator (placeholder -- not functional)
        self.llm_translator = LLMTranslatorPlaceholder(
            num_codes=num_codes,
            hidden_dim=hidden_dim,
        )

        # Visual encoder: projects raw features to world model obs_dim
        if self.feature_dim != obs_dim:
            self.visual_encoder = nn.Linear(feature_dim, obs_dim)
        else:
            self.visual_encoder = nn.Identity()
    
    def forward(self, video_features: Tensor) -> dict:
        """
        video_features: [B, T, 2048] - pre-extracted I3D features
        """
        # Project I3D features to world model observation dimension
        obs = self.visual_encoder(video_features)  # [B, T, obs_dim]

        # Encode to world model latent space
        wm_out = self.world_model(obs)

        # Tokenize
        z_q, tokens, commit_loss = self.tokenizer(wm_out["z"])

        # LLM translation (placeholder -- outputs near-uniform logits)
        text_logits = self.llm_translator(tokens)

        return {
            "gloss_logits": wm_out["gloss_logits"],
            "tokens": tokens,
            "latent": z_q,
            "commitment_loss": commit_loss,
            "world_loss": wm_out.get("obs_pred"),
            "z_mean": wm_out["z_mean"],
            "z_logvar": wm_out["z_logvar"],
            "z_prior_mean": wm_out["z_prior_mean"],
            "z_prior_logvar": wm_out["z_prior_logvar"],
            "text_logits": text_logits,  # placeholder -- not functional
        }
    
    def generate_trajectory(
        self,
        z_init: Tensor,
        num_steps: int = 30,
    ) -> Tensor:
        """
        Generate future trajectories (imagination).
        
        Given an initial world state, the model predicts
        how the signing will evolve.
        """
        generated = [z_init]
        z = z_init
        
        for _ in range(num_steps):
            z = self.world_model.predict_next(z)
            generated.append(z)
        
        return torch.stack(generated, dim=1)  # [B, T, L]


if __name__ == "__main__":
    # Smoke test
    model = SignLanguageWorldModel(obs_dim=2048, latent_dim=256)
    
    # Fake I3D features: [B, T, 2048]
    video = torch.randn(2, 30, 2048)
    
    out = model(video)
    print(f"Gloss logits: {out['gloss_logits'].shape}")       # [2, 30, 100]
    print(f"Tokens: {out['tokens'].shape}")                   # [2, 30]
    print(f"Latent: {out['latent'].shape}")                   # [2, 30, 256]
    
    # Test generation
    z0 = out["latent"][:, 0, :]  # first frame latent
    traj = model.generate_trajectory(z0, num_steps=10)
    print(f"Generated trajectory: {traj.shape}")  # [2, 11, 256]
    
    print("\nWorld Model test passed.")
