"""Tests for model components."""

import pytest
import torch
import torch.nn as nn

from src.signlang.world_model.signer_world_model import (
    SinusoidalPosEmb,
    ResidualBlock,
    TemporalTransformer,
    ContinuousWorldModel,
    VQVAETokenizer,
    SignLanguageWorldModel,
)


class TestSinusoidalPosEmb:
    """Tests for SinusoidalPosEmb module."""

    def test_output_shape(self):
        """Test output shape is correct."""
        dim = 64
        batch_size = 4
        seq_len = 10

        model = SinusoidalPosEmb(dim)
        t = torch.arange(seq_len).float()

        output = model(t)

        assert output.shape == (seq_len, dim)

    def test_different_dimensions(self):
        """Test with different embedding dimensions."""
        for dim in [32, 64, 128, 256]:
            model = SinusoidalPosEmb(dim)
            t = torch.tensor([0.0, 1.0, 2.0])

            output = model(t)

            assert output.shape == (3, dim)

    def test_gradient_flow(self):
        """Test that gradients can flow through the module."""
        dim = 64
        model = SinusoidalPosEmb(dim)
        t = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)

        output = model(t)
        loss = output.sum()
        loss.backward()

        assert t.grad is not None


class TestResidualBlock:
    """Tests for ResidualBlock module."""

    def test_output_shape(self):
        """Test output shape matches input shape."""
        dim = 128
        batch_size = 4
        seq_len = 10

        model = ResidualBlock(dim)
        x = torch.randn(batch_size, seq_len, dim)

        output = model(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection is applied."""
        dim = 64
        model = ResidualBlock(dim, dropout=0.0)
        x = torch.randn(2, 5, dim)

        # Set all weights to zero to test residual
        for param in model.parameters():
            nn.init.zeros_(param)

        output = model(x)

        # With zero weights, output should be close to input (residual)
        assert torch.allclose(output, x, atol=1e-6)

    def test_different_input_sizes(self):
        """Test with different batch and sequence sizes."""
        dim = 128
        model = ResidualBlock(dim)

        for batch_size in [1, 2, 4, 8]:
            for seq_len in [1, 5, 10, 20]:
                x = torch.randn(batch_size, seq_len, dim)
                output = model(x)
                assert output.shape == (batch_size, seq_len, dim)


class TestTemporalTransformer:
    """Tests for TemporalTransformer module."""

    def test_output_shape(self):
        """Test output shape is correct."""
        dim = 128
        num_heads = 8
        num_layers = 4
        batch_size = 2
        seq_len = 10

        model = TemporalTransformer(dim, num_heads, num_layers)
        x = torch.randn(batch_size, seq_len, dim)

        output = model(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_causal_masking(self):
        """Test that causal masking is applied (output depends on position)."""
        dim = 64
        num_heads = 4
        num_layers = 2
        batch_size = 1
        seq_len = 5

        model = TemporalTransformer(dim, num_heads, num_layers, dropout=0.0)
        model.eval()

        x = torch.randn(batch_size, seq_len, dim)

        # Get output for all positions
        with torch.no_grad():
            output_all = model(x)

        # Get output for first position only
        with torch.no_grad():
            output_first = model(x[:, :1, :])

        # First position output should be the same
        assert torch.allclose(
            output_all[:, 0, :], output_first[:, 0, :], atol=1e-5
        )

    def test_with_time_embeddings(self):
        """Test with explicit time embeddings."""
        dim = 64
        num_heads = 4
        num_layers = 2
        batch_size = 2
        seq_len = 8

        model = TemporalTransformer(dim, num_heads, num_layers)
        x = torch.randn(batch_size, seq_len, dim)
        t = torch.arange(seq_len).float()

        output = model(x, t)

        assert output.shape == (batch_size, seq_len, dim)


class TestContinuousWorldModel:
    """Tests for ContinuousWorldModel."""

    def test_encode(self, model_config):
        """Test encoding observations to latent distribution."""
        obs_dim = model_config["obs_dim"]
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(obs_dim=obs_dim, latent_dim=latent_dim)

        batch_size = 4
        seq_len = 10
        obs = torch.randn(batch_size, seq_len, obs_dim)

        mean, logvar = model.encode(obs)

        assert mean.shape == (batch_size, seq_len, latent_dim)
        assert logvar.shape == (batch_size, seq_len, latent_dim)

    def test_reparameterize(self, model_config):
        """Test reparameterization trick."""
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(latent_dim=latent_dim)

        batch_size = 4
        seq_len = 10

        mean = torch.randn(batch_size, seq_len, latent_dim)
        logvar = torch.randn(batch_size, seq_len, latent_dim)

        z = model.reparameterize(mean, logvar)

        assert z.shape == (batch_size, seq_len, latent_dim)

        # Test that sampling is stochastic
        z1 = model.reparameterize(mean, logvar)
        z2 = model.reparameterize(mean, logvar)

        # Should be different (with very high probability)
        assert not torch.allclose(z1, z2)

    def test_forward(self, model_config, sample_video_features):
        """Test full forward pass."""
        obs_dim = model_config["obs_dim"]
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(obs_dim=obs_dim, latent_dim=latent_dim)

        output = model(sample_video_features)

        assert "z_mean" in output
        assert "z_logvar" in output
        assert "z" in output
        assert "z_prior_mean" in output
        assert "z_prior_logvar" in output
        assert "obs_pred" in output
        assert "gloss_logits" in output

        batch_size, seq_len, _ = sample_video_features.shape

        assert output["z_mean"].shape == (batch_size, seq_len, latent_dim)
        assert output["z_logvar"].shape == (batch_size, seq_len, latent_dim)
        assert output["z"].shape == (batch_size, seq_len, latent_dim)
        assert output["z_prior_mean"].shape == (batch_size, seq_len, latent_dim)
        assert output["z_prior_logvar"].shape == (batch_size, seq_len, latent_dim)
        assert output["obs_pred"].shape == (batch_size, seq_len, obs_dim)

    def test_world_loss(self, model_config, sample_video_features):
        """Test world model loss computation."""
        obs_dim = model_config["obs_dim"]
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(obs_dim=obs_dim, latent_dim=latent_dim)

        output = model(sample_video_features)

        loss, loss_dict = model.world_loss(
            sample_video_features,
            output["z"],
            output["obs_pred"],
            output["z_mean"],
            output["z_logvar"],
            output["z_prior_mean"],
            output["z_prior_logvar"],
        )

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_predict_next(self, model_config):
        """Test next state prediction."""
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(latent_dim=latent_dim)

        batch_size = 4
        z_prev = torch.randn(batch_size, latent_dim)

        z_next = model.predict_next(z_prev)

        assert z_next.shape == (batch_size, latent_dim)

    def test_predict_next_with_observation(self, model_config):
        """Test next state prediction with observation."""
        obs_dim = model_config["obs_dim"]
        latent_dim = model_config["latent_dim"]

        model = ContinuousWorldModel(obs_dim=obs_dim, latent_dim=latent_dim)

        batch_size = 4
        z_prev = torch.randn(batch_size, latent_dim)
        obs_current = torch.randn(batch_size, obs_dim)

        z_next = model.predict_next(z_prev, obs_current)

        assert z_next.shape == (batch_size, latent_dim)


class TestVQVAETokenizer:
    """Tests for VQVAETokenizer."""

    def test_forward(self):
        """Test forward pass."""
        latent_dim = 256
        num_codes = 8192
        batch_size = 4
        seq_len = 10

        model = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        z = torch.randn(batch_size, seq_len, latent_dim)

        z_q, tokens, commitment_loss = model(z)

        assert z_q.shape == (batch_size, seq_len, latent_dim)
        assert tokens.shape == (batch_size, seq_len)
        assert commitment_loss.dim() == 0  # Scalar

        # Tokens should be valid indices
        assert (tokens >= 0).all()
        assert (tokens < num_codes).all()

    def test_decode_tokens(self):
        """Test token decoding."""
        latent_dim = 256
        num_codes = 8192
        batch_size = 4
        seq_len = 10

        model = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        tokens = torch.randint(0, num_codes, (batch_size, seq_len))

        decoded = model.decode_tokens(tokens)

        assert decoded.shape == (batch_size, seq_len, latent_dim)

    def test_straight_through_estimator(self):
        """Test that straight-through estimator works."""
        latent_dim = 64
        num_codes = 100

        model = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        z = torch.randn(2, 5, latent_dim, requires_grad=True)

        z_q, tokens, commitment_loss = model(z)

        # Check that gradients can flow
        loss = z_q.sum()
        loss.backward()

        assert z.grad is not None

    def test_codebook_utilization(self):
        """Test that codebook entries are being used."""
        latent_dim = 64
        num_codes = 10  # Small codebook for testing

        model = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        # Run multiple forward passes
        all_tokens = []
        for _ in range(10):
            z = torch.randn(4, 10, latent_dim)
            _, tokens, _ = model(z)
            all_tokens.append(tokens.flatten())

        all_tokens = torch.cat(all_tokens)

        # Should use multiple codebook entries
        unique_tokens = torch.unique(all_tokens)
        assert len(unique_tokens) > 1


class TestSignLanguageWorldModel:
    """Tests for SignLanguageWorldModel."""

    def test_forward(self, model_config, sample_video_features):
        """Test full forward pass."""
        model = SignLanguageWorldModel(**model_config)

        output = model(sample_video_features)

        assert "gloss_logits" in output
        assert "tokens" in output
        assert "latent" in output
        assert "commitment_loss" in output

        batch_size, seq_len, _ = sample_video_features.shape
        num_glosses = model_config["num_glosses"]

        assert output["gloss_logits"].shape == (batch_size, seq_len, num_glosses)
        assert output["tokens"].shape == (batch_size, seq_len)

    def test_generate_trajectory(self, model_config, sample_video_features):
        """Test trajectory generation."""
        model = SignLanguageWorldModel(**model_config)

        # Get initial latent
        output = model(sample_video_features)
        z_init = output["latent"][:, 0, :]

        # Generate trajectory
        num_steps = 10
        trajectory = model.generate_trajectory(z_init, num_steps)

        batch_size = sample_video_features.shape[0]
        latent_dim = model_config["latent_dim"]

        assert trajectory.shape == (batch_size, num_steps + 1, latent_dim)

    def test_model_parameters(self, model_config):
        """Test model parameter count."""
        model = SignLanguageWorldModel(**model_config)

        num_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert num_params > 0
        assert num_params < 1e9  # Less than 1B parameters

    def test_model_eval_mode(self, model_config, sample_video_features):
        """Test model in eval mode."""
        model = SignLanguageWorldModel(**model_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_video_features)

        assert "gloss_logits" in output

    def test_model_train_mode(self, model_config, sample_video_features):
        """Test model in train mode."""
        model = SignLanguageWorldModel(**model_config)
        model.train()

        output = model(sample_video_features)

        assert "gloss_logits" in output

        # Compute full loss (gloss + world model) so all submodules get gradients
        wm_loss, _ = model.world_model.world_loss(
            sample_video_features,
            output["latent"],
            output["world_loss"],
            output["z_mean"],
            output["z_logvar"],
            output["z_prior_mean"],
            output["z_prior_logvar"],
        )
        gloss_loss = output["gloss_logits"][:, -1, :].sum()
        loss = wm_loss + gloss_loss + output["commitment_loss"]
        loss.backward()

        # Check that encoder/dynamics/posterior/decoder/sign_head params get gradients
        # Skip: codebook (updated via EMA), llm_translator (placeholder, not in loss)
        checked = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(skip in name for skip in ("ema_", "steps_since", "codebook", "llm_translator")):
                    continue  # Updated manually via EMA or placeholder
                assert param.grad is not None, f"No gradient for {name}"
                checked += 1
        assert checked > 0