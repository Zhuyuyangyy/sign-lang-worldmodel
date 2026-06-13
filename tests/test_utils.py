"""Tests for utility functions and configuration."""

import pytest
import torch
import yaml
from pathlib import Path


class TestModelConfigurations:
    """Tests for different model configurations."""

    def test_default_config(self):
        """Test default model configuration."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel()

        # Check default parameters
        assert model.world_model.obs_dim == 2048
        assert model.world_model.latent_dim == 256
        assert model.tokenizer.num_codes == 8192

    def test_custom_config(self):
        """Test custom model configuration."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(
            obs_dim=1024,
            latent_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=4,
            num_codes=1024,
            num_glosses=50,
        )

        assert model.world_model.obs_dim == 1024
        assert model.world_model.latent_dim == 128
        assert model.tokenizer.num_codes == 1024

    def test_small_config(self):
        """Test very small model configuration."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(
            obs_dim=32,
            latent_dim=16,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            num_codes=64,
            num_glosses=10,
        )

        # Should work without errors
        x = torch.randn(1, 5, 32)
        output = model(x)

        assert output["gloss_logits"].shape == (1, 5, 10)

    def test_large_config(self):
        """Test large model configuration."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(
            obs_dim=2048,
            latent_dim=512,
            hidden_dim=1024,
            num_heads=16,
            num_layers=12,
            num_codes=16384,
            num_glosses=1000,
        )

        # Check parameter count is reasonable
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 1e6  # At least 1M parameters


class TestTensorOperations:
    """Tests for tensor operations used in the model."""

    def test_reparameterization(self):
        """Test reparameterization trick."""
        from src.signlang.world_model.signer_world_model import ContinuousWorldModel

        model = ContinuousWorldModel(latent_dim=64)

        mean = torch.zeros(2, 10, 64)
        logvar = torch.zeros(2, 10, 64)

        # With zero logvar, samples should be close to mean
        z = model.reparameterize(mean, logvar)

        assert z.shape == (2, 10, 64)
        # Samples should be centered around mean
        assert torch.abs(z.mean()) < 0.5

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        from src.signlang.world_model.signer_world_model import ContinuousWorldModel

        model = ContinuousWorldModel(latent_dim=64)

        # Create known distribution
        mean = torch.zeros(1, 1, 64)
        logvar = torch.zeros(1, 1, 64)

        # KL should be 0 when prior and posterior match, but free_bits
        # clamps the minimum to 0.5 * latent_dim = 32
        _, loss_dict = model.world_loss(
            torch.randn(1, 1, 64),  # obs_seq
            torch.randn(1, 1, 64),  # latent
            torch.randn(1, 1, 64),  # obs_pred
            mean,
            logvar,
            mean,     # z_prior_mean (same as posterior mean)
            logvar,   # z_prior_logvar (same as posterior logvar)
        )

        # KL should equal free_bits (32) when distributions match
        expected_free_bits = 0.5 * 64
        assert abs(loss_dict["kl_loss"] - expected_free_bits) < 1.0

    def test_mse_loss(self):
        """Test MSE loss computation."""
        x = torch.randn(2, 10, 64)
        y = x.clone()

        # MSE should be 0 for identical tensors
        mse = torch.nn.functional.mse_loss(x, y)
        assert mse.item() < 1e-6

        # MSE should be positive for different tensors
        y = x + 1.0
        mse = torch.nn.functional.mse_loss(x, y)
        assert mse.item() > 0


class TestCodebookOperations:
    """Tests for VQ-VAE codebook operations."""

    def test_codebook_lookup(self):
        """Test codebook lookup."""
        from src.signlang.world_model.signer_world_model import VQVAETokenizer

        latent_dim = 64
        num_codes = 100

        tokenizer = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        # Test decode_tokens
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
        decoded = tokenizer.decode_tokens(tokens)

        assert decoded.shape == (2, 3, 64)

        # Same token should give same embedding
        assert torch.equal(decoded[0, 0, :], decoded[0, 0, :])

    def test_codebook_initialization(self):
        """Test codebook initialization."""
        from src.signlang.world_model.signer_world_model import VQVAETokenizer

        latent_dim = 64
        num_codes = 100

        tokenizer = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        # Codebook should be initialized with small values
        assert tokenizer.codebook.abs().mean() < 1.0

        # Codebook shape should be correct
        assert tokenizer.codebook.shape == (num_codes, latent_dim)

    def test_quantization_error(self):
        """Test that quantization introduces some error."""
        from src.signlang.world_model.signer_world_model import VQVAETokenizer

        latent_dim = 64
        num_codes = 10  # Small codebook for testing

        tokenizer = VQVAETokenizer(latent_dim=latent_dim, num_codes=num_codes)

        # Create input far from codebook
        z = torch.randn(1, 1, latent_dim) * 10

        z_q, tokens, _ = tokenizer(z)

        # Quantized should be different from original
        error = (z - z_q).abs().mean()
        assert error.item() > 0


class TestTemporalOperations:
    """Tests for temporal operations."""

    def test_causal_mask(self):
        """Test causal attention mask."""
        seq_len = 5

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Mask should be upper triangular
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == True
                else:
                    assert mask[i, j] == False

    def test_positional_encoding(self):
        """Test sinusoidal positional encoding."""
        from src.signlang.world_model.signer_world_model import SinusoidalPosEmb

        dim = 64
        model = SinusoidalPosEmb(dim)

        # Test different positions
        positions = torch.arange(10).float()
        embeddings = model(positions)

        assert embeddings.shape == (10, dim)

        # Different positions should have different encodings
        assert not torch.equal(embeddings[0], embeddings[1])

    def test_sequence_ordering(self):
        """Test that sequence ordering matters."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(obs_dim=64, latent_dim=32, num_layers=2)
        model.eval()

        # Create sequence
        x = torch.randn(1, 5, 64)

        # Forward pass
        with torch.no_grad():
            output1 = model(x)

        # Reverse sequence
        x_reversed = torch.flip(x, [1])

        with torch.no_grad():
            output2 = model(x_reversed)

        # Outputs should be different
        assert not torch.equal(
            output1["gloss_logits"], output2["gloss_logits"]
        )


class TestModelGradients:
    """Tests for gradient flow in the model."""

    def test_gradient_flow_world_model(self):
        """Test gradient flow through world model."""
        from src.signlang.world_model.signer_world_model import ContinuousWorldModel

        model = ContinuousWorldModel(obs_dim=64, latent_dim=32)
        x = torch.randn(2, 5, 64)

        output = model(x)

        # Compute full loss including world_loss (KL) so dynamics/posterior get gradients
        wm_loss, _ = model.world_loss(
            x,
            output["z"],
            output["obs_pred"],
            output["z_mean"],
            output["z_logvar"],
            output["z_prior_mean"],
            output["z_prior_logvar"],
        )
        loss = wm_loss + output["gloss_logits"].sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_flow_tokenizer(self):
        """Test gradient flow through tokenizer."""
        from src.signlang.world_model.signer_world_model import VQVAETokenizer

        model = VQVAETokenizer(latent_dim=64, num_codes=100)
        z = torch.randn(2, 5, 64, requires_grad=True)

        z_q, tokens, commitment_loss = model(z)

        # Check gradient flows through commitment loss
        commitment_loss.backward()

        assert z.grad is not None

    def test_gradient_flow_full_model(self):
        """Test gradient flow through full model."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(obs_dim=64, latent_dim=32)
        x = torch.randn(2, 5, 64)

        output = model(x)

        # Compute full loss (world model + gloss + commitment) so all submodules get gradients
        wm_loss, _ = model.world_model.world_loss(
            x,
            output["latent"],
            output["world_loss"],
            output["z_mean"],
            output["z_logvar"],
            output["z_prior_mean"],
            output["z_prior_logvar"],
        )
        loss = wm_loss + output["gloss_logits"].sum() + output["commitment_loss"]
        loss.backward()

        # Check gradients exist
        # Skip: codebook (updated via EMA), llm_translator (placeholder, not in loss)
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(skip in name for skip in ("ema_", "steps_since", "codebook", "llm_translator")):
                    continue  # Updated manually via EMA or placeholder
                assert param.grad is not None, f"No gradient for {name}"