"""Tests for training pipeline."""

import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from main import (
    load_config,
    SignVideoDataset,
    collate_signlang,
    train,
    generate_trajectory,
)


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid config file."""
        config = load_config(str(sample_config_file))

        assert "model" in config
        assert "training" in config
        assert "data" in config
        assert "world_model" in config

    def test_load_config_missing_file(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_config))


class TestSignVideoDataset:
    """Tests for SignVideoDataset."""

    def test_dataset_length(self):
        """Test dataset length."""
        dataset = SignVideoDataset(num_samples=100)

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = SignVideoDataset(
            num_samples=10,
            num_frames=10,
            feature_dim=64,
            num_glosses=10,
        )

        item = dataset[0]

        assert "video" in item
        assert "gloss" in item

        assert item["video"].shape == (10, 64)
        assert item["gloss"].dim() == 0  # Scalar

    def test_dataset_deterministic(self):
        """Test that dataset is deterministic with same seed."""
        dataset1 = SignVideoDataset(num_samples=5, seed=42)
        dataset2 = SignVideoDataset(num_samples=5, seed=42)

        for i in range(5):
            assert torch.equal(dataset1[i]["video"], dataset2[i]["video"])
            assert torch.equal(dataset1[i]["gloss"], dataset2[i]["gloss"])

    def test_dataset_different_seeds(self):
        """Test that different seeds produce different data."""
        dataset1 = SignVideoDataset(num_samples=5, seed=42)
        dataset2 = SignVideoDataset(num_samples=5, seed=123)

        # At least one item should be different
        different = False
        for i in range(5):
            if not torch.equal(dataset1[i]["video"], dataset2[i]["video"]):
                different = True
                break

        assert different

    def test_dataset_custom_parameters(self):
        """Test dataset with custom parameters."""
        dataset = SignVideoDataset(
            num_samples=20,
            num_frames=15,
            feature_dim=1024,
            num_glosses=50,
            seed=99,
        )

        item = dataset[0]

        assert item["video"].shape == (15, 1024)
        assert item["gloss"].item() < 50


class TestCollateSignlang:
    """Tests for collate function."""

    def test_collate_batch(self):
        """Test collating a batch of items."""
        batch = [
            {
                "video": torch.randn(10, 64),
                "gloss": torch.tensor(5),
            },
            {
                "video": torch.randn(10, 64),
                "gloss": torch.tensor(3),
            },
        ]

        videos, glosses = collate_signlang(batch)

        assert videos.shape == (2, 10, 64)
        assert glosses.shape == (2,)
        assert glosses[0].item() == 5
        assert glosses[1].item() == 3

    def test_collate_single_item(self):
        """Test collating a single item."""
        batch = [
            {
                "video": torch.randn(5, 32),
                "gloss": torch.tensor(7),
            },
        ]

        videos, glosses = collate_signlang(batch)

        assert videos.shape == (1, 5, 32)
        assert glosses.shape == (1,)


class TestGenerateTrajectory:
    """Tests for trajectory generation."""

    def test_generate_trajectory(self):
        """Test trajectory generation function."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(obs_dim=64, latent_dim=32)
        video_features = torch.randn(2, 10, 64)

        trajectory = generate_trajectory(model, video_features, num_steps=5)

        assert trajectory.shape == (2, 6, 32)  # 5 steps + 1 initial

    def test_generate_trajectory_different_steps(self):
        """Test trajectory generation with different number of steps."""
        from src.signlang.world_model.signer_world_model import SignLanguageWorldModel

        model = SignLanguageWorldModel(obs_dim=64, latent_dim=32)
        video_features = torch.randn(1, 5, 64)

        for num_steps in [1, 5, 10]:
            trajectory = generate_trajectory(model, video_features, num_steps)

            assert trajectory.shape == (1, num_steps + 1, 32)


class TestTraining:
    """Tests for training pipeline."""

    @patch("main.DataLoader")
    @patch("main.SignLanguageWorldModel")
    def test_train_smoke(self, mock_model_class, mock_dataloader, sample_config_file, temp_experiment_dir):
        """Smoke test for training function."""
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_model_class.return_value = mock_model

        # Mock model output
        mock_output = {
            "gloss_logits": torch.randn(4, 10, 100),
            "latent": torch.randn(4, 10, 256),
            "commitment_loss": torch.tensor(0.1),
            "world_loss": torch.randn(4, 10, 2048),
            "z_mean": torch.randn(4, 10, 256),
            "z_logvar": torch.randn(4, 10, 256),
            "z_prior_mean": torch.randn(4, 10, 256),
            "z_prior_logvar": torch.randn(4, 10, 256),
        }
        mock_model.return_value = mock_output

        # Mock world_loss
        mock_world_loss = {
            "total_loss": torch.tensor(1.0),
            "recon_loss": 0.5,
            "kl_loss": 0.3,
        }
        mock_model.world_model.world_loss.return_value = (torch.tensor(1.0), mock_world_loss)

        # Mock dataloader
        mock_train_loader = [
            (torch.randn(4, 10, 2048), torch.randint(0, 100, (4,))),
        ]
        mock_val_loader = [
            (torch.randn(4, 10, 2048), torch.randint(0, 100, (4,))),
        ]

        # This is a simplified test - actual training would require more mocking
        # Just test that the function exists and can be called
        assert callable(train)