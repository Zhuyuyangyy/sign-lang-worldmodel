"""Pytest configuration and fixtures for sign-lang-worldmodel tests."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_video_features():
    """Create sample video features for testing."""
    batch_size = 2
    num_frames = 30
    feature_dim = 2048
    return torch.randn(batch_size, num_frames, feature_dim)


@pytest.fixture
def sample_gloss_labels():
    """Create sample gloss labels for testing."""
    batch_size = 2
    num_glosses = 100
    return torch.randint(0, num_glosses, (batch_size,))


@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return {
        "obs_dim": 2048,
        "latent_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "num_codes": 8192,
        "num_glosses": 100,
    }


@pytest.fixture
def training_config():
    """Default training configuration for testing."""
    return {
        "epochs": 2,
        "batch_size": 4,
        "lr": 1e-4,
        "max_lr": 1e-3,
        "weight_decay": 1e-5,
        "max_grad_norm": 1.0,
        "commitment_cost": 0.25,
        "device": "cpu",
        "num_workers": 0,
        "save_interval": 10,
    }


@pytest.fixture
def temp_experiment_dir(tmp_path):
    """Create a temporary experiment directory."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


@pytest.fixture
def sample_config_file(tmp_path, model_config, training_config):
    """Create a sample config file for testing."""
    config = {
        "model": model_config,
        "training": training_config,
        "data": {
            "data_dir": "data/signlang",
            "num_frames": 30,
        },
        "world_model": {
            "beta_kl": 0.1,
        },
    }

    config_path = tmp_path / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path