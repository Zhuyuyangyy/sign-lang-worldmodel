"""
Sign Language World Model — Continuous Latent World Model (CLWM)
==============================================================
Training entry point for the sign language translation world model.

Components:
  - ContinuousWorldModel: maintains latent state z_t for signing scene
  - VQVAETokenizer: discretizes latent into discrete tokens
  - TemporalTransformer: sequence modeling of latent dynamics
  - Video prediction + sign recognition

Reference: Ha World Models (2018), DreamerV3 (2023), CNNSa-LSTM (2024)
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add src/ to path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

from src.signlang.world_model.signer_world_model import (
    SignLanguageWorldModel,
)


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset ─────────────────────────────────────────────────────────────────

class SignVideoDataset(Dataset):
    """
    Placeholder dataset for sign language video features.

    In production, replace with real I3D / SlowFast feature extraction
    from AUTSL, WLASL, or similar datasets.

    Each sample: [T, 2048] I3D video features + gloss label
    """
    def __init__(
        self,
        root: str = "data/signlang",
        num_samples: int = 1000,
        num_frames: int = 30,
        feature_dim: int = 2048,
        num_glosses: int = 100,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.num_glosses = num_glosses
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Synthetic I3D features: [T, 2048]
        video = self.rng.standard_normal((self.num_frames, self.feature_dim)).astype(np.float32)
        video = np.clip(video, -3, 3)
        video = video / 3.0  # normalize to [-1, 1]

        # Random gloss label
        gloss = self.rng.integers(0, self.num_glosses)

        return {
            "video": torch.from_numpy(video),
            "gloss": torch.tensor(gloss, dtype=torch.long),
        }


def collate_signlang(batch):
    """Collate batch of video sequences."""
    videos = torch.stack([b["video"] for b in batch])
    glosses = torch.stack([b["gloss"] for b in batch])
    return videos, glosses


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    config_path: str = "configs/default.yaml",
    experiment_dir: str = "experiments",
):
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    wm_cfg = config.get("world_model", {})

    device = torch.device(
        train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n{'='*60}")
    print(f"Sign Language World Model Training")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Model
    model = SignLanguageWorldModel(
        obs_dim=model_cfg.get("obs_dim", 2048),
        latent_dim=model_cfg.get("latent_dim", 256),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 8),
        num_codes=model_cfg.get("num_codes", 8192),
        num_glosses=model_cfg.get("num_glosses", 100),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Dataset
    data_cfg = config.get("data", {})
    train_ds = SignVideoDataset(
        root=data_cfg.get("data_dir", "data/signlang"),
        num_samples=1000,
        num_frames=data_cfg.get("num_frames", 30),
        feature_dim=model_cfg.get("obs_dim", 2048),
        num_glosses=model_cfg.get("num_glosses", 100),
        seed=42,
    )
    val_ds = SignVideoDataset(
        root=data_cfg.get("data_dir", "data/signlang"),
        num_samples=200,
        num_frames=data_cfg.get("num_frames", 30),
        feature_dim=model_cfg.get("obs_dim", 2048),
        num_glosses=model_cfg.get("num_glosses", 100),
        seed=123,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_signlang,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_signlang,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    total_steps = len(train_loader) * train_cfg.get("epochs", 100)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.get("max_lr", 1e-3),
        total_steps=total_steps,
        pct_start=0.1,
    )

    # Loss
    gloss_criterion = nn.CrossEntropyLoss()

    # Experiment directory
    exp_dir = Path(experiment_dir) / "signlang" / time.strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Training loop
    best_gloss_acc = 0.0
    beta_kl = wm_cfg.get("beta_kl", 0.1)
    commitment_cost = train_cfg.get("commitment_cost", 0.25)

    for epoch in range(1, train_cfg.get("epochs", 100) + 1):
        model.train()
        epoch_loss, epoch_recon, epoch_kl, epoch_commit, epoch_gloss = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch, (videos, glosses) in enumerate(train_loader):
            videos = videos.to(device)
            glosses = glosses.to(device)

            optimizer.zero_grad()

            # Forward
            out = model(videos)

            # World model loss: reconstruction + KL divergence
            wm_loss_dict = model.world_model.world_loss(
                videos,
                out["latent"],
                videos,  # obs_pred placeholder
                out["z_mean"],
                out["z_logvar"],
                out.get("z_prior", out["z_mean"]),
            )
            wm_loss = wm_loss_dict["total_loss"]

            # Commitment loss
            commit_loss = out["commitment_loss"] * commitment_cost

            # Gloss loss (last timestep)
            gloss_logits = out["gloss_logits"]
            gloss_pred = gloss_logits[:, -1, :]
            gloss_loss = gloss_criterion(gloss_pred, glosses)

            # Total loss
            total_loss = wm_loss + commit_loss + gloss_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg.get("max_grad_norm", 1.0),
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_recon += wm_loss_dict.get("recon_loss", 0)
            epoch_kl += wm_loss_dict.get("kl_loss", 0)
            epoch_commit += commit_loss.item()
            epoch_gloss += gloss_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for videos, glosses in val_loader:
                videos = videos.to(device)
                glosses = glosses.to(device)
                out = model(videos)
                gloss_pred = out["gloss_logits"][:, -1, :]
                preds = gloss_pred.argmax(dim=1)
                val_correct += (preds == glosses).sum().item()
                val_total += glosses.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        avg = lambda x: x / n_batches
        print(
            f"Epoch {epoch:3d} | "
            f"loss={avg(epoch_loss):.4f} | "
            f"recon={avg(epoch_recon):.2f} | "
            f"kl={avg(epoch_kl):.2f} | "
            f"commit={avg(epoch_commit):.4f} | "
            f"gloss_ce={avg(epoch_gloss):.4f} | "
            f"val_acc={val_acc:.3f}"
        )

        # Checkpoint
        if val_acc > best_gloss_acc:
            best_gloss_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, exp_dir / "best_model.pt")
            print(f"  ★ New best val_acc: {best_gloss_acc:.3f}")

        if epoch % train_cfg.get("save_interval", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
            }, exp_dir / f"checkpoint_ep{epoch}.pt")

    result = {"best_val_acc": best_gloss_acc, "epochs": epoch}
    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTraining complete. Best val_acc: {best_gloss_acc:.3f}")
    print(f"Results in {exp_dir}")
    return result


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_trajectory(
    model: SignLanguageWorldModel,
    video_features: torch.Tensor,
    num_steps: int = 30,
) -> torch.Tensor:
    """Generate future latent trajectories from initial video features."""
    model.eval()
    out = model(video_features)
    z_init = out["latent"][:, 0, :]
    return model.generate_trajectory(z_init, num_steps=num_steps)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language World Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--exp-dir", type=str, default="experiments")
    args = parser.parse_args()

    train(config_path=args.config, experiment_dir=args.exp_dir)
