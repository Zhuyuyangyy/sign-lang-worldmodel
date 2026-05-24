#!/bin/bash
# Sign Language World Model — start script
# Usage:
#   ./start.sh train       — train model
#   ./start.sh generate    — run generation demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

echo "=== Sign Language World Model ==="

case "${1:-train}" in
  train)
    echo "Starting training..."
    $PYTHON main.py --config configs/default.yaml --exp-dir experiments
    ;;
  generate)
    CHECKPOINT="${CHECKPOINT:-experiments/best_model.pt}"
    echo "Running generation demo..."
    $PYTHON -c "
import torch, sys
sys.path.insert(0, 'src')
from signlang.world_model.signer_world_model import SignLanguageWorldModel
model = SignLanguageWorldModel()
ckpt = torch.load('$CHECKPOINT', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
video = torch.randn(1, 30, 2048)
out = model(video)
print(f'Gloss logits: {out[\"gloss_logits\"].shape}')
print(f'Tokens: {out[\"tokens\"].shape}')
print(f'Latent: {out[\"latent\"].shape}')
print('Generation test passed.')
"
    ;;
  *)
    echo "Usage: $0 {train|generate}"
    exit 1
    ;;
esac
