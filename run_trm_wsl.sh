#!/bin/bash
# TRM Training Script for WSL using Windows venv with CUDA

# Convert Windows path to WSL path (using quotes for spaces)
VENV_PYTHON='/mnt/c/Users/theo-/OneDrive/Documents/VS Code project/.venv/Scripts/python.exe'
PROJECT_DIR='/mnt/c/Users/theo-/OneDrive/Documents/VS Code project/Deep learning/Project'

echo "============================================"
echo "TRM Multi-Configuration Training (WSL)"
echo "============================================"

cd "$PROJECT_DIR" || exit 1

echo ""
echo "Checking PyTorch and CUDA..."
"$VENV_PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "Starting training..."
"$VENV_PYTHON" train_trm_multi.py
