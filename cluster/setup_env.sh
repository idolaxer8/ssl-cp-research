#!/bin/bash
# One-time conda env setup for the cluster. Run from repo root:
#   bash cluster/setup_env.sh
#
# Creates a conda env named `ssl-cp` matching requirements.txt.
# Skip this if you're using modules + venv instead — see extract_cifar100.sbatch Option B/C.

set -euo pipefail

ENV_NAME=ssl-cp
PY_VERSION=3.10

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install miniconda first, e.g.:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh"
    echo "  bash ~/miniconda.sh -b -p \$HOME/miniconda3"
    echo "  ~/miniconda3/bin/conda init bash && exec \$SHELL"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' (python $PY_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

conda activate "$ENV_NAME"

echo "Installing pinned requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA-aware torch is usable.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible — load CUDA module before running, or reinstall torch with CUDA wheels'; print('OK — torch', torch.__version__, 'sees', torch.cuda.get_device_name(0))"

echo "Done. Activate later with: conda activate $ENV_NAME"
