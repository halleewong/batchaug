#!/bin/bash
# Create a fresh conda environment, install batchaug, and run the tests.
#
# Usage:
#   bash scripts/setup_and_test.sh
#
set -e

ENV_NAME="batchaug"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Remove existing env if present
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing conda env '${ENV_NAME}'..."
    conda env remove -n "$ENV_NAME" -y
fi

echo "Creating conda env '${ENV_NAME}' with Python 3.10..."
conda create -n "$ENV_NAME" python=3.10 -y

echo "Activating env..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "Installing batchaug with test and triton extras..."
pip install -e "${PROJECT_DIR}[test,triton]"

echo "Running tests..."
python -m pytest "${PROJECT_DIR}/tests/" -v

echo "Done. All tests passed."
