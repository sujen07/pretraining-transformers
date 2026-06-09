#!/usr/bin/env bash
# Create a local virtual environment and install project dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "Error: $PYTHON not found. Install Python 3.10+ and try again." >&2
  exit 1
fi

echo "Using $($PYTHON --version)"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
else
  echo "Virtual environment .venv already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Verify installation:"
echo "  python -c \"import torch; import matplotlib; print('torch', torch.__version__)\""
