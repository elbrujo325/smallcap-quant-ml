#!/usr/bin/env bash
# Setup virtual environment and install requirements
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VE_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VE_DIR" ]; then
    python3 -m venv "$VE_DIR"
fi

echo "Activating virtualenv at $VE_DIR"
# shellcheck disable=SC1090
source "$VE_DIR/bin/activate"

echo "Installing requirements from requirements.txt"
python3 -m pip install --upgrade pip setuptools wheel
pip install -r "$ROOT_DIR/requirements.txt"

echo "Environment ready. Activate with: source $VE_DIR/bin/activate"