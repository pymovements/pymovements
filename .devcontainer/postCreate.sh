#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/pymovements

# Initialize a virtual environment inside your workspace container 
# This eliminates the "site-packages is not writeable" warning entirely
uv venv .venv
source .venv/bin/activate

# Use uv instead of pip for lightning-fast concurrent downloads
uv pip install --upgrade pip
uv pip install -e ".[dev,docs]"

if [ -f .pre-commit-config.yaml ]; then
  pre-commit install || true
fi

python -c "import pymovements; print('pymovements import OK')"
pytest --collect-only -q || true

node --version
npm --version
python --version
R --version