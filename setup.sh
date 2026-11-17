#!/usr/bin/env bash
# One-time setup on Ubuntu: Python env, dependencies, tests, child model, and the Ollama teacher.
#   bash setup.sh                      everything (installs Ollama if the config uses it; may ask for sudo)
#   SKIP_OLLAMA=1 bash setup.sh        skip Ollama (use teacher.backend: scripted or anthropic)
#   WITH_CLAUDE=1 bash setup.sh        also install the Anthropic SDK for the Claude teacher
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "python3 not found. Install it with: sudo apt install python3 python3-venv" >&2
  exit 1
fi
if ! "$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "Python venv support is missing. Install it with: sudo apt install python3-venv" >&2
  exit 1
fi

echo "==> Python environment"
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt -r requirements-dev.txt
if [ "${WITH_CLAUDE:-0}" = "1" ]; then
  pip install --quiet -r requirements-claude.txt
fi

python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"GPU ready: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: PyTorch cannot see a GPU, so it will run on CPU (slower, still works). Check `nvidia-smi`.")
EOF

echo "==> Unit tests"
python -m pytest -q

echo "==> Child model"
python -m devlm download

read -r BACKEND MODEL < <(python - <<'EOF'
from pathlib import Path
from devlm.config import load_config
cfg = load_config(Path("config.yaml"))
print(cfg.teacher.backend, cfg.teacher.model)
EOF
)

if [ "$BACKEND" = "ollama" ] && [ "${SKIP_OLLAMA:-0}" != "1" ]; then
  echo "==> Teacher (Ollama, model $MODEL)"
  if ! command -v ollama >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is needed to install Ollama: sudo apt install curl" >&2
      exit 1
    fi
    echo "Installing Ollama from https://ollama.com (it may ask for your sudo password)..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if ! ollama list >/dev/null 2>&1; then
    echo "Starting the Ollama server in the background (log: ~/.ollama-serve.log)"
    nohup ollama serve >"$HOME/.ollama-serve.log" 2>&1 &
    for _ in $(seq 1 20); do
      ollama list >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  ollama pull "$MODEL"
elif [ "$BACKEND" = "ollama" ]; then
  echo "Skipped Ollama. Set teacher.backend to scripted or anthropic in config.yaml, or install Ollama later."
fi

echo
echo "Setup done. Start the simulation with:  ./run.sh start --days 10"
