#!/usr/bin/env bash
# (moved to scripts/bin/) Activate local virtual environment and run a Python script.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts/bin}"
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[info] Creating virtual environment at .venv" >&2
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate" 2>/dev/null
# Always ensure dependencies (idempotent, quiet)
if [ -f "$PROJECT_ROOT/scripts/rag/requirements.txt" ]; then
  echo "[info] Syncing dependencies (scripts/rag/requirements.txt)" >&2
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
  python -m pip install -q -r "$PROJECT_ROOT/scripts/rag/requirements.txt" || true
elif [ -f "$PROJECT_ROOT/requirements.txt" ]; then
  echo "[info] Syncing dependencies (requirements.txt)" >&2
  python -m pip install -q -r "$PROJECT_ROOT/requirements.txt" || true
fi
exec python "$@"
