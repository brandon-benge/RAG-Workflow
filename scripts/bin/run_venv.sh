#!/usr/bin/env bash

# (moved to scripts/bin/) Activate local virtual environment and run a Python script.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts/bin}"

# Read virtual environment path from params.yaml
VENV_PATH=".venv"  # default fallback
PYTHON_EXEC="python3"  # default fallback
if [ -f "$PROJECT_ROOT/params.yaml" ]; then
  # Extract venv path from YAML (simple grep/sed approach)
  VENV_PATH=$(grep "path:" "$PROJECT_ROOT/params.yaml" | head -1 | sed 's/.*path: *"\([^"]*\)".*/\1/')
  PYTHON_EXEC=$(grep "python_executable:" "$PROJECT_ROOT/params.yaml" | head -1 | sed 's/.*python_executable: *"\([^"]*\)".*/\1/')
  # Use defaults if extraction failed
  [ -z "$VENV_PATH" ] && VENV_PATH=".venv"
  [ -z "$PYTHON_EXEC" ] && PYTHON_EXEC="python3"
fi

# Set VENV_DIR based on relative or absolute path
if [[ "$VENV_PATH" = /* ]]; then
  VENV_DIR="$VENV_PATH"  # absolute path
else
  VENV_DIR="$PROJECT_ROOT/$VENV_PATH"  # relative path
fi
if [ ! -d "$VENV_DIR" ]; then
  echo "[info] Creating virtual environment at $VENV_PATH using $PYTHON_EXEC" >&2
  "$PYTHON_EXEC" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate" 2>/dev/null
# Ensure the virtual environment's bin directory is in PATH
export PATH="$VENV_DIR/bin:$PATH"
# Always ensure dependencies (idempotent, quiet)
# Prefer root requirements.txt; allow params.yaml to override if needed
PRIMARY_REQ="requirements.txt"  # default
if [ -f "$PROJECT_ROOT/params.yaml" ]; then
  PR=$(grep "^\s*primary:\s*\".*\"" "$PROJECT_ROOT/params.yaml" | head -1 | sed 's/.*primary: *"\([^"]*\)".*/\1/') || true
  [ -n "$PR" ] && PRIMARY_REQ="$PR"
fi

python -m pip install --upgrade pip >/dev/null 2>&1 || true
if [ -f "$PROJECT_ROOT/$PRIMARY_REQ" ]; then
  echo "[info] Syncing dependencies ($PRIMARY_REQ)" >&2
  python -m pip install -q -r "$PROJECT_ROOT/$PRIMARY_REQ" || true
else
  echo "[info] Installing minimal dependency: PyYAML (no requirements file found)" >&2
  python -m pip install -q PyYAML || true
fi

# Check if the first argument is a command that should be executed directly
if [ $# -gt 0 ] && [ "$1" = "pip" ] || [ "$1" = "python" ] || [ "$1" = "pytest" ] || [ "$1" = "jupyter" ]; then
  exec "$@"
else
  exec python "$@"
fi
