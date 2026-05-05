#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
STAMP_FILE="$VENV_DIR/.deps-installed"

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 nao encontrado no PATH." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Criando ambiente virtual..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  echo "Inicializando pip no ambiente virtual..."
  python -m ensurepip --upgrade
fi

if [[ ! -f "$STAMP_FILE" || "$ROOT_DIR/pyproject.toml" -nt "$STAMP_FILE" || "$ROOT_DIR/setup.py" -nt "$STAMP_FILE" ]]; then
  echo "Instalando dependencias do projeto..."
  python -m pip install --upgrade pip
  python -m pip install .
  touch "$STAMP_FILE"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Aviso: ffmpeg nao encontrado no PATH. A interface abre, mas a exportacao nao vai funcionar." >&2
fi

exec python main.py "$@"