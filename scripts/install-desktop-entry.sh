#!/usr/bin/env bash
# Linux: register Space Video Enhancer in the applications menu with its own name and icon.
#
# GNOME/KDE show a window's name and icon from the .desktop entry whose name matches the window's
# app id ("space-video-enhancer", set by the app); without one they fall back to "python" and the
# Python icon. Run this once from the project folder after ./run.sh created the .venv:
#
#   ./scripts/install-desktop-entry.sh            # installs for the current user
#   ./scripts/install-desktop-entry.sh --remove   # removes it again

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
APP_ID="space-video-enhancer"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
DESKTOP_FILE="$APPS_DIR/$APP_ID.desktop"
ICON_FILE="$ICON_DIR/$APP_ID.png"

if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$DESKTOP_FILE" "$ICON_FILE"
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true
  echo "Removido: $DESKTOP_FILE"
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Ambiente virtual nao encontrado em $ROOT_DIR/.venv — rode ./run.sh uma vez antes." >&2
  exit 1
fi

mkdir -p "$APPS_DIR" "$ICON_DIR"
QT_QPA_PLATFORM=offscreen "$PYTHON" -c "
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from space_video_enhancer.ui.app_icon import write_png
QGuiApplication([])
write_png(Path('$ICON_FILE'), 256)
"

cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Space Video Enhancer
Comment=Editor de vídeo offline: preview antes/depois, estabilização, ruído, cor e exportação local
Exec=$PYTHON -m space_video_enhancer
Path=$ROOT_DIR
Icon=$ICON_FILE
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
Keywords=video;estabilizacao;denoise;ffmpeg;
StartupNotify=true
StartupWMClass=$APP_ID
DESKTOP

chmod 644 "$DESKTOP_FILE"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" 2>/dev/null || true
echo "Instalado: $DESKTOP_FILE"
echo "Icone:     $ICON_FILE"
echo "Abra o app pelo menu (ou rode ./run.sh): o dock passa a mostrar nome e icone proprios."
