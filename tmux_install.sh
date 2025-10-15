#!/bin/bash

# Exit on error
set -e

echo "[tmux] Setting up tmux..."

# Install tmux if needed
if ! command -v tmux &> /dev/null; then
    echo "[tmux] Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Set home directory
HOME_DIR="${HOME:-/home/$USER}"
[ "$USER" = "root" ] && HOME_DIR="/root"

# Copy tmux config
if [ -f "docker/.tmux.conf" ]; then
    cp "docker/.tmux.conf" "$HOME_DIR/.tmux.conf"
    chmod 644 "$HOME_DIR/.tmux.conf"
else
    echo "[tmux] Error: Run this script from project root as: ./docker/run_tmux.sh"
    exit 1
fi

# Install TPM and plugins
TPM_DIR="$HOME_DIR/.tmux/plugins/tpm"
if [ ! -d "$TPM_DIR" ]; then
    git clone https://github.com/tmux-plugins/tpm "$TPM_DIR"
fi

# Fix ownership in Docker
if [ -f /.dockerenv ] && [ "$(stat -c '%U' "$HOME_DIR/.tmux" 2>/dev/null)" != "$USER" ]; then
    sudo chown -R $USER:$USER "$HOME_DIR/.tmux"
fi

# Install plugins
if [ -f "$TPM_DIR/bin/install_plugins" ]; then
    tmux start-server
    $TPM_DIR/bin/install_plugins
fi

echo "[tmux] Setup complete! Shortcuts: Mouse on, Alt+r reload, Alt+|/- split, Alt+arrows navigate"

# Check if already in tmux
if [ -n "$TMUX" ]; then
    echo "[tmux] Already in tmux! Use Ctrl+b,s to switch sessions"
    exit 0
fi

# Attach or create groot session
if tmux list-sessions 2>/dev/null | grep -q "^groot:"; then
    exec tmux attach-session -t groot
else
    exec tmux new-session -s groot
fi   
