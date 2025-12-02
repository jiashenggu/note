#!/usr/bin/env bash
set -euo pipefail
USER=$1
[[ -z "$USER" ]] && exit 1
[[ $EUID -ne 0 ]] && echo "need root" && exit 1

# 先踢已登录会话
pkill -9 -u "$USER" || true

# 删用户、删家目录、清 sudo
userdel -rf "$USER" 2>/dev/null || true
log "user $USER removed"
