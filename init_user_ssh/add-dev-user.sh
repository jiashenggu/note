#!/usr/bin/env bash
# add-dev-user.sh  ——  幂等创建开发账号，可选把 /root 迁走
set -euo pipefail
IFS=$'\n\t'
SCRIPT=$(basename "$0")
LOG="/var/log/${SCRIPT%.sh}.log"

exec 1> >(tee -a "$LOG")
exec 2> >(tee -a "$LOG" >&2)

function log()  { echo "[$(date '+%F %T')] $*"; }
function fail() { log "ERROR: $*"; exit 1; }

########## 入参 ##########
USERNAME=${1:-}
UID_=${2:-}
SSH_PUB=${3:-}
HOME_PARENT=${4:-/home}          # 想放到 /mnt/data/home 就传 /mnt/data/home
PASSWORD=${5:-}                   # 可空，空则锁定密码
MIGRATE_ROOT=${6:-false}          # true 时会把 /root 迁到 ${HOME_PARENT}/root

########## 基本校验 ##########
[[ -z "$USERNAME" ]] && fail "USAGE: $0 <username> <uid> [ssh_pub] [home_parent] [password] [migrate_root:true|false]"
[[ -z "$UID_" ]]     && fail "uid required"
[[ $EUID -ne 0 ]]    && fail "请用 root 运行"

########## 1. 可选：迁移 /root ##########
if [[ "$MIGRATE_ROOT" == "true" ]]; then
    NEW_ROOT="${HOME_PARENT}/root"
    if [[ "$(realpath /root)" != "$(realpath "$NEW_ROOT")" ]]; then
        log "migrate /root -> $NEW_ROOT"
        rsync -aHAX /root/ "${NEW_ROOT}/"
        usermod -d "$NEW_ROOT" -m root
        log "/root migrated"
    fi
fi

########## 2. 幂等：用户已存在则跳过 ##########
if getent passwd "$USERNAME" &>/dev/null; then
    log "$USERNAME already exists, skip"
    exit 0
fi

########## 3. 创建用户主目录上层 ##########
mkdir -p "$HOME_PARENT"
if ! mountpoint -q "$HOME_PARENT"; then
    log "WARN: $HOME_PARENT 不是独立挂载点，建议把数据盘挂到这里"
fi

########## 4. 生成密码哈希 ##########
if [[ -n "$PASSWORD" ]]; then
    PASS_HASH=$(openssl passwd -6 <<<"$PASSWORD")
else
    PASS_HASH="!"   # 锁定密码
fi

########## 5. 建用户 ##########
useradd -m "$USERNAME" \
        --home-dir "${HOME_PARENT}/${USERNAME}" \
        --uid "$UID_" \
        --gid "$UID_" \
        --shell /bin/bash \
        --password "$PASS_HASH" \
        --comment "DevUser $USERNAME"

########## 6. SSH 公钥 ##########
if [[ -n "$SSH_PUB" ]]; then
    SSH_DIR="${HOME_PARENT}/${USERNAME}/.ssh"
    mkdir -p "$SSH_DIR"
    echo "$SSH_PUB" > "$SSH_DIR/authorized_keys"
    chmod 700 "$SSH_DIR"
    chmod 600 "$SSH_DIR/authorized_keys"
    chown -R "$USERNAME:$USERNAME" "$SSH_DIR"
fi

########## 7. sudo 权限 ##########
usermod -aG sudo "$USERNAME"
log "$USERNAME created successfully (uid=$UID_, home=${HOME_PARENT}/${USERNAME})"
