#!/usr/bin/env bash

set -e  # exit immediately if any command exits with a non-zero status
set -x  # all executed commands are printed to the terminal

username=${1:-""}
uid=${2:-""}
ssh_key_pub=${3:-""}
home_dir=${4:-"/home"}
password=${5:-""}
if [ "${username}" == "" ]; then
  echo "need username"
  exit 1
fi

# -o option allows creating a user with a non-unique UID
useradd "${username}" --home-dir ${home_dir}/${username} -o -u ${uid} -s /bin/bash --password ${password}
if [ "$uid" -ne 0 ]; then
    groupmod -g ${uid} ${username}
else
    echo "uid is equal to root"
    groupmod -o -g ${uid} ${username}
fi


mkdir -p ${home_dir}/${username}/.ssh

echo "${ssh_key_pub}" >> ${home_dir}/${username}/.ssh/authorized_keys
chown -R ${username}:${username} ${home_dir}/${username}/.ssh
chmod 700 ${home_dir}/${username}/.ssh
chmod 600 ${home_dir}/${username}/.ssh/authorized_keys
usermod -aG sudo ${username}
usermod -aG root ${username}

# MUST execute: chown root:root /ML-A100/team/mm/gujiasheng
# passwd gujiasheng  # g