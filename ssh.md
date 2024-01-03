
adduser.sh
```
#!/usr/bin/env bash

set -e
set -x

username=${1:-""}
ssh_key_pub=${2:-""}
if [ "${username}" == "" ]; then
  echo "need username"
  exit 1
fi
sudo adduser "${username}" --disabled-password
mkdir -p /home/${username}/.ssh
echo "${ssh_key_pub}" >> /home/${username}/.ssh/authorized_keys
chown -R ${username}:${username} /home/${username}
chmod 700 /home/${username}/.ssh
chmod 600 /home/${username}/.ssh/authorized_keys
usermod -aG sudo ${username}
```

```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/ML-A100/home/gujiasheng/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ML-A100/home/gujiasheng/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ML-A100/home/gujiasheng/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ML-A100/home/gujiasheng/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
alias proxy_on="export http_proxy=x;export https_proxy=x"
alias proxy_off="unset http_proxy https_proxy"
alias gjs="cd /ML-A100/home/gujiasheng"
# export CUDA_HOME="/usr/local/cuda-12.1"
export HF_HOME=/ML-A100/home/gujiasheng/cache
```

use an already existed directory as home
```
useradd <username> --home-dir .../<username> --no-create-home --uid 1234 --non-unique --shell /bin/bash
passwd <username> 

vi /etc/group  # add user to group
sudo visudo  # add user as sudoers
sudo chown -R <newowner>:<newgroup> <directory>
```
