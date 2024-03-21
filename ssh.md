
adduser.sh
```
#!/usr/bin/env bash

set -e
set -x

username=${1:-""}
uid=${2:-""}
ssh_key_pub=${3:-""}
if [ "${username}" == "" ]; then
  echo "need username"
  exit 1
fi
adduser "${username}" --uid ${uid}  --disabled-password
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
__conda_setup="$('/ML-A100/team/mm/gujiasheng/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ML-A100/team/mm/gujiasheng/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ML-A100/team/mm/gujiasheng/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ML-A100/team/mm/gujiasheng/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
alias proxy_on="export http_proxy=100.66.28.72:3128;export https_proxy=100.66.28.72:3128"
alias proxy_off="unset http_proxy https_proxy"
alias gjs="cd /ML-A100/team/mm/gujiasheng"
alias ns="watch -n 1 nvidia-smi"
alias ll="ls -l"
alias sb="source ~/.bashrc"
export HF_HOME=/ML-A100/team/mm/gujiasheng/.cache/huggingface
export TMPDIR=/ML-A100/team/mm/gujiasheng/tmp
export PIP_CACHE_DIR=/ML-A100/team/mm/gujiasheng/.cache/pip
export HOME=/ML-A100/team/mm/gujiasheng

[ -r ~/.volc/.profile ] && source ~/.volc/.profile #[volc installer]

 export PATH="/usr/local/cuda-11.8/bin:$PATH"
 # export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

```

use an already existed directory as home
```
useradd <username> --home-dir .../<username> --no-create-home --uid 1234 --non-unique --shell /bin/bash
passwd <username> 

vi /etc/group  # add user to group
sudo visudo  # add user as sudoers
sudo chown -R <newowner>:<newgroup> <directory>
```
