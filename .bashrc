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
alias gpu="echo $CUDA_VISIBLE_DEVICES"
function 2gpu() {
    export CUDA_VISIBLE_DEVICES="$1"
}

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
