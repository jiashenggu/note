```
export HF_ENDPOINT=https://hf-mirror.com
```
```
export HF_HOME=${HOME}/cache
```
```
from huggingface_hub import snapshot_download

snapshot_download(repo_id='lmsys/vicuna-13b-v1.5-16k',
                  repo_type='model',
                  local_dir='./vicuna-13b-v1.5-16k',
                  cache_dir='./cache',
                  resume_download=True)
```
