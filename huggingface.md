 ```
export HF_ENDPOINT=https://hf-mirror.com
```
```
export HF_HOME=${HOME}/cache
```
```
# https://huggingface.co/docs/huggingface_hub/package_reference/file_download
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="adept/fuyu-8b", help='Name of the model to download')
args = parser.parse_args()

snapshot_download(repo_id=args.model_name,
                  repo_type='model',
                  local_dir='./'+args.model_name.split('/')[1],
                  cache_dir='./cache',
                  resume_download=True,
                  local_dir_use_symlinks=False)

```
### Command:
```
pip install -U huggingface_hub
pip install -U hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --resume-download playgroundai/playground-v2.5-1024px-aesthetic --local-dir 要保存的路径 --local-dir-use-symlinks False
```
