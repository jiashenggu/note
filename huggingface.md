```
export HF_ENDPOINT=https://hf-mirror.com
```
```
export HF_HOME=${HOME}/cache
```
```
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="adept/fuyu-8b", help='Name of the model to download')
args = parser.parse_args()

snapshot_download(repo_id=args.model_name,
                  repo_type='model',
                  local_dir='./'+args.model_name.split('/')[1],
                  cache_dir='./cache',
                  resume_download=True)
```
