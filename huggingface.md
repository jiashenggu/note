 ```
export HF_ENDPOINT=https://hf-mirror.com
```
```
export HF_HOME=${HOME}/.cache
```
```python
# https://huggingface.co/docs/huggingface_hub/package_reference/file_download
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="adept/fuyu-8b", help='Name of the model to download')
args = parser.parse_args()

snapshot_download(repo_id=args.model_name,
                  repo_type='model',
                  local_dir='./'+args.model_name.split('/')[1],
                  cache_dir='~/.cache',
                  resume_download=True,
                  local_dir_use_symlinks=False)


# For example, you can use allow_patterns to only download JSON configuration files:
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")

# On the other hand, ignore_patterns can exclude certain files from being downloaded. The following example ignores the .msgpack and .h5 file extensions:

from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])

```
### Command:
```bash
pip install -U huggingface_hub
pip install -U hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```
```bash
huggingface-cli download --resume-download $MODEL_NAME --local-dir $LOCAL_DIR --local-dir-use-symlinks False
```
```bash
huggingface-cli download --resume-download $DATASET_NAME --repo-type dataset --local-dir 要保存的路径 --local-dir-use-symlinks False
huggingface-cli download --resume-download $SPACE --repo-type space --local-dir 要保存的路径 --local-dir-use-symlinks False
```
```bash
huggingface-cli download bigscience/bloom --include *.safetensors
huggingface-cli download bigscience/bloom --exclude *.safetensors
```
