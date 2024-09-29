![60e84fa46f60964f8f791f9bcda01f95](https://github.com/user-attachments/assets/e2afa524-fc07-45e3-a68b-864f2e793581)
```
export HF_ENDPOINT=https://hf-mirror.com
```
```
export HF_HOME=${HOME}/.cache
```
```python
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default="adept/fuyu-8b", help='Name of the model to download')
args = parser.parse_args()

def download_model():
    try:
        snapshot_download(repo_id=args.model_name,
                          repo_type='model',
                          local_dir='./'+(args.model_name.split('/')[1] if len(args.model_name.split('/'))==2 else args.model_name.split('/')[0]),
                          cache_dir='/ML-A100/team/mm/gujiasheng/.cache',
#                           allow_patterns=["*.json", "*.bin", "*.md", "*.safetensors", "*.pth", "*.onnx"],
                          ignore_patterns=[])
    except Exception as e:
        print(e)
        print("Download failed. Resuming download...")
        download_model()

download_model()


print("Model download complete!")
```

```python
from huggingface_hub import snapshot_download
import argparse
import huggingface_hub 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', type=str, default="adept/fuyu-8b", help='Name of the dataset to download')
args = parser.parse_args()

def download_dataset():
    try:
        snapshot_download(repo_id=args.dataset_name,
                          repo_type='dataset',
                          local_dir='./'+(args.dataset_name.split('/')[1] if len(args.dataset_name.split('/'))==2 else args.dataset_name.split('/')[0]),
                          local_dir_use_symlinks=False,
                          cache_dir='/ML-A100/team/mm/gujiasheng/.cache',
                          force_download=False)
    except Exception as e:
        print("Error:", e)
        print("Download failed. Resuming download...")
        download_dataset()

download_dataset()

print("Dataset download complete!")
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
### Gated dataset

Change auth settings to fit usage.
```python
import huggingface_hub

huggingface_hub.login(token="")
```

```bash
huggingface-cli login
```
