### Use wget to download models:
```
wget https://civitai.com/api/download/models/{modelVersionId}?token=0806782ed30e485eddfb4c4f03d3ece9 --content-disposition
```

### VAE fp16 format: sdxl-vae-fp16-fix:
```
https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main
```

### reproduce results: "copy generation data" and paste it to the "prompt" box in Webui.

### How to convert SDXL .safetensors format to diffusers folder format
```
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file()
pipeline.save_pretrained(...)
```
```
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file("<your-safetensors-file>", torch_dtype=torch.float16)
pipeline.save_pretrained("<your-diffusers-folder>", variant="fp16")
```

### How to convert SDXL diffusers folder format to .safetensors format
https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py
