### Use wget to download models:
```
wget https://civitai.com/api/download/models/{modelVersionId} --content-disposition
```

### VAE fp16 format: sdxl-vae-fp16-fix:
```
https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main
```

### reproduce results: "copy generation data" and paste it to the "prompt" box in Webui.

### How to convert pretrained SDXL .safetensors model to diffusers folder format
```
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file()
pipeline.save_pretrained(...)
```
