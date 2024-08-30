```python
import safetensors.torch
sd = safetensors.torch.load_file("/xpfs/public/gjs/x-flux/lora/watercolor.safetensors")
sd_out = {}

for k in sd:
  sd_out["diffusion_model.{}".format(k.replace(".down.weight", ".lora_down.weight").replace(".up.weight", ".lora_up.weight").replace(".processor.proj_lora1.", ".img_attn.proj.").replace(".processor.proj_lora2.", ".txt_attn.proj.").replace(".processor.qkv_lora1.", ".img_attn.qkv.").replace(".processor.qkv_lora2.", ".txt_attn.qkv."))] = sd[k]

safetensors.torch.save_file(sd_out, "/xpfs/public/gjs/x-flux/lora/watercolor_xflux.safetensors")
```
