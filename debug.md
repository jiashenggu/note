# 把默认后端改成 Tk
```json
            "env": {
                "MPLBACKEND": "TkAgg"
            }
```
# debug: 
```json
{
    "justMyCode": false, will not skip code in packages
}
```
## accelerate
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: accelerate",
            "type": "debugpy",
            "request": "launch",
            "module": "accelerate.commands.launch",
            "args": [
                "--config_file", "megatron_gpt_config.yaml",
                "--debug",
                "./examples/by_feature/megatron_lm_gpt_pretraining.py",
                "--config_name ", "gpt2-large",
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```
## torchrun
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: olmo",
            "type": "debugpy",
            "request": "launch",
            "program": "~/miniconda3/envs/aa/lib/python3.10/site-packages/torch/distributed/run.py",
            "args": [
                "--nproc_per_node=1",
                "--master_port=31500",
                "scripts/train.py",
                "configs/official/OLMo-1B.yaml",
                "--save_overwrite"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

## jupyter:  
在设置里，搜索 debug jupyter 可以很容易看到
![image](https://github.com/jiashenggu/note/assets/32376856/b6612658-4f97-4acf-b501-7780ac78a798)

# others 

## text automatically change lines
```json
{
    "notebook.output.wordWrap": true
}
```
## tabs wrap to multiple lines
workspace settings
```json
{
    "workbench.editor.wrapTabs": true
}
```

## save the intermediate values
```
import pickle
filename = "obs_clean_train_rtc.pkl"
with open(filename, "wb") as f:
    pickle.dump(obs_clean, f)
```
