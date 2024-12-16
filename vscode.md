# Plugins:
Live server: Launch a development local Server with live reload feature for static & dynamic pages

Bookmarks: Mark lines and jump to them

Gitlens: Supercharge Git within VS Code

Partial Diff: Compare (diff) text selections within a file, across files, or to the clipboard

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

## vscode在最上面显示当前类：  
打开 VSCode。  
进入设置，你可以通过点击左下角的齿轮图标然后选择“设置”，或者直接使用快捷键 Ctrl + , (Windows/Linux) 或 Cmd + , (macOS)。  
在设置搜索框中，输入“Breadcrumbs”进行搜索。  
确保勾选了“Editor › Breadcrumbs: Enabled”选项，以启用 Breadcrumbs 功能。  
你还可以调整其他相关设置，如“Editor › Breadcrumbs: FilePath”和“Editor › Breadcrumbs: SymbolPath”来控制显示的信息详细程度。  

## input() 可以用方向键
```python
import readline
```

## vscode解决“无法将“xxx”项识别为 cmdlet、函数、脚本文件或可运行程序的名称“
```bash
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")
```
https://juejin.cn/post/7411516362915151899

