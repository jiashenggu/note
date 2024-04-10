debug: 
```
{
    "justMyCode": false, will not skip code in packages
}
```
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "module": "accelerate.commands.launch",
            "args": [
                "--config_file", "megatron_gpt_config.yaml",
                "./examples/by_feature/megatron_lm_gpt_pretraining.py",
                "--config_name ", "gpt2-large",
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```
jupyter debug:  
在设置里，搜索 debug jupyter 可以很容易看到
![image](https://github.com/jiashenggu/note/assets/32376856/b6612658-4f97-4acf-b501-7780ac78a798)



text automatically change lines
```
{
    "notebook.output.wordWrap": true:
}
```
workspace settings

```
{
    "workbench.editor.wrapTabs": true
}
```

vscode在最上面显示当前类：  
打开 VSCode。  
进入设置，你可以通过点击左下角的齿轮图标然后选择“设置”，或者直接使用快捷键 Ctrl + , (Windows/Linux) 或 Cmd + , (macOS)。  
在设置搜索框中，输入“Breadcrumbs”进行搜索。  
确保勾选了“Editor › Breadcrumbs: Enabled”选项，以启用 Breadcrumbs 功能。  
你还可以调整其他相关设置，如“Editor › Breadcrumbs: FilePath”和“Editor › Breadcrumbs: SymbolPath”来控制显示的信息详细程度。  


import readline
input() 可以用方向键

