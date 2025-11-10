# ssh
同一个ip的同一个端口重新链接需要：
```
ssh-keygen -f "/home/gear/.ssh/known_hosts" -R "[localhost]:8026"
```
# Plugins:
Live server: Launch a development local Server with live reload feature for static & dynamic pages

Bookmarks: Mark lines and jump to them

Gitlens: Supercharge Git within VS Code

Partial Diff: Compare (diff) text selections within a file, across files, or to the clipboard

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


## cursorLineStartSelect
VS Code 里“Shift+Home”不起作用，99% 都是下面三类原因。请按顺序排查，基本都能定位到问题。

--------------------------------
1. 快捷键被别的软件全局拦截  
   • 笔记本键盘 Fn 模式：  
     – Home 实际要 Fn+←，于是真正的组合是 **Shift+Fn+←**（Lenovo、Dell 很多机型如此）。  
   • 外接键盘/游戏键盘：  
     – 检查键盘驱动（iCUE、Razer Synapse、Logitech G Hub）里是否把 Home 或 Shift+Home 绑成了宏/灯光。  
   • 输入法/翻译软件：  
     – 搜狗、QQ、有道、GoldenDict 等常把 Shift+Home 登记为“取词”或“划译”，在它们的设置里关掉或换成别的组合。

   **验证方法**：  
   把 VS Code 最小化，打开记事本，按 Shift+Home。  
   – 记事本里也不能选到行首 → 问题在系统/键盘/输入法，与 VS Code 无关。  
   – 记事本正常，只有 VS Code 不行 → 继续看第 2 步。

--------------------------------
2. 快捷键在 VS Code 里被改掉了  
   1. 打开命令面板 `Ctrl+Shift+P` → 输入并执行 `Preferences: Open Keyboard Shortcuts (JSON)`。  
   2. 在打开的 keybindings.json 里搜索 `cursorLineStartSelect`（官方默认命令 ID）。  
      如果看到类似  
      ```json
      { "key": "shift+home", "command": "-cursorLineStartSelect" }
      ```  
      或者把 shift+home 绑给了别的命令，就说明被覆盖。  
   3. 把冲突行删掉或改成你想要的即可；改完保存立即生效，无需重启。

   **图形界面检查**（可选）：  
   `Ctrl+K Ctrl+S` 打开键盘快捷键列表 → 右上角搜索框输入 `shift+home` → 看是否出现黄色警告“已有冲突”。

--------------------------------
3. 你正在用 Remote/WSL/虚拟机，按键被终端先吃掉  
   • 远程窗口（SSH、WSL、Docker、Codespaces）里，如果光标落在“终端”面板，Shift+Home 会被 shell 先处理，VS Code 收不到。  
   • 虚拟机（VMware/VirtualBox/Parallels）默认把 Home/End 映射成 Host 键，也会失效。  
   **解决**：点击一下文件编辑器区域，让焦点回到编辑器再试；或在虚拟机设置里把 Host 键换成别的。

--------------------------------
快速自检清单（30 秒）
1. 记事本里 Shift+Home 能否选到行首？  
   ❌ 不行 → 先解决系统/键盘/输入法。  
2. 只在 VS Code 里不行 → 打开 keybindings.json 搜 `cursorLineStartSelect`，有减号或冲突就删掉。  
3. 用 Remote/终端/虚拟机 → 先把焦点切回编辑器再试。

按上面三步基本都能恢复。如果还有问题，把 keybindings.json 里关于 shift+home 的几行贴给我，我帮你看具体冲突来源。
