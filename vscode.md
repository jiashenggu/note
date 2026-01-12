## win11 ssh

你需要创建一个 **SSH wrapper 脚本**，让 Cursor 通过这个脚本调用 WSL 的 SSH。

### 方法：创建 Windows 批处理脚本

1. 在 Windows 上创建文件 `C:\Users\scruple\wsl-ssh.bat`，内容：

```batch
@echo off
C:\Windows\System32\wsl.exe ssh %*
```

2. 然后在 Cursor 设置中：

```json
{
  "remote.SSH.path": "C:\\Users\\scruple\\wsl-ssh.bat"
}
```

这样 Cursor 调用 `wsl-ssh.bat -T -D 63100 gb200 ...` 时，实际执行的是 `wsl.exe ssh -T -D 63100 gb200 ...`，就正确了！

---

## 同时打开多个分支

在 VS Code 中，**一个窗口只能绑定一个 Git 分支**（即当前 `HEAD` 指向的分支）。  
所以“同时打开多个分支”并不是指在一个窗口里同时 checkout 多个分支，而是指：

---

### ✅ 可行的做法：开 **多个窗口**，每个窗口 checkout 一个分支
1. 把同一仓库克隆 **N 份**到不同文件夹（或用 `git worktree`，见下文）。  
2. 每个文件夹 `checkout` 到你要的分支。  
3. 用 VS Code 分别打开这些文件夹 → 得到 N 个窗口，每个窗口对应一个分支。  
4. 可把窗口拖成 **多列/多屏** 布局，实现“同时可见”。

---

### 🚀 最干净方案：git worktree（无需重复克隆）
一条命令即可在同一份仓库里创建第二个工作目录，且不互相污染：

```bash
# 在仓库根目录
git worktree add ../my-repo-branch-b branch-b
```

VS Code 里  
- 窗口 A 打开原目录（分支 A）  
- 窗口 B 打开 `../my-repo-branch-b`（分支 B）

用完随手删：

```bash
git worktree remove ../my-repo-branch-b
```

---

### ❌ 不可行的做法
- 在一个窗口里“同时显示”多个分支的文件树——VS Code 的 Source Control 面板只能反映当前分支。  
- 装任何插件也无法突破“一个文件系统目录只能处于一个 HEAD” 的 Git 限制。

---

### 小结
| 需求描述               | 推荐操作                              |
|------------------------|---------------------------------------|
| 同时看/改多个分支      | `git worktree` + 多窗口               |
| 快速对比两个分支文件   | 用 GitLens 的“Compare with Branch”    |
| 临时看另一个分支文件   | 命令面板 `Git: Checkout to...` 来回切 |

这样即可在 VS Code 里“实质地”同时打开多个 Git 分支。

## ssh 同一个ip的同一个端口重新链接需要：
```
ssh-keygen -f "/home/gear/.ssh/known_hosts" -R "[localhost]:8026"
```
# Plugins:
Live server: Launch a development local Server with live reload feature for static & dynamic pages

Bookmarks: Mark lines and jump to them

Gitlens: Supercharge Git within VS Code

Partial Diff: Compare (diff) text selections within a file, across files, or to the clipboard



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
