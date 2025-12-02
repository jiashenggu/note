# uv
`--active` 就是告诉 uv：

> **“别用项目目录下的 `.venv`，直接用当前 shell 里已经激活的那个虚拟环境。”**

------------------------------------------------
出现背景  
- 你手动用 `uv venv test`（或 `python -m venv test`）建了一个环境并 `source test/bin/activate` 了。  
- 此时 shell 的 `VIRTUAL_ENV` 指向 `.../test`，可项目根目录下还有默认的 `.venv`。  
- 默认情况下 `uv sync` / `uv run` **永远只认 `.venv`**，于是出现  
  “shell 在 test，uv 却往 .venv 里装” 的错位。

------------------------------------------------
加 `--active` 后  
```bash
uv sync --active      # 装到当前激活的 test
uv run --active python script.py
```
uv 会**忽略 `.venv`**，把包装进你正在用的 `test`，同时把锁文件里对应的包一次性同步过去。

------------------------------------------------
一句话  
`--active` = **“别自己挑目录，我让你装哪你就装哪”** ——也就是当前 shell 里已激活的那个虚拟环境。


# label-studio
https://github.com/HumanSignal/label-studio/issues/7202
```bash
cd web
# Make sure yarn is installed
yarn install --forzen-lockfile
yarn build
```

If you need development, install yarn and follow web/README.md to complie frontend.

# 机器里**同时**装了两个“主”metapackage：

- `nvidia-driver-575`（服务器版，575.64.03）
- `nvidia-driver-580`（普通版，580.95.05）

**只能有一个**真正生效，否则内核模块和用户空间的版本就会对不上——这正是你之前看到的  
“Failed to initialize NVML: Driver/library version mismatch” 的根本原因。

------------------------------------------------
1. 先确认当前**实际加载**的是哪一套内核模块
------------------------------------------------
```bash
cat /proc/driver/nvidia/version | head -1
```
输出里能看到类似 `Kernel Module 575.64.03` 或 `580.95.05`，那就是**正在运行**的版本。

------------------------------------------------
2. 保留你要用的那一套，把另一套卸干净
------------------------------------------------
假设你想用 **580**（新特性多，桌面/计算通用）：

```bash
# 1. 完全删掉 575 系列
sudo apt remove --purge '^nvidia-.*-575.*' '^libnvidia.*-575.*' '^nvidia-driver-575.*'

# 2. 确保 580 的内核模块已编译并加载
sudo dkms autoinstall
sudo modprobe -r nvidia-drm nvidia-modeset nvidia-uvm nvidia   # 如有占用先卸载
sudo modprobe nvidia

# 3. 重启验证（可选但最稳）
sudo reboot
```

如果你想**回退到 575**（服务器场景更稳），就把上面命令里的 575 和 580 互换即可。

------------------------------------------------
3. 清理残留配置文件（可选）
------------------------------------------------
```bash
sudo apt autoremove --purge
sudo dpkg -l | grep nvidia   # 再检查一遍，只剩 580 系列即可
```

------------------------------------------------
4. 验证
------------------------------------------------
```bash
nvidia-smi
```
不再报 “Driver/library version mismatch”，且版本号统一为 580.95.05（或你保留的那套），就说明冲突解决。

# line_profiler
[https://blog.csdn.net/weixin_43135178/article/details/117352757](https://blog.csdn.net/weixin_43135178/article/details/117352757)
