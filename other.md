# label-studio
https://github.com/HumanSignal/label-studio/issues/7202
```bash
cd web
# Make sure yarn is installed
yarn install --forzen-lockfile
yarn build
```

开发web/libs/editor/src/代码，需要，yarn dev和yarn watch都开着

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

# line-profiler
[https://blog.csdn.net/weixin_43135178/article/details/117352757](https://blog.csdn.net/weixin_43135178/article/details/117352757)
