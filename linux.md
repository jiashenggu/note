### wsl
需要安装一些来接近真是linux系统
WSL 官方已经把 headers 打包成 linux-headers-virtual
```
sudo apt update
sudo apt install build-essential linux-headers-virtual python3-dev
```

## ubuntu改任务栏
https://blog.csdn.net/qq_36711448/article/details/134679319


### ✅ 什么是 X11？

**X11**（也叫 **X Window System**）是 Linux 系统中**负责图形界面显示的老牌协议/系统**。它管理窗口、鼠标、键盘输入等，是 GNOME、KDE 这些桌面环境“画”出界面的基础。

- 你可以把 X11 想象成“画布”，GNOME 是“画家”。
- 它是 **目前 Ubuntu 默认的图形会话之一**（另一个是 Wayland）。

---

### ✅ 那 `Alt + F2 → r` 是什么原理？

这是 **GNOME Shell 的一个隐藏快捷键**，只在 **X11 会话**下有效。

#### 🔍 原理拆解：

| 步骤 | 作用 |
|------|------|
| `Alt + F2` | 打开 GNOME 的“运行命令”小窗口（一个小黑框） |
| 输入 `r` | 是 GNOME 的“重启 Shell”命令 |
| 回车 | GNOME Shell 会**不注销、不关闭程序**地重新加载自己 |

#### ✅ 效果：
- 就像“刷新桌面”一样，**界面卡了、图标丢了、扩展崩了**，都能靠它恢复。
- **不会关闭你打开的窗口或文件**，非常安全。

---

### ⚠️ 注意：
这个操作**只在 X11 下有效**！

如果你用的是 **Wayland**（Ubuntu 默认在某些显卡上用它），`r` 会无效，因为 Wayland 不允许 GNOME Shell 这么“热重启”。

---

### ✅ 如何确认你用的是 X11 还是 Wayland？

打开终端，输入：

```bash
echo $XDG_SESSION_TYPE
```

- 输出 `x11` → 你在用 X11，`Alt+F2 → r` 有效。
- 输出 `wayland` → 你在用 Wayland，需要**注销再登录**来重启 Shell。

---

### 🧠 小结一句话：

> `Alt+F2 → r` 是 GNOME 在 X11 下的“软重启”秘籍，**不关机、不丢活、只刷新桌面**，是 Linux 用户的“Ctrl+Alt+Del”。

---

如果你以后遇到图标消失、扩展崩溃、面板卡死，**先试试这个**，比重启电脑快多了。
