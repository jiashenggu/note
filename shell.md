## show git branch in bash

把颜色加回来即可，下面给出两种常用做法，你挑一条粘到 `~/.bashrc` 里（放在 `source ~/.git-prompt.sh` 之后）再 `source ~/.bashrc` 就好了。

------------------------------------------------
1. 最简单：把原来的彩色 PS1 捡回来  
先注释掉或删掉我那行纯文本的 `export PS1=...`，然后在 `~/.bashrc` 末尾加：
```bash
curl -Lo ~/.git-prompt.sh \
  https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
```
```bash
# 保留系统原来的彩色提示符，只在末尾追加分支名
source ~/.git-prompt.sh
export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]$(__git_ps1 " \[\033[01;31m\](%s)\[\033[00m\]")\$ '
```

（这是 Debian/Ubuntu 默认的彩色模板，红括号里放分支名。）

------------------------------------------------
2. 自己拼一个彩色 PS1  
如果你想完全自定义，可写成：

```bash
source ~/.git-prompt.sh
# 绿色用户@主机 : 蓝色路径 红色(分支) 白色$
export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[1;31m\]$(__git_ps1 " (%s)")\[\e[0m\]\$ '
```

------------------------------------------------
3. 让 `__git_ps1` 自己带颜色  
`git-prompt.sh` 还提供了变量，可让分支名高亮而不用手动写转义：

```bash
source ~/.git-prompt.sh
GIT_PS1_SHOWCOLORHINTS=1        # 启用内置颜色
# 必须选用 \w 这类提示串，并且把 __git_ps1 放在最后
export PS1='\u@\h:\w$(__git_ps1 " (%s)")\$ '
```

（注意：这个变量**只在** `PROMPT_COMMAND` 模式或 `__git_ps1` 放在末尾时生效，详情看 `git-prompt.sh` 头部注释。）

------------------------------------------------
一句话  
**“没颜色”是因为 PS1 被覆盖成纯文本了；把颜色转义序列加回去，或者打开 `GIT_PS1_SHOWCOLORHINTS`，就又有颜色了。**


`set -euxo pipefail` 是 Bash 脚本中常用的组合命令，用于增强脚本的健壮性和调试能力。具体作用如下：

---

### 1. `-e`（`errexit`）
- **作用**：当任意命令执行失败（返回非零状态码）时，立即终止脚本。
- **示例**：
  ```bash
  # 脚本会在此处终止，因为 ls 一个不存在的文件会失败
  ls non_existent_file
  echo "这行不会执行"
  ```

---

### 2. `-u`（`nounset`）
- **作用**：当使用未定义的变量时，报错并终止脚本。
- **示例**：
  ```bash
  # 若未定义 $MY_VAR，脚本会在此处终止
  echo $MY_VAR
  ```

---

### 3. `-x`（`xtrace`）
- **作用**：打印每个执行的命令及其参数，便于调试。
- **示例**：
  ```bash
  # 终端会输出：+ echo 'Hello World'
  echo "Hello World"
  ```

---

### 4. `-o pipefail`
- **作用**：确保管道命令的退出状态是最后一个非零状态码（默认仅返回最后一个命令的退出状态）。
- **示例**：
  ```bash
  # 假设 command1 失败，command2 成功
  command1 | command2
  # 默认返回 command2 的退出状态（0），但启用 pipefail 后会返回 command1 的退出状态（非零）
  ```

---

### 组合效果
- **增强安全性**：避免脚本在错误状态下继续执行。
- **调试友好**：通过 `-x` 显示详细执行过程。
- **管道健壮性**：严格捕获管道中的任何错误。

---

### 常见用法
```bash
#!/bin/bash
set -euxo pipefail

# 脚本内容
echo "Start"
your_command_here
```

---

### 注意事项
- 若需忽略某条命令的失败，可使用 `|| true`：
  ```bash
  # 即使 command 失败，脚本仍会继续执行
  command || true
  ```
- 临时禁用选项：使用 `set +<option>`（如 `set +x` 关闭调试）。


# Shell 历史前缀搜索

------------------------------------------------
bash（≥ 4.0，系统自带的一般都够）
------------------------------------------------
1. 编辑 ~/.inputrc（没有就新建）  
2. 把下面 4 行粘进去：

```
# 上下键：以“已输入部分”为前缀来回翻
"\e[A": history-search-backward        # ↑
"\e[B": history-search-forward         # ↓

# 可选：PgUp/PgDn 在所有历史里继续翻（不用也可）
"\e[5~": history-search-backward
"\e[6~": history-search-forward
```

3. 保存后执行  
   bind -f ~/.inputrc  
   或者重开一个终端即可。

用法示例  
终端里输入  
  git ⏎  
再按 ↑，就会逐条翻出以前执行过的“git …”命令；按 ↓ 就向“未来”翻。  
如果先输入  
  docker c  
再按 ↑，就只会翻出“docker container …”“docker compose …”这类以“docker c”开头的记录。

------------------------------------------------
zsh（oh-my-zsh 用户也一样）
------------------------------------------------
zsh 自带这个功能，只要打开两个选项：

```
# 写到 ~/.zshrc 里
autoload -Uz up-line-or-beginning-search down-line-or-beginning-search
zle -N up-line-or-beginning-search
zle -N down-line-or-beginning-search

# 把上下箭头绑定到这两个小部件
bindkey '^[[A' up-line-or-beginning-search
bindkey '^[[B' down-line-or-beginning-search
```

保存后  
  source ~/.zshrc  
用法跟 bash 完全一样：先敲前缀，再按 ↑↓ 即可在匹配里来回翻。

------------------------------------------------
老系统 bash 3.x 的 Fallback
------------------------------------------------
如果机器太老，readline 版本低，没有 history-search-backward，可以退而求其次用：

```
"\e[A": previous-history
"\e[B": next-history
```

这样至少能让 ↑↓ 跳过“当前匹配”后继续在完整历史里翻，不至于直接断掉。

------------------------------------------------
小结
------------------------------------------------
一句话：把“↑↓”绑定到 *-history-search-* 系列函数，就能实现“先输入前缀，再用上下键只翻匹配项”的效果，比 Ctrl-r 更直观、更快速。


## scp with regex
```bash
ssh myuser@remote.example.com 'ls -1 /var/log/app/*.log' | xargs -I@ scp myuser@remote.example.com:@ .
```
```bash
ls -1 ~/backups/*_backup.tar.gz | xargs -I {} scp {} myuser@remote.example.com:/mnt/backups/
```

## 让终端里**所有**输出都自动带上时间戳”的做法，按“改动范围”从小到大排列

------------------------------------------------
1. 只想让 **Python 代码里 print 带时间**  
（最小侵入，老脚本不动，只要在最上方加 4 行）

```python
import builtins, datetime, functools
_ts = functools.partial(datetime.datetime.now().strftime, '%F %T')
builtins.print = lambda *a, **k: builtins.print(f"[{_ts()}]", *a, **k)

# 之后任何 print(...) 都会变成：
# [2025-10-24 14:38:07] hello world
```

------------------------------------------------
2. 让 **当前 shell 会话里所有命令** 带时间  
（对已有脚本、第三方命令都生效，无需改代码）

bash / zsh  
```bash
# 临时生效，当前会话内
export PROMPT_COMMAND='printf "[%(%F %T)T] ";'
# 恢复：unset PROMPT_COMMAND
```

fish  
```fish
function fish_prompt
    printf '[%s] ' (date '+%F %T')
end
```

------------------------------------------------
3. 永久生效——把 2 写进启动脚本

bash  
```bash
echo 'export PROMPT_COMMAND='"'"'printf "[%(%F %T)T] ";'"'"''  >> ~/.bashrc
source ~/.bashrc
```

zsh  
```bash
echo 'export PROMPT_COMMAND='"'"'printf "[%(%F %T)T] ";'"'"''  >> ~/.zshrc
source ~/.zshrc
```

fish  
```fish
echo 'function fish_prompt; printf "[%s] " (date "+%F %T"); end' >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

效果  
每敲一次回车，终端就会在**下一行开头**先打印时间戳，再显示命令输出；  
如果命令自己产生多行输出，时间戳只出现在最前面一行——这已经能大致知道“什么时候跑完”。

------------------------------------------------
4. 想“**每一行输出**都带时间”——用 `ts`（moreutils）  
安装  
```bash
# Ubuntu / Debian
sudo apt install moreutils
# macOS
brew install moreutils
```

用法  
```bash
# 把任何命令管道给 ts，每行前面都会加时间
$ ls -l | ts '%F %T'
2025-10-24 14:38:10 total 24
2025-10-24 14:38:10 -rw-rw-r-- 1 user user  215 Oct 24 14:30 a.py
...

# 如果希望“秒”级精度
$ long_running_cmd | ts '%.s'
1698146290.123456 Starting...
1698146291.234567 Processing...
```

------------------------------------------------
5. 在 **Python 里重定向 stdout / stderr**，让**所有**输出（包括第三方库）都带时间  
```python
import sys, datetime, functools
ts = functools.partial(datetime.datetime.now().strftime, '%F %T')

class TimestampedFile:
    def __init__(self, raw):
        self.raw = raw
    def write(self, text):
        for line in text.splitlines(True):          # 保留换行
            self.raw.write(f'[{ts()}] {line}')
    def flush(self):
        self.raw.flush()

sys.stdout = TimestampedFile(sys.stdout)
sys.stderr = TimestampedFile(sys.stderr)

# 之后所有 print、logging、子进程输出都会带时间
```

------------------------------------------------
6. Windows PowerShell  
```powershell
# 当前会话
function prompt { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PS $PWD>" }

# 永久
# 把上面一行加到 $PROFILE 文件里
```

------------------------------------------------
一句话总结  
- **只改 Python**：用 1 或 5。  
- **不改代码，只让 shell 会话带时间**：用 2 / 3。  
- **要求“每行”都有时间戳**：用 4 的 `ts` 最干净。
#### 终端实时看输出，同时日志文件也保存一份
2>&1：把标准错误也一起重定向到标准输出  
tee：把输出同时写入文件 和 显示在终端
-a 参数告诉 tee 追加而不是覆盖。
```bash
python your_script.py 2>&1 | tee -a output.log
```
#### 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
#### 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

## 查看物理CPU个数
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

## 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo| grep "cpu cores"| uniq

## 查看逻辑CPU的个数
cat /proc/cpuinfo| grep "processor"| wc -l

## 查看CPU信息（型号）
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
 
## 查看内存信息
cat /proc/meminfo

## 如何查看Linux 内核
uname -a
cat /proc/version

## 查看当前操作系统是ubuntu还是centos
```bash
cat /etc/issue
```
 
## 查看机器型号（机器硬件型号）

dmidecode | grep "Product Name"

## 如何查看linux 系统版本
cat /etc/redhat-release
lsb_release -a
cat  /etc/issue
 
## 如何查看linux系统和CPU型号，类型和大小
cat /proc/cpuinfo

##如何查看linux 系统内存大小的信息，可以查看总内存，剩余内存，可使用内存等信息  
cat /proc/meminfo



## 删除指定文件名后缀
```bash
for file in *_42.jpg; do mv "$file" "${file/_42.jpg/.jpg}"; done
```
## 查看后台运行的进程。
```bash
ps aux | grep [进程名]
```

## 安装工具psmisc，为了使用fuser
```
apt-get update
apt-get install psmisc
```
### 找到卡号对应的进程
```bash
fuser -v /dev/nvidia*
```

fuser 是一个用于显示哪些进程正在使用指定文件、文件系统或套接字的工具。以下是 fuser -v /dev/nvidia3 命令的详细解释：

fuser：命令本身，用于显示文件、套接字或文件系统的使用情况。
-v：详细模式（verbose），提供更多的输出信息，包括进程ID（PID）、用户（USER）、访问模式（ACCESS）等。
/dev/nvidia3：这是你想要检查的设备文件。在这种情况下，是第三个NVIDIA GPU设备文件。
执行此命令后，你会看到类似如下的输出（假设有进程在使用该设备）：
```
                     USER        PID ACCESS COMMAND
/dev/nvidia3:        root       1234  F...  process_name
                     user1      5678  F...  process_name
```

USER：正在使用该文件或设备的用户。
PID：进程ID。
ACCESS：访问模式（例如，F表示文件打开，R表示读，W表示写）。
COMMAND：使用该文件或设备的进程名称。

## 杀死目标进程
```bash
kill -9 {processId}
```
# tar

```bash
tar -tvf archive.tar
```
-t 表示列出内容  
-v 表示详细模式，显示更多信息  
-f 指定要操作的tar文件名  

如果是压缩的tar文件（如.tar.gz或.tgz），可以使用：
```bash
tar -ztvf archive.tar.gz
```

要从tar文件中解压特定的文件夹，可以使用以下命令：
-x 表示解压  
-v 表示详细模式  
-f 指定tar文件名  
path/to/folder 是你想要解压的特定文件夹路径  

## 要直接查看tar文件中的顶层文件夹（即只列出第一层目录结构），我们可以使用一些额外的命令行工具结合tar命令。这里有几种方法：

1. 使用 `tar` 和 `grep`：

```bash
tar -tf archive.tar | grep -v '/' | sort -u
```

对于gzip压缩的文件：

```bash
tar -ztf archive.tar.gz | grep -v '/' | sort -u
```

这个命令的工作原理：
- `tar -tf` 列出所有文件
- `grep -v '/'` 排除包含 '/' 的行（即子目录）
- `sort -u` 排序并去重

2. 使用 `tar` 和 `awk`：

```bash
tar -tf archive.tar | awk -F/ '{print $1}' | sort -u
```

对于gzip压缩的文件：

```bash
tar -ztf archive.tar.gz | awk -F/ '{print $1}' | sort -u
```

这个命令的工作原理：
- `awk -F/ '{print $1}'` 只打印第一个斜杠前的内容
- `sort -u` 排序并去重

3. 使用 `tar` 和 `sed`：

```bash
tar -tf archive.tar | sed -e 's/\/.*//' | sort -u
```

对于gzip压缩的文件：

```bash
tar -ztf archive.tar.gz | sed -e 's/\/.*//' | sort -u
```

这个命令的工作原理：
- `sed -e 's/\/.*//'` 删除第一个斜杠及其后的所有内容


- `sort -u` 排序并去重

这些方法都能让你只看到tar文件的顶层目录结构。选择哪种方法主要取决于你的个人偏好和系统上可用的工具。



## linux下查看当前用户、所有用户的方法
直接查看/etc/passwd 文件后面第二个冒号的值大于1000时，这个就是一个用户
```bash
/etc/passwd
```
[https://blog.csdn.net/tsummer2010/article/details/104427776](https://blog.csdn.net/tsummer2010/article/details/104427776)
