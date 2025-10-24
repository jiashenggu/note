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
