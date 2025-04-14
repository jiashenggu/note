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

## 查看当前操作系统是ubuntu还是centos
```bash
cat /etc/issue
```

## linux下查看当前用户、所有用户的方法
直接查看/etc/passwd 文件后面第二个冒号的值大于1000时，这个就是一个用户
```bash
/etc/passwd
```
[https://blog.csdn.net/tsummer2010/article/details/104427776](https://blog.csdn.net/tsummer2010/article/details/104427776)
