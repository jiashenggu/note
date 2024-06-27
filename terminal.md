查看后台运行的进程。
```bash
ps aux | grep [进程名]
```
找到卡号对应的进程
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

