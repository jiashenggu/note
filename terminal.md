查看后台运行的进程。
```bash
ps aux | grep [进程名]
```
找到卡号对应的进程应该可以
```bash
fuser -v /dev/nvidia*
```
