### 注意：清华镜像源中的torch只有CPU版本的，因此pip自动安装的pytorch都是cpu

# 单次
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple torch
```

# 配置
```bash
vi ~/.pip/pip.conf
```
或者在
pip源配置文件可以放置的位置：
```
Linux/Unix:

/etc/pip.con

~/.pip/pip.conf

~/.config/pip/pip.conf

Mac OSX:

~/Library/Application Support/pip/pip.conf

~/.pip/pip.conf

/Library/Application Support/pip/pip.conf

Windows:

%APPDATA%\pip\pip.ini

%HOME%\pip\pip.ini

C:\Documents and Settings\All Users\Application Data\PyPA\pip\pip.conf (Windows XP)

C:\ProgramData\PyPA\pip\pip.conf (Windows 7及以后)
```
```bash
[global]
no-cache-dir = true
index-url = https://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://pypi.ngc.nvidia.com
trusted-host = mirrors.aliyun.com
```
```bash
[global]
index-url = http://pypi.douban.com/simple #豆瓣源，可以换成其他的源
trusted-host = pypi.douban.com            #添加豆瓣源为可信主机，要不然可能报错
disable-pip-version-check = true          #取消pip版本检查，排除每次都报最新的pip
timeout = 120
```

## 常用源
阿里云：http://mirrors.aliyun.com/pypi/simple/

豆瓣 ：http://pypi.doubanio.com/simple/

清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/

中国科学技术大学：http://pypi.mirrors.ustc.edu.cn/simple/

华中理工大学：http://pypi.hustunique.com/

山东理工大学：http://pypi.sdutlinux.org/

## 命令行pip和conda添加和删除镜像源

一、pip

1、添加源

比如添加清华源
https://pypi.tuna.tsinghua.edu.cn/simple
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
如果set多次，似乎只能保存最后一次set的镜像源。

2、删除源
```
pip config unset global.index-url
```
3、查看现在用的哪个源
```
pip config list
```

二、conda

1、添加源

比如清华源：
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
2、删除源
```
conda config --remove-key channels
```
3、查看现在用的是哪些源
```
conda config --show channels
```
也可以使用
```
conda config --show-sources
```

