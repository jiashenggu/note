### 注意：清华镜像源中的torch只有CPU版本的，因此pip自动安装的pytorch都是cpu
# 指定cuda版本
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://mirrors.aliyun.com/pytorch-wheels/cu118 -i https://mirrors.aliyun.com/pypi/simple
```
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
默认：https://pypi.org/simple

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

# 在 Linux 环境下修改 pip 的缓存地址通常涉及修改 pip 的配置文件或者环境变量。在这里，我将介绍两种方法来实现这一目的：
 
### 方法一：修改 pip 配置文件
 
1. 打开 pip 的配置文件 `pip.conf` 或 `pip.ini`，该文件通常位于以下位置之一：
   - 用户级配置文件：`~/.config/pip/pip.conf` 或 `~/.pip/pip.conf`
   - 全局配置文件：`/etc/pip.conf` 或 `/etc/pip/pip.conf`
 
2. 如果文件不存在，则创建该文件。在配置文件中添加如下内容来修改 pip 的缓存地址：
   ```
   [global]
   cache-dir = /path/to/your/cache/directory
   ```
 
3. 将 `/path/to/your/cache/directory` 替换为你想要设置的新缓存目录路径。
 
### 方法二：设置环境变量
 
1. 打开 shell 配置文件，如 `~/.bashrc`、`~/.bash_profile` 或 `~/.zshrc`，根据你使用的 shell 不同而有所不同。
 
2. 在文件末尾添加以下行来设置 pip 缓存目录的环境变量：
   ```
   export PIP_CACHE_DIR=/path/to/your/cache/directory
   ```
 
3. 保存文件并执行以下命令使更改生效：
   ```bash
   source ~/.bashrc
   ```
 
4. 将 `/path/to/your/cache/directory` 替换为你想要设置的新缓存目录路径。
 
### 原理：
 
- 当你运行 `pip` 命令时，pip 会首先查找配置文件中的缓存目录设置，如果找到了就使用该设置的缓存目录；如果没有找到，pip 将会使用默认的缓存目录。
- 通过修改配置文件或设置环境变量，你可以指定 pip 使用的缓存目录，从而改变 pip 在下载和缓存包时使用的目录。
 
通过这两种方法中的任何一种，你可以修改 pip 的缓存地址，使其使用你指定的缓存目录来存储下载的包，而不是使用默认的缓存目录。
