pip 自动安装的pytorch都是cpu

清华镜像源中的torch 只有CPU 版本的

pip和conda添加和删除镜像源

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
