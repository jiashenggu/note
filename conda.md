# 【解决】pip/conda bad interpreter: /home/username/anaconda/bin/python: no such file or directory
去以下位置改第一行#后的地址，envs里同理
```
vim ~/anaconda3/bin/pip
```

```
vim ~/anaconda3/bin/conda
```

```
vim ~/anaconda3/bin/uvicorn
```

在envs里也同理

# How to Relocate Your Anaconda or Miniconda Installation on Linux
https://www.earthinversion.com/utilities/How-to-Relocate-Your-Anaconda-or-Miniconda-Installation-on-Linux/

```python
find . -type f -exec sed -i 's|/path/to/old/conda|/new/path/to/conda|g' {} +
```
