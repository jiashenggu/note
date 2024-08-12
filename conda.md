# 【解决】pip/conda bad interpreter: /home/username/anaconda/bin/python: no such file or directory

```
vim ~/anaconda3/bin/pip
```

```
vim ~/anaconda3/bin/conda
```

在envs里也同理

# How to Relocate Your Anaconda or Miniconda Installation on Linux
https://www.earthinversion.com/utilities/How-to-Relocate-Your-Anaconda-or-Miniconda-Installation-on-Linux/

```python
find . -type f -exec sed -i 's|/path/to/old/conda|/new/path/to/conda|g' {} +
```
