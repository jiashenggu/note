# LayerNorm 的实现
LayerNorm 对输入的每个样本进行归一化，维度是按最后一个轴进行的。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * x_normalized + self.beta
```
# RMSNorm 的实现
RMSNorm 是 LayerNorm 的一种变体，它不计算均值，而是基于输入向量的均方根值进行归一化。
```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.gamma * x_normalized
```
# 主要区别
## 归一化依据：

LayerNorm 会计算均值和方差，并对每个维度进行归一化。
RMSNorm 不计算均值，仅基于均方根值进行归一化。
## 参数数量：

两者参数量一致，都是 gamma（缩放参数），而 LayerNorm 额外还有 beta（偏移参数）。
## 计算效率：

RMSNorm 计算成本更低，因为它省略了均值计算。
