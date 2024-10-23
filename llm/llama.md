## hidden_act silu
The SiLU function is also known as the swish function.

$silu(x)=x∗σ(x)$,

where σ(x) is the logistic sigmoid.
![image](https://github.com/user-attachments/assets/e868497e-75ba-4d19-b634-461d9cdcf094)
#### 关于正则化效果
x轴越靠近左半轴，纵坐标的值越小，甚至接近于0，如果x值是-10，那么经过激活之后的值接近于0，那么就可以一定程度上过滤掉一部分信息，起到正则化的效果。

它的全局最小值约为-0.28。SiLU 的一个吸引人的特点是它具有自稳定特性。
导数为零的全局最小值在权重上起到“软地板”的作用，作为隐式正则化器，抑制了大数量权重的学习。
```python
def swish(x):
    return x * F.sigmoid(x)

# used as class:
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)
```

#### SwiGLU和SiLU的主要区别如下：

结构差异：

$SiLU (Sigmoid Linear Unit): f(x) = x * sigmoid(x)$

$SwiGLU: f(x, y) = x * SiLU(y) = x * (y * sigmoid(y))$

计算方式：
SiLU是单输入激活函数

SwiGLU是门控机制，需要两个输入向量，一个作为主要输入(x)，另一个作为门控信号(y)



https://blog.csdn.net/Roaddd/article/details/114793441
