# torch.argsort()
https://blog.csdn.net/weixin_42052231/article/details/110941232
返回的是一个排序好的列表值的索引。也就是根据所给的索引，依次取出元素，就会得到一个排序好的tensor。

# 统计模型参数
```python
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

```
# 通过设置PYTORCH_CUDA_ALLOC_CONF中的max_split_size_mb解决Pytorch的显存碎片化导致的CUDA:Out Of Memory问题

https://blog.csdn.net/MirageTanker/article/details/127998036

TLDR：对于显存碎片化引起的CUDA OOM，解决方法是将PYTORCH_CUDA_ALLOC_CONF的max_split_size_mb设为较小值。

由于默认策略是所有大小的空闲Block都可以被分割，所以导致OOM的显存请求发生时，所有大于该请求的空闲Block有可能都已经被分割掉了。而将max_split_size_mb设置为小于该显存请求的值，**会阻止大于该请求的空闲Block被分割**。如果显存总量确实充足，即可保证大于该请求的空闲Block总是存在，


进一步优化
有了上述结论，就可以导出最优设置策略：将max_split_size_mb设置为小于OOM发生时的显存请求大小最小值的最大整数值，就可以在保证跑大图的可行性的同时最大限度照顾性能。
观测OOM请求最小值是6.18GB，所以最终选择了6144作为最优设置：
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144
