

先回顾一下Megtron的SP(Sequence Parallelism)操作，SP完成sequence维度的并行，覆盖操作包括LayerNorm、Dropout、FC，但不能切分self-attention模块。如下图所示在SP/TP的组合中案例中，self-attention计算前聚合（all-gather）了sequence的内容。

![image](https://github.com/user-attachments/assets/2bb50cc9-e1ee-416a-8caa-330da976c968)


https://www.zhihu.com/collection/910140491

sequence parallel/context parallel: https://www.mltalks.com/posts/1017283893/
