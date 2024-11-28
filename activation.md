[Rethinking Softmax: Self-Attention with Polynomial Activations](https://arxiv.org/abs/2410.18613)

论文首先从理论上分析了softmax注意力机制的有效性，指出其成功并非仅仅因为能够生成一个概率分布，而是因为它在训练过程中隐式地对注意力矩阵的Frobenius范数进行了正则化。
论文提出了一个理论框架，用以分析softmax如何控制注意力矩阵的范数，防止权重过大或过小，从而维持训练过程中的稳定性。
