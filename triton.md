denominator of softmax in triton is different from torch
```python
# torch
 x_exp.sum(1, keepdim=True)
# triton
qk = q * k
m_prev = m
m = tl.maximum(m, tl.max(qk, axis=1, keep_dims=True))
s = s*tl.exp2(log2_e*(m_prev-m)) + tl.sum(tl.exp2(log2_e*(qk-m)), axis=1, keep_dims=True)
```

作者：霸王手枪腿
链接：https://zhuanlan.zhihu.com/p/721477452
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

认真了解Triton算起来也有大半年了，简单梳理一下自己的一些思考。我认为Triton让自己与之前的AI Compiler不一样的技术哲学主要是三点。

#### 一 定位
之前的AI Compiler，不论是TVM还是MLIR，在定位上就希望自己能够在网络层级，端到端地完成代码生成。或者说，从它们的视角来看，AI Compiler的定位是输入网络结构，输出网络级别的可执行的文件。这种远大的理想至少在NV的GPU战场上很难说非常成功，核心原因是性能上难以匹敌调用算子库或者模板库的竞品。而**Triton的的定位很清楚，就是要做一个比CUDA更简单的DSL**（这发生在openai接手Triton之后），或者说，它的定位就是让用户写Kernel更简单。这个定位延展下去，就是为什么Triton要认真打造一套足够好用的Python DSL，以及相应的功能（Autotune等）。

#### 二 放弃硬件中立
Triton的原始论文的发表是在2019年，如果把这篇论文和同时期的MLIR的社区对于Linalg的讨论放在一起比较，就看的很清楚了，**Triton的目标就是只考虑在CUDA生态下的优化，直接考虑要解决的问题就是Pre Fetching，访存合并，Shared Memory的分配与同步，这些显然都是CUDA生态下Kernel优化的要点**；而MLIR社区考虑的Linalg（严格来说当时好像还没取Linalg这个名字）的Core Guarding Principles包括“Transformations and Simplicity First”，“Composable and Declarative”等。基于这个思路，Triton大概也发现，Kernel的优化最难搞定的还是tiling，而流水编排，数据分布这些手写算子时候不太好搞的事情翻到适合交给Compiler，于是就形成了现在大家看到的编程模型。后来的发展出现了两个有意思的点，第一个点是Triton的编程模型大家发现反而好像是各种硬件的公约数，如何划分数据块，load到片上再store回去在各种硬件架构下都是难搞的问题；第二个点是Hopper架构相比于之前的N卡架构，引入很多编程难题，试图保持硬件中立的TVM和MLIR看起来都难以适应这些改变，而Triton却通过简单增加了一层NVGPU的IR基本搞定了这个问题。

#### 三 保持简洁
Triton基于MLIR，但我们必须把MLIR分成两部分讨论，一部分是基于tablegen的Compiler的基础设施，一部分是MLIR试图通过层次化的dialect复用来形成一个生态。应该说Triton基于的是第一部分，而几乎彻底地放弃第二部分。Triton对于MLIR生态的dialect的使用是非常克制的，本身自己的dialect也非常的简洁，不考虑Hopper架构引入的NVGPU的IR的话，**可以直接理解为从TritonGPU这一层IR直接非常陡峭地lower到汇编层**。而TritonGPU的实现也非常有意思，Triton尽可能使用attribute的标记而不是引入新的IR来解决问题。比如访存合并，按照之前（从多面体优化传承下来）的思路，应该是往IR里面加个一两层循环之类的操作，而Triton的做法则是标记诸如“sizePerThread”之类的attribute，记录访存合并的策略并验证其合法性，然后通过陡峭地lowering直接生成代码。从结果来看，Triton的简洁化是成功的，60K的代码行数相对于Triton取得的成绩来说是一个非常小的数字。当然简洁能够得以实现的重要前提也是前面说的硬件中立。这三点设计哲学从根本来说都来自于OpenAI对于Triton的诉求，而Triton确实也做到了，在实际应用中能体现Triton价值的核心场景大概是大模型中的各种Attention变体。而时至今日，大家对于Triton的诉求却也在变化，Triton团队未必能满足只做一款方便OpenAI写Kernel方便的软件，而硬件厂商们也希望Triton能够成为对抗CUDA生态垄断的新的战场。诉求的改变意味着设计哲学的改变，而设计哲学改变了，那之前的竞争力与独特性是否还能存在，也尚未可知。在软件迭代的进程中，最难回答的问题也许还是：多大的野心才是最合适的？


对Triton的一些理解 - 霸王手枪腿的文章 - 知乎
https://zhuanlan.zhihu.com/p/721477452
