由于整个前向计算的流程中出现了argmin，所以梯度无法回传到encoder，这意味着我们无法优化encoder。此时常见的手段是Gumbel Softmax，但Gumbel Softmax的效果通常也是次优的，所以作者巧妙地借助了Straight-Through为VQ-VAE设计了更好的梯度。
具体来说，VQ-VAE利用了深度学习框架基本上都自带的stop_gradient（即公式中的sg）函数来自定义梯度，所有经过sg的输入，都会保持同样的输出，但梯度被强迫为零。

FSQ只是用来替代VQ-VAE中的“VQ”的，它的离散化思路非常非常简单，就是“四舍五入”。


![image](https://github.com/user-attachments/assets/cbead4fb-5a6e-40b1-a132-06e3d4aff394)

从形式上来看，不管是从参数量、几何直观或者表达能力来看，其实FSQ都不如VQ，但为什么FSQ有机会取得比VQ更好的结果呢？笔者认为有两方面的原因。

第一个原因，是encoder和decoder太强。虽然FSQ本身弱一些，但是encoder和decoder都足够强了，所以基于神经网络的万能拟合能力假设，FSQ相对于VQ的劣势，完全可以在encoder和decoder中弥补过来。而在
$K=L^d$
的设定下，两者的离散化程度都是一样的，也就是说encoder与decoder之间的“信息瓶颈”是一样的，因此FSQ本身的问题就显得微不足道了。

第二个原因，是VQ的“队友”（梯度）太弱。VQ的经典问题是编码表坍缩：当编码表增大时，编码表并没有被充分利用起来，反而由于恶性竞争导致编码表聚集到一块了，经典表现就是一个5000的编码表，最终效果还不如500的编码表。

如果单纯是作为“图像的Tokenzier”，那么FSQ或许已经可以取代VQ，但这并不意味着任意场景下VQ都可以被FSQ取代。

https://papers.cool/arxiv/2309.15505

https://spaces.ac.cn/archives/9826
