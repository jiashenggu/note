如何理解直接采样没有梯度而重参数之后就有梯度呢？其实很简单，比如我说从
$\mathcal{N}\left(z;\mu_{\theta},\sigma_{\theta}^2\right)$
中采样一个数来，然后你跟我说采样到5，我完全看不出5跟 $θ$有什么关系呀（求梯度只能为0）；但是如果先从
$\mathcal{N}\left(z;0,1\right)$
中采样一个数比如
0.2
，然后计算
$0.2 \sigma_{\theta} + \mu_{\theta}$
，这样我就知道采样出来的结果跟 $θ$的关系了（能求出有效的梯度）。



https://spaces.ac.cn/archives/6705
