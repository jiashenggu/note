deepseekmoe: 
1. more experts, smaller experts
2. shared experts

   
deepseek v3:
1. auxiliary free, add bias in router
2. a multi-token prediction training objective

MOE模型的过去、现状和未来是怎样的？ - 甜菜欣欣的回答 - 知乎
https://www.zhihu.com/question/664040671/answer/62344159797

1. 你们发现混合专家模型需要更低的学习率和更大的批大小，因为专家是稀疏激活的。每个专家接收到的 token 更少，也就是在训练时批大小更小，所以它就需要更大的批大小。
2. 负载平衡是为了确保 MoE 可以把 token 平均分配给各个专家。我们发现负载平衡越少，效果反而越好。
   
experiments summary

1. **Gating Mechanisms and Auxiliary Loss**:
   - **Baseline**: The softmax gating mechanism with expert-level auxiliary loss (aux_loss) was the starting point for comparison.
   - **Sigmoid Gating + Auxiliary Loss-Free Training**:
     - Found to be more stable compared to the baseline.
     - Achieved better performance on MMLU (Massive Multitask Language Understanding) benchmark.
   - **Sigmoid Gating Only**:
     - Simply switching from softmax to sigmoid gating yielded results similar to the baseline.
     - However, combining sigmoid gating with auxiliary loss-free training showed distinct improvements over baseline methods.

2. **Why MLA Outperforms MHA**:
   - When transitioning back to the **Multi-Head Attention (MHA)** stage, improvements in performance can be attributed to increasing the number of heads (`num_head`) or the dimension of each head (`head_dim`). This scaling boosts capacity and expressivity.

---

### Details on Model and Configuration:

- **Aria Model Architecture**:
  - **Vision Component**: Aria 0.4B.
  - **Text Component**: Aria 3.5B.
  - Vision Transformer (ViT) weights initialized using the **SigLIP-SO400M model**.

- **Performance and Data Processing**:
  - Aria supports **22A/247B tokens**, processing a maximum of **195 images**.
  - For video processing: 
    - Each frame requires approximately **2 seconds** of computation.
    - A **6-minute 30-second video** can be fully processed within the framework.

employ z-loss to stabilize training.
![442cfa0a-60ae-4487-b2c4-11837eb93e35](https://github.com/user-attachments/assets/f49bbcb1-55fe-467a-86af-0502f204fdc7)

![5009d76e-354e-416f-af32-89d860ddf321](https://github.com/user-attachments/assets/8cb5f75f-7559-46e3-8cea-d1af044f71eb)

Complementary Sequence-Wise Auxiliary Loss主要是为了解决单个输入序列内部的极端负载不均衡。 
$T$是输入序列的总长度， 
$s_{i, t}'$代表归一化的输入序列和各个专家的亲和力， 
$P_i$代表第i个专家和序列内的每一个token的亲合度均值，代表了该专家和序列的整体亲合度， 
$f_i$代表第i个专家在该序列预测过程中的选中频率， 
$\alpha$为较小的常数超参数。

可以看到， 
$f_iP_i$代表了第i个专家的负载强度，当部分专家反复在topk中被选中的时候， 
$L_{Bal}$会增大，即体现了对负载不均衡的惩罚。

最后，在通信方面，DeepSeek-V3 使用限制路由机制来限制训练期间的通信成本，即每个token最多被发送到M个算力节点，这些节点是根据分布在每个节点上的专家的最高 
个 $\frac{K_{r}}{M}$亲和力分数之和来选择的。在此约束下，deepseek v3的 MoE训练框架几乎可以实现完全的计算-通信重叠。DeepSeek-V3 的负载均衡策略保证了在训练期间不会丢弃任何token。

https://zhuanlan.zhihu.com/p/15797610465
