[https://github.com/huggingface/trl/blob/main/docs/source/how_to_train.md](https://github.com/huggingface/trl/blob/main/docs/source/how_to_train.md)

## DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）
### 移除KL散度

### 提高上限：Clip-Higher
### 动态采样
过滤掉准确率为1和0的提示，确保每个批次中的提示都具有有效的梯度信号。实验表明，动态采样不仅提升了训练效率，还加速了模型的收敛。
### Token-Level策略梯度损失
传统的GRPO算法采用样本级损失计算，导致长响应中的token对整体损失的贡献较低。DAPO引入了Token-Level策略梯度损失，确保长序列中的每个token都能对梯度更新产生同等影响。这一改进不仅提升了训练稳定性，还避免了过长响应中的低质量模式。
### 过长奖励整形

## 大模型面经答案—强化学习：理论解释与讲解

https://developer.aliyun.com/article/1373044


## DPO公式

$$
-\log\sigma(\beta\log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)})
$$


转换为最大化

$$
\log\sigma(\beta\log \frac{\pi_{\theta}(y_w|x)}{\pi_{\theta}(y_w|x)} - \beta\log \frac{\pi_{ref}(y_l|x)}{\pi_{ref}(y_l|x)})
$$

从这个公式可以看出只要训练的策略生成正样本和负样本的概率的比值高于参考策略就可以降低损失。如果参考模型和训练策略生成正负样本的比值分别为 0.5/0.25=2， 0.3/0.1=3，此时损失也会下降，但是生成正样本的概率反而下降了。

既然生成正负样本的概率都降低了，那生成什么的概率提高了呢？答案是偏好数据集分布外的输出。模型可能会输出一些不包含在偏好数据集的奇奇怪怪的输出，比如问题是“意大利面应该拌什么”，数据集正样本是“蕃茄肉酱”，负样本是“油泼辣子”，DPO 优化后却可能回应“意大利面就应该拌42号混凝土”。所以 DPO 对数据集要求很高，高质量的偏好数据集上进行 DPO 对齐才能取得好效果。

[简单例子说明 DPO 为什么可能表现不好](https://zhuanlan.zhihu.com/p/18603295907)

## DPO、ReMax、PPO、GRPO到XDPO的解析：

https://zhuanlan.zhihu.com/p/679904863

## PPO:  
https://chatgpt.com/share/6783bb17-c23c-8008-931d-aa82b0e0a535

## Policy Loss和Value Loss
在强化学习中的 **Proximal Policy Optimization (PPO)** 算法中，**policy loss** 和 **value loss** 是两个主要的损失函数，分别对应策略网络（Policy Network）和价值网络（Value Network）的优化目标。

---

### 1. **Policy Loss**

**Policy Loss** 的目标是优化策略网络，使得策略能够选择更优的动作来最大化长期回报。它计算的是新策略和旧策略的相对变化，以及策略对优势函数的提升。

#### 核心思想
PPO 通过裁剪的方式，限制策略更新的幅度，避免策略变得过于激进，影响训练的稳定性。

#### 数学公式
假设：
- $\( \pi_\theta \)$ 是当前策略（新策略）。
- $\( \pi_{\text{old}} \)$ 是旧策略。
- $\( r_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)} \)$ 是策略概率比率。
- $\( A_t \)$ 是优势函数。

PPO 的 policy loss 定义为：
$$\[
\mathcal{L}_{\text{policy}} = -\mathbb{E} \left[ \min \left( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]
\]$$

- **第一项 \( r_t A_t \)**：
  直接使用策略概率比率 $\( r_t \)$ 放大或缩小优势函数 $\( A_t \)$。
  
- **第二项裁剪项**：
  将 $\( r_t \)$ 限制在 $\( [1-\epsilon, 1+\epsilon] \)$ 的范围内，防止策略更新过大。

- **取最小值**：
  PPO 选择较保守的更新幅度，避免过大的策略偏移。

#### 总结
**Policy Loss** 衡量策略网络输出动作概率与旧策略的偏差，同时考虑动作的优势值，目的是调整策略使其优先选择高优势动作。

---

### 2. **Value Loss**

**Value Loss** 的目标是优化价值网络，使其能够准确估计状态的价值（即长期回报的期望）。

#### 核心思想
价值网络的任务是通过回归，逼近实际的目标回报 $\( G_t \)$ 或 TD 目标 $\( V_{\text{target}} \)$。

#### 数学公式
假设：
- $\( V_\theta(s_t) \)$ 是当前价值网络对状态 $\( s_t \)$ 的估计值。
- $\( V_{\text{target}} \)$ 是目标值，通常是由实际回报 $\( G_t \)$ 或 TD 目标计算得出。

Value Loss 通常定义为均方误差（MSE）：

$$L_{\text{value}} = E \left[ \left( V_\theta(s_t) - V_{\text{target}} \right)^2 \right]$$

#### 目标
- 减小估计值 $\( V_\theta(s_t) \)$ 与目标值 $\( V_{\text{target}} \)$ 之间的误差，使得价值网络的预测更准确。

---

### 3. **Policy Loss 和 Value Loss 的区别**

| **损失函数**   | **作用**                                                                                  | **优化目标**                                                                                           | **更新的网络**      |
|-----------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| **Policy Loss** | 用于优化策略网络，使其选择能够最大化长期回报的动作                                           | 最大化 \( r_t A_t \)，同时限制 \( r_t \) 的变化范围，避免策略更新过大                                     | 策略网络（Policy Network） |
| **Value Loss**  | 用于优化价值网络，使其能够准确估计每个状态的价值                                             | 减小 \( V_\theta(s_t) \) 与 \( V_{\text{target}} \) 的差距                                               | 价值网络（Value Network）   |

---

### PPO 中两者的结合
在 PPO 中，最终的损失函数是两个部分的加权和，通常形式为：
\[
\mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} + c_2 \mathcal{L}_{\text{entropy}}
\]

- \( \mathcal{L}_{\text{policy}} \)：策略损失。
- \( \mathcal{L}_{\text{value}} \)：价值损失。
- \( \mathcal{L}_{\text{entropy}} \)：熵正则化项，鼓励策略保持一定的探索性。
- \( c_1 \) 和 \( c_2 \)：超参数，用于调整三者的相对重要性。

GAE：

你感觉**\(\gamma\)**（折扣因子）也在平衡短期和长期，这是完全正确的！事实上，**\(\gamma\)** 和 **\(\lambda\)** 的作用确实都有平衡短期和长期影响的效果，但它们的作用机制和意义有所不同。我们来深入分析二者之间的异同。

---

### **1. \(\gamma\)：控制未来奖励的折扣**
- **定义**：\(\gamma \in [0, 1]\)，用于衰减未来奖励的权重。
- **作用**：
  - \(\gamma\) 决定了智能体对 **未来奖励的重视程度**。
  - 当 \(\gamma = 0\)：智能体只关心当前时间步的即时奖励。
  - 当 \(\gamma \to 1\)：智能体会考虑非常长的时间跨度中的未来奖励。
- **平衡机制**：
  - **短期**：\(\gamma\) 小，智能体专注于优化短期的行为决策，不会被远期的不确定性干扰。
  - **长期**：\(\gamma\) 大，智能体会权衡长远利益，但同时可能引入更多的不确定性（例如远期奖励估计误差）。

#### **总结**：  
\(\gamma\) 决定了奖励函数中的“时间视野”（时间折扣范围）。

---

### **2. \(\lambda\)：控制优势估计中的平滑程度**
- **定义**：\(\lambda \in [0, 1]\)，用于控制 **GAE（Generalized Advantage Estimation）** 中的时间步回溯权重。
- **作用**：
  - \(\lambda\) 决定了优势估计中多时间步的融合程度。
  - 当 \(\lambda = 0\)：只使用单步 TD 误差。
  - 当 \(\lambda = 1\)：融合了所有时间步上的 TD 误差（权重随时间步衰减）。
- **平衡机制**：
  - **短期**：\(\lambda\) 小，估计依赖于单步 TD 误差，关注短期反馈，但可能受噪声影响较大。
  - **长期**：\(\lambda\) 大，通过多个时间步的回溯进行估计，降低噪声，但可能引入更多偏差。

#### **总结**：  
\(\lambda\) 主要影响**如何平滑优势估计**，它更多是一种技术上的实现细节，用于控制 TD 误差的聚合方式。

---

### **\(\gamma\) 和 \(\lambda\) 的区别**
| 特性               | \(\gamma\)                                   | \(\lambda\)                                  |
|--------------------|----------------------------------------------|---------------------------------------------|
| **主要作用**        | 决定未来奖励的折扣，直接影响价值函数        | 决定多步 TD 误差的融合，影响优势估计        |
| **本质**           | 调整时间折扣范围，影响策略目标函数的优化    | 平滑优势估计，控制估计的方差和偏差平衡    |
| **影响范围**        | 直接影响奖励和价值函数的计算                | 间接影响策略更新的效率和稳定性            |
| **控制的时间尺度**  | 时间视野（短期 VS 长期）                    | 多步回溯（短期 TD VS 长期 TD 的平衡）      |

---

### **直观类比**
- \(\gamma\)：像一个**望远镜的焦距**，决定你能看到多远。
- \(\lambda\)：像一个**相机的防抖机制**，决定你在观察时的稳定性。

---

### **同时调整的影响**
- 一般情况下：
  - 较小的 \(\gamma\) 和 \(\lambda\) 使估计偏向短期，更直接、更准确，但容易受到噪声干扰。
  - 较大的 \(\gamma\) 和 \(\lambda\) 会增强长期考虑，结果更平滑，但可能引入更高的偏差。
  
实际应用中，\(\gamma\) 和 \(\lambda\) 往往需要结合任务的具体性质，通过实验调参达到最佳平衡。

**Generalized Advantage Estimation (GAE)** 是一种用于估算优势函数 \(A(s_t, a_t)\) 的方法。它通过加权多个时间步的 TD 误差，兼顾了短期与长期奖励估计，从而平衡了偏差与方差。

以下是 GAE 的完整公式及推导：

---

### **1. GAE 的定义**
\[
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
\]

其中：
- \(\delta_t\) 是第 \(t\) 步的 **TD误差**：
  \[
  \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
  \]
- \(\gamma\)：折扣因子，用于权衡即时奖励与未来奖励。
- \(\lambda\)：权重因子，用于平衡单步 TD 误差和多步累计回报。

---

### **2. GAE 的分步展开**
为了更直观地理解公式，我们可以展开 GAE 的递归形式：
\[
A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
\]

这是一个以 \((\gamma \lambda)\) 为衰减因子的加权和。

---

### **3. 递归形式**
为了高效计算，可以将 GAE 写成递归形式：
\[
A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) A_{t+1}^{\text{GAE}(\gamma, \lambda)}
\]

这意味着当前时间步的 GAE 是当前 TD 误差加上未来优势估计的折扣累加。

---

### **4. 结合时间步回报的展开**
将 TD 误差的公式 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) 代入 GAE 的定义，可以得到：
\[
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \big(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\big)
\]

这里的每一项综合了即时奖励 \(r_t\)、未来状态的价值估计 \(V(s)\)、以及对时间步的加权衰减。

---

### **5. 截断的 GAE（有限轨迹的实现）**
在实际计算中，由于轨迹的长度是有限的，我们通常只计算有限步长的 GAE。设轨迹长度为 \(T\)，GAE 的截断形式为：
\[
A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T-t-1} \delta_{T-1}
\]

---

### **6. 优化后的计算方式**
为了简化实际实现中的计算，我们常用递归方式逆向计算：
1. 初始化：
   \[
   A_{T-1} = \delta_{T-1}
   \]
2. 递归计算：
   \[
   A_t = \delta_t + (\gamma \lambda) A_{t+1}, \quad t = T-2, T-3, \dots, 0
   \]

这种逆向递归的实现非常高效，避免了显式展开求和。

---

### **7. 总结**
GAE 的完整公式主要包括：
- 基本定义：
  \[
  A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
  \]
- 递归形式：
  \[
  A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) A_{t+1}
  \]
- TD误差的定义：
  \[
  \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
  \]

GAE 提供了一种灵活的方法来平滑优势估计，\(\lambda\) 和 \(\gamma\) 共同决定了它对短期和长期奖励的权衡。

https://chatgpt.com/share/67812c3d-9c14-8008-9976-643af6c5517d
```python
    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
```
在强化学习 (Reinforcement Learning, RL) 中，**value**、**advantage** 和 **return** 是核心概念，它们之间有着紧密的关系，具体如下：

---

### **1. Return (回报)**
- **定义**: Return 通常指某一时间点开始到未来的累计奖励。  
  - 数学表示为：
  - $$\[
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
    \]$$
    其中，\( $\gamma \in [0, 1] \$) 是折扣因子，用于权衡短期和长期奖励。
  - $\( G_t \)$ 是从时间 $\( t \)$ 开始的未来总回报。

- **意义**: Return 是强化学习中用来衡量某一序列的最终收益的关键指标，目标是最大化长期回报。

---

### **2. Value (价值)**
- **定义**: Value 是一种对 **return 的期望值**的估计，用于评估某一状态或某一状态-动作对的好坏。
  - **状态价值函数** $\( V(s) \)$:  
    $$\[
    V(s) = \mathbb{E}[G_t | S_t = s]
    \]$$
    即在状态 \( s \) 下，按照策略 \( $\pi$ \) 行动能够获得的期望回报。
  - **动作价值函数** $\( Q(s, a) \)$:  
    $$\[
    Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]
    \]$$
    即在状态 $\( s \)$ 下采取动作 $\( a \)$，并随后按照策略 \( $\pi$ \) 行动能够获得的期望回报。

- **意义**: Value 是强化学习中用来衡量决策优劣的核心指标。通过最大化 Value，智能体可以找到最优策略。

---

### **3. Advantage (优势)**
- **定义**: Advantage 表示某个具体动作 \( a \) 相对于策略下平均动作的优势。  
  - 数学公式：  
    $$\[
    A(s, a) = Q(s, a) - V(s)
    \]$$
    即 Advantage 是动作价值 $\( Q(s, a) \)$ 与状态价值 $\( V(s) \)$ 之间的差。

- **意义**:
  - Advantage 衡量的是**当前动作**与**平均水平动作**相比的优越性。如果 $\( A(s, a) > 0 \)$，说明该动作比平均水平更优；反之则更差。
  - 在策略梯度方法（如 A2C、PPO）中，Advantage 用来指导策略的优化，使智能体更倾向于选择优势更大的动作。

---

### **三者的关系**
1. **Return 与 Value**:
   - Return 是实际获得的累计奖励，而 Value 是对 Return 的期望值。  
     比如，Return 是某一次完整执行策略后的具体结果，而 Value 是从历史经验中学习到的平均结果。

2. **Value 与 Advantage**:
   - Advantage 是 Value 的偏差，它反映了在某个状态下选择具体动作的相对好坏。  
     $\( A(s, a) \)$ 可以通过 $\( Q(s, a) \)$ 和 $\( V(s) \)$ 的差值计算得出，也可以通过经验数据近似。

3. **Return 与 Advantage**:
   - Return 可以用来计算 Advantage：在离线方法中，通过回合的实际 Return 估算 Advantage，从而调整策略。

---

### **示例**
假设在某状态 $\( s \)$：
- 当前策略 $\( \pi \)$ 的期望值 \( $V(s) = 5$ \)。
- 某动作 $\( a \)$ 的价值 $\( Q(s, a) = 8 \)$。

则：
- **Advantage**:
  $$\[
  A(s, a) = Q(s, a) - V(s) = 8 - 5 = 3
  \]$$
  说明选择动作 $\( a \)$ 比平均水平要优越。

强化学习的目标是找到一组策略，使得 $\( V(s) \)$、$\( Q(s, a) \)$ 和 Return 最大化，同时通过优化 Advantage 来提高决策质量。
