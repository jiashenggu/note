# https://zhuanlan.zhihu.com/p/1936606157826405989
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        return context

def create_causal_mask(seq_len):
  """
  创建一个因果注意力掩码。

  Args:
    seq_len: 序列长度。

  Returns:
    一个形状为 (seq_len, seq_len) 的注意力掩码张量。
  """
  mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
  return mask

# 示例
sequence_length = 5
causal_mask = create_causal_mask(sequence_length)
print(causal_mask)

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_length, embed_dim = query.size()

        # Linear projections
        Q = (
            self.q_proj(query)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled Dot-Product Attention
        context = self.attention(Q, K, V, attn_mask)
        # reshape 和 view 的主要区别是：
        # view 要求张量必须是连续的，如果不连续就会报错
        # reshape 会自动处理非连续的情况，如果需要的话会隐式地调用 contiguous()
        # 因此使用 reshape 更方便，也能让代码更简洁。除非有特殊的性能考虑（因为 view 不会复制数据），否则推荐使用 reshape。
        # 所以：
        # # 这两种写法的性能是完全等价的
        # context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        # context = context.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        # 在实际应用中：
        # 如果确定张量一定是连续的，用 view 和 reshape 性能都一样
        # 如果不确定是否连续，用 reshape 更安全，而且代码更简洁
        # 性能差异通常很小，除非在非常关键的性能场景下，否则这点差异基本可以忽略
        # 如果真的对性能特别敏感，可以考虑重新设计算法避免需要进行 transpose 操作。因为真正的性能开销主要来自必要的 transpose 操作，而不是 reshape 与 view 的选择。

        # contiguous() 是复制操作，不是移动。它的工作原理是：
        # 创建一个新的内存空间
        # 将原始数据按照新的内存布局复制到这个新空间中
        # 返回指向这个新内存空间的张量

        # 这就是为什么有时候在性能敏感的场景下，需要仔细考虑是否真的需要调用 contiguous()。如果可能的话，最好在设计算法时就避免会导致非连续内存布局的操作。
        # 但也要注意，即使 contiguous() 会导致数据复制，这个开销相比于神经网络中的计算开销（如矩阵乘法）通常是很小的，所以在大多数情况下不需要过度优化这一点。
        context = context.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        return output


def test_custom_multihead_attention():
    # 固定随机种子
    torch.manual_seed(0)

    # 参数设置
    batch_size = 2
    seq_length = 5
    embed_dim = 128
    num_heads = 4

    # 随机生成输入数据
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)

    attn_mask = create_causal_mask(sequence_length)

    # 自定义的 MultiheadAttention
    custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=0.0)

    # 使用 torch.nn.MultiheadAttention 进行对比
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)

    # 复制权重
    custom_mha.q_proj.weight.data = torch_mha.in_proj_weight[:embed_dim].clone()
    custom_mha.k_proj.weight.data = torch_mha.in_proj_weight[
        embed_dim : 2 * embed_dim
    ].clone()
    custom_mha.v_proj.weight.data = torch_mha.in_proj_weight[2 * embed_dim :].clone()
    custom_mha.q_proj.bias.data = torch_mha.in_proj_bias[:embed_dim].clone()
    custom_mha.k_proj.bias.data = torch_mha.in_proj_bias[
        embed_dim : 2 * embed_dim
    ].clone()
    custom_mha.v_proj.bias.data = torch_mha.in_proj_bias[2 * embed_dim :].clone()
    custom_mha.out_proj.weight.data = torch_mha.out_proj.weight.clone()
    custom_mha.out_proj.bias.data = torch_mha.out_proj.bias.clone()

    custom_output = custom_mha(query, key, value, attn_mask)

    query_torch = query.transpose(0, 1)  # (seq_length, batch_size, embed_dim)
    key_torch = key.transpose(0, 1)
    value_torch = value.transpose(0, 1)
    torch_output, _ = torch_mha(
        query_torch, key_torch, value_torch, attn_mask=attn_mask.transpose(0, 1)
    )

    # 将 torch 输出重新调整为与自定义输出相同的形状
    torch_output = torch_output.transpose(0, 1)

    # 比较结果
    assert torch.allclose(
        custom_output, torch_output, atol=1e-6
    ), "Results do not match"

    print("Test passed: Results match")


# 运行测试
test_custom_multihead_attention()
