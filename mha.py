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
        Q = self.q_proj(query).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1)

        # Scaled Dot-Product Attention
        context = self.attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

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
    attn_mask = torch.zeros(seq_length, seq_length).bool()

    # 自定义的 MultiheadAttention
    custom_mha = CustomMultiheadAttention(embed_dim, num_heads, dropout=0.0)

    # 使用 torch.nn.MultiheadAttention 进行对比
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)

    # 复制权重
    custom_mha.q_proj.weight.data = torch_mha.in_proj_weight[:embed_dim].clone()
    custom_mha.k_proj.weight.data = torch_mha.in_proj_weight[embed_dim:2*embed_dim].clone()
    custom_mha.v_proj.weight.data = torch_mha.in_proj_weight[2*embed_dim:].clone()
    custom_mha.q_proj.bias.data = torch_mha.in_proj_bias[:embed_dim].clone()
    custom_mha.k_proj.bias.data = torch_mha.in_proj_bias[embed_dim:2*embed_dim].clone()
    custom_mha.v_proj.bias.data = torch_mha.in_proj_bias[2*embed_dim:].clone()
    custom_mha.out_proj.weight.data = torch_mha.out_proj.weight.clone()
    custom_mha.out_proj.bias.data = torch_mha.out_proj.bias.clone()

    custom_output = custom_mha(query, key, value, attn_mask)

    query_torch = query.transpose(0, 1)  # (seq_length, batch_size, embed_dim)
    key_torch = key.transpose(0, 1)
    value_torch = value.transpose(0, 1)
    torch_output, _ = torch_mha(query_torch, key_torch, value_torch, attn_mask=attn_mask.transpose(0, 1))

    # 将 torch 输出重新调整为与自定义输出相同的形状
    torch_output = torch_output.transpose(0, 1)

    # 比较结果
    assert torch.allclose(custom_output, torch_output, atol=1e-6), "Results do not match"

    print("Test passed: Results match")

# 运行测试
test_custom_multihead_attention()
