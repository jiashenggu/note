import torch
from torch import nn

class GroupQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, group_num):
        super(MutiQueryAttention, self).__init__()
        
        # 设置头数、每个头的维度和组数
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num
        
        # 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)
        
        # 输出的线性变换层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        
        # 计算Q、K、V
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        
        # 将Q、K、V拆分成多个头
        query = self.split_head(query)
        # 将键和值向量拆分为多个注意力头，传入head_num参数为1时是mqa
        key = self.split_head(key, self.group_num)
        value = self.split_head(value, self.group_num)  # 按照组数拆分值
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        
        # 对注意力分数进行softmax归一化，得到注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        
        # 对输出进行维度转换，并恢复原始形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过输出线性层映射到最终的输出空间
        output = self.o_linear(output)
        return output

    # 拆分头部
    def split_head(self, x, group_num=None):
        
        # 获取批次大小和序列长度
        batch_size, seq_len = x.size()[:2]
        
        # 如果没有给定group_num，按照头数拆分
        if group_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 按照给定的组数拆分
            x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
            # 扩展x的维度并重新排列，以符合多头注意力的需求
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim).reshape(batch_size, self.num_heads // group_num * group_num, seq_len, self.head_dim)
            return x
