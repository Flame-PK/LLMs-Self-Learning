import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

import math

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int = 768 # hidden size, or called hidden_dim
    # 主要是为了可以tie_embeddings_weights
    dropout: float = 0.1
    head_size: int = n_embed // n_heads
    # vocab_size
    # gpt2
    vocab_size: int = 50257

# 定义GPT的结构

# 1. Single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)

        # attention_mask 新的方法 通过register_buffer 注册
        # 这样不用计算梯度，所以节约内存和显存，速度也更快

        self.register_buffer(
            "mask", 
            # 生成一个下三角矩阵 tril
            # block size 就是文本的最大长度，512
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 计算权重
        # @ 矩阵乘法 torch.matmul的简化写法
        weight = q @ k.transpose(-2, -1) 
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        )
        # 要注意计算weight的时候，要除以根号下d_k
        weight = F.softmax(weight, dim = -1) / math.sqrt(self.head_size)
        # dropout 要放到weight后面， 而不是放到乘完的value那
        weight = self.dropout(weight)


# 2. multi head attention
class MultiHeadAttention(nn.model):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )

        output = self.proj(output)
        output = self.dropout(output)
        return output

# 3. feed forward MLP
class FeedForward(nn.Module):
    output = torch.cat(
        
    )