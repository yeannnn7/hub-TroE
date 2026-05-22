import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# ---------------------- 1. 缩放点积注意力 ----------------------
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力：Attention(Q,K,V) = softmax(QK^T/√d_k)V"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q/k/v shape: [batch_size, n_heads, seq_len, d_k]
        d_k = q.size(-1)
        
        # 计算 Q*K^T / √d_k
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 掩码（可选，用于屏蔽padding/未来token）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重 + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# ---------------------- 2. 多头注意力 ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型总维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q/K/V 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出线性层 + 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影 + 拆分为多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 3. 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最终线性层 + dropout
        output = self.dropout(self.fc_out(attn_output))
        return output, attn_weights

# ---------------------- 3. 前馈网络 ----------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 两层线性层 + 激活函数（ReLU/GELU）
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ---------------------- 4. Transformer Encoder Layer（核心） ----------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 核心组件
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1：自注意力 + 残差 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 子层2：前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x
    # 超参数
d_model = 512    # 模型维度
n_heads = 8     # 注意力头数
d_ff = 2048     # 前馈网络中间维度
batch_size = 2  # 批次大小
seq_len = 10    # 序列长度

# 初始化Transformer层
encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

# 构造随机输入 [batch, seq_len, d_model]
x = torch.randn(batch_size, seq_len, d_model)
# 前向传播
output = encoder_layer(x)

# 打印输出形状（输入输出形状完全一致）
print("输入形状:", x.shape)
print("输出形状:", output.shape)
