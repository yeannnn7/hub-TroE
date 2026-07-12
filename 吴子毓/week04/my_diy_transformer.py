import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制（封装成类）"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, encoder_output=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [seq_len, seq_len] 或 None
            encoder_output: [batch, src_len, hidden_size] 或 None
        Returns:
            attn_output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Q 总是来自 hidden_states
        Q = self.q_linear(hidden_states)
        
        # K 和 V 的来源
        if encoder_output is None:
            # 自注意力
            K = self.k_linear(hidden_states)
            V = self.v_linear(hidden_states)
            src_len = seq_len
        else:
            # 交叉注意力
            K = self.k_linear(encoder_output)
            V = self.v_linear(encoder_output)
            src_len = encoder_output.size(1)
        
        # 切分多头: [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（只对自注意力）
        if attention_mask is not None and encoder_output is None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_linear(attn_output)
        
        return attn_output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder层"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        # 1. Self-Attention + 残差 + Norm
        attn_output = self.self_attention(hidden_states)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        
        # 2. Feed Forward + 残差 + Norm
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states


class DecoderLayer(nn.Module):
    """Decoder层"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.cross_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def create_look_ahead_mask(self, size):
        """创建自回归掩码"""
        mask = torch.tril(torch.ones(size, size))
        return mask
    
    def forward(self, hidden_states, encoder_output=None):
        """
        Args:
            hidden_states: [batch, tgt_seq_len, hidden_size]
            encoder_output: [batch, src_seq_len, hidden_size] 或 None
        Returns:
            [batch, tgt_seq_len, hidden_size]
        """
        seq_len = hidden_states.size(1)
        
        # 1. Masked Self-Attention + 残差 + Norm
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        attn_output = self.masked_self_attention(hidden_states, attention_mask=look_ahead_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        
        # 2. Cross-Attention + 残差 + Norm
        if encoder_output is not None:
            attn_output = self.cross_attention(hidden_states, encoder_output=encoder_output)
            hidden_states = self.norm2(hidden_states + self.dropout(attn_output))
        
        # 3. Feed Forward + 残差 + Norm
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm3(hidden_states + ff_output)
        
        return hidden_states


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072
    dropout_rate = 0.1
    
    print("=" * 60)
    print("Transformer 类封装版本测试")
    print("=" * 60)
    
    # 创建模型实例
    encoder = EncoderLayer(hidden_size, num_heads, intermediate_size, dropout_rate)
    decoder = DecoderLayer(hidden_size, num_heads, intermediate_size, dropout_rate)
    
    # 设置为评估模式（关闭dropout）
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # ========== Encoder ==========
        print("\n【Encoder部分】")
        encoder_input = torch.randn(batch_size, src_seq_len, hidden_size)
        encoder_output = encoder(encoder_input)
        print(f"Encoder 输入形状:  {encoder_input.shape}")
        print(f"Encoder 输出形状:  {encoder_output.shape}")
        
        # ========== Decoder ==========
        print("\n【Decoder部分】")
        decoder_input = torch.randn(batch_size, tgt_seq_len, hidden_size)
        decoder_output = decoder(decoder_input, encoder_output)
        print(f"Decoder 输入形状:  {decoder_input.shape}")
        print(f"Decoder 输出形状:  {decoder_output.shape}")
    
    print("\n" + "=" * 60)
    print("模型结构:")
    print(f"  Encoder: {encoder}")
    print(f"  Decoder: {decoder}")
    print("=" * 60)
