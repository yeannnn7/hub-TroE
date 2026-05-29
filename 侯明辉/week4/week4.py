"""
用 PyTorch 实现完整的 Transformer 层（Encoder + Decoder）
包含：多头自注意力、多头交叉注意力、前馈网络、残差连接、层归一化、位置编码
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        B, T_q, _ = query.shape
        T_k = key.size(1)

        # 线性投影
        Q = self.W_q(query)  
        K = self.W_k(key)    
        V = self.W_v(value)  

        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, heads, T_q, T_k)

        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)  # (B, heads, T_q, head_dim)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # 输出投影
        output = self.proj_dropout(self.W_o(context))
        return output


# ─────────────────────── 2. 前馈网络 ───────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ─────────────────────── 3. Transformer Encoder Layer ───────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# ─────────────────────── 4. Transformer Decoder Layer ───────────────────────

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 掩码自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_tgt_mask=None):
        # 1. 掩码自注意力 + 残差 + LayerNorm
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # 2. 交叉注意力（Q 来自 decoder，K/V 来自 encoder）+ 残差 + LayerNorm
        cross_attn_out = self.cross_attn(x, enc_output, enc_output, src_tgt_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 3. FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


# ─────────────────────── 5. 位置编码 ───────────────────────

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────── 6. 完整 Transformer ───────────────────────

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_len=512, dropout=0.1):
        super().__init__()

        # Embedding
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Encoder 堆叠
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder 堆叠
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_mask=None, src_tgt_mask=None):
        x = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_tgt_mask=None):

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_tgt_mask)
        logits = self.output_proj(dec_output)  # (B, T_tgt, tgt_vocab_size)
        return logits


# ─────────────────────── 7. 生成因果掩码 ───────────────────────

def generate_causal_mask(seq_len):

    return torch.tril(torch.ones(seq_len, seq_len)).bool()


# ─────────────────────── 8. 验证 Encoder 层与 PyTorch  ───────────────────────

def verify_encoder_layer():
    torch.manual_seed(42)

    d_model, num_heads, d_ff, dropout = 512, 8, 2048, 0.0

    my_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

    official_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
        dropout=dropout, batch_first=True, norm_first=False,
    )

    # 复制权重使参数一致
    with torch.no_grad():
        official_layer.self_attn.in_proj_weight.copy_(
            torch.cat([my_layer.self_attn.W_q.weight,
                       my_layer.self_attn.W_k.weight,
                       my_layer.self_attn.W_v.weight], dim=0)
        )
        official_layer.self_attn.in_proj_bias.copy_(
            torch.cat([my_layer.self_attn.W_q.bias,
                       my_layer.self_attn.W_k.bias,
                       my_layer.self_attn.W_v.bias], dim=0)
        )
        official_layer.self_attn.out_proj.weight.copy_(my_layer.self_attn.W_o.weight)
        official_layer.self_attn.out_proj.bias.copy_(my_layer.self_attn.W_o.bias)

        official_layer.linear1.weight.copy_(my_layer.ffn.fc1.weight)
        official_layer.linear1.bias.copy_(my_layer.ffn.fc1.bias)
        official_layer.linear2.weight.copy_(my_layer.ffn.fc2.weight)
        official_layer.linear2.bias.copy_(my_layer.ffn.fc2.bias)

        official_layer.norm1.weight.copy_(my_layer.norm1.weight)
        official_layer.norm1.bias.copy_(my_layer.norm1.bias)
        official_layer.norm2.weight.copy_(my_layer.norm2.weight)
        official_layer.norm2.bias.copy_(my_layer.norm2.bias)

    x = torch.randn(2, 10, d_model)
    my_layer.eval()
    official_layer.eval()

    with torch.no_grad():
        out_my = my_layer(x)
        out_official = official_layer(x)

    diff = (out_my - out_official).abs().max().item()
    print(f"[Encoder] 自制层 vs PyTorch 官方层 最大误差: {diff:.2e}")
    print(f"[Encoder] 验证结果: {'✓ 通过' if diff < 1e-4 else '✗ 不匹配'}")


# ─────────────────────── 9. 验证 Decoder 层与 PyTorch  ───────────────────────

def verify_decoder_layer():
    torch.manual_seed(42)

    d_model, num_heads, d_ff, dropout = 512, 8, 2048, 0.0

    my_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)

    official_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
        dropout=dropout, batch_first=True, norm_first=False,
    )

    # 复制权重
    with torch.no_grad():
        # 自注意力
        official_layer.self_attn.in_proj_weight.copy_(
            torch.cat([my_layer.self_attn.W_q.weight,
                       my_layer.self_attn.W_k.weight,
                       my_layer.self_attn.W_v.weight], dim=0)
        )
        official_layer.self_attn.in_proj_bias.copy_(
            torch.cat([my_layer.self_attn.W_q.bias,
                       my_layer.self_attn.W_k.bias,
                       my_layer.self_attn.W_v.bias], dim=0)
        )
        official_layer.self_attn.out_proj.weight.copy_(my_layer.self_attn.W_o.weight)
        official_layer.self_attn.out_proj.bias.copy_(my_layer.self_attn.W_o.bias)

        # 交叉注意力
        official_layer.multihead_attn.in_proj_weight.copy_(
            torch.cat([my_layer.cross_attn.W_q.weight,
                       my_layer.cross_attn.W_k.weight,
                       my_layer.cross_attn.W_v.weight], dim=0)
        )
        official_layer.multihead_attn.in_proj_bias.copy_(
            torch.cat([my_layer.cross_attn.W_q.bias,
                       my_layer.cross_attn.W_k.bias,
                       my_layer.cross_attn.W_v.bias], dim=0)
        )
        official_layer.multihead_attn.out_proj.weight.copy_(my_layer.cross_attn.W_o.weight)
        official_layer.multihead_attn.out_proj.bias.copy_(my_layer.cross_attn.W_o.bias)

        # FFN
        official_layer.linear1.weight.copy_(my_layer.ffn.fc1.weight)
        official_layer.linear1.bias.copy_(my_layer.ffn.fc1.bias)
        official_layer.linear2.weight.copy_(my_layer.ffn.fc2.weight)
        official_layer.linear2.bias.copy_(my_layer.ffn.fc2.bias)

        # LayerNorm
        official_layer.norm1.weight.copy_(my_layer.norm1.weight)
        official_layer.norm1.bias.copy_(my_layer.norm1.bias)
        official_layer.norm2.weight.copy_(my_layer.norm2.weight)
        official_layer.norm2.bias.copy_(my_layer.norm2.bias)
        official_layer.norm3.weight.copy_(my_layer.norm3.weight)
        official_layer.norm3.bias.copy_(my_layer.norm3.bias)

    tgt = torch.randn(2, 8, d_model)
    memory = torch.randn(2, 10, d_model)

    my_layer.eval()
    official_layer.eval()

    with torch.no_grad():
        out_my = my_layer(tgt, memory)
        out_official = official_layer(tgt, memory)

    diff = (out_my - out_official).abs().max().item()
    print(f"[Decoder] 自制层 vs PyTorch 官方层 最大误差: {diff:.2e}")
    print(f"[Decoder] 验证结果: {'✓ 通过' if diff < 1e-4 else '✗ 不匹配'}")


# ─────────────────────── 10. 完整 Transformer 训练演示 ───────────────────────

def demo_transformer():
    torch.manual_seed(42)

    # 超参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    max_len = 32
    batch_size = 4
    src_len = 10
    tgt_len = 12

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, num_heads, d_ff, num_layers, max_len
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 模拟数据
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    # 因果掩码（decoder 看不到未来）
    tgt_mask = generate_causal_mask(tgt_len)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for step in range(30):
        logits = model(src, tgt, tgt_mask=tgt_mask)  # (B, T_tgt, vocab_size)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    print("训练完成！")


# ─────────────────────── 主函数 ───────────────────────

if __name__ == "__main__":

    print("\n[1] 验证 Encoder 层与 PyTorch 官方层一致性...")
    verify_encoder_layer()

    print("\n[2] 验证 Decoder 层与 PyTorch 官方层一致性...")
    verify_decoder_layer()

    print("\n[3] 用完整 Transformer 做序列到序列任务演示...")
    demo_transformer()
