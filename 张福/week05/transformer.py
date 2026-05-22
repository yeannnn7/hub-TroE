import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制类
    
    实现Transformer中的多头自注意力机制，允许模型同时关注输入序列的不同位置特征。
    
    Args:
        embed_dim (int): 嵌入维度，即输入特征的维度
        num_heads (int): 注意力头的数量
        
    Attributes:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数量
        head_dim (int): 每个注意力头的维度，embed_dim // num_heads
        w_q (nn.Linear): 查询向量的线性变换层
        w_k (nn.Linear): 键向量的线性变换层lfdz
        w_v (nn.Linear): 值向量的线性变换层
        w_o (nn.Linear): 输出的线性变换层
    """
    
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        
        # 确保嵌入维度能被头数整除
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # 定义Q、K、V、O的线性变换层
        self.w_q = nn.Linear(embed_dim, embed_dim)  # 查询变换
        self.w_k = nn.Linear(embed_dim, embed_dim)  # 键变换
        self.w_v = nn.Linear(embed_dim, embed_dim)  # 值变换
        self.w_o = nn.Linear(embed_dim, embed_dim)  # 输出变换
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力计算
        
        计算注意力分数并应用mask，最后通过softmax得到注意力权重。
        
        Args:
            q (torch.Tensor): 查询张量，形状 [batch_size, num_heads, seq_len, head_dim]
            k (torch.Tensor): 键张量，形状 [batch_size, num_heads, seq_len, head_dim]
            v (torch.Tensor): 值张量，形状 [batch_size, num_heads, seq_len, head_dim]
            mask (torch.Tensor, optional): 掩码张量，形状 [seq_len, seq_len]，0表示遮掩位置
        
        Returns:
            tuple: (输出张量, 注意力权重)
        """
        d_k = q.size(-1)
        
        # 计算注意力分数：Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 应用mask：将mask为0的位置填充为负无穷，softmax后概率接近0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def split_heads(self, x, batch_size):
        """
        将输入张量按头数分割
        
        将形状 [batch_size, seq_len, embed_dim] 的张量转换为
        [batch_size, num_heads, seq_len, head_dim] 的形状。
        
        Args:
            x (torch.Tensor): 输入张量
            batch_size (int): 批次大小
        
        Returns:
            torch.Tensor: 分割后的张量
        """
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query (torch.Tensor): 查询输入，形状 [batch_size, seq_len, embed_dim]
            key (torch.Tensor): 键输入，形状 [batch_size, seq_len, embed_dim]
            value (torch.Tensor): 值输入，形状 [batch_size, seq_len, embed_dim]
            mask (torch.Tensor, optional): 掩码张量
        
        Returns:
            tuple: (输出张量, 注意力权重)
        """
        batch_size = query.size(0)
        
        # 通过线性层得到Q、K、V
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # 分割多头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # 扩展mask维度以匹配多头结构: [batch_size, 1, 1, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # 计算缩放点积注意力
        output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多头结果：先转置再reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 输出线性变换
        output = self.w_o(output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    位置编码类
    
    实现Transformer中的正弦/余弦位置编码，为输入序列添加位置信息。
    
    Args:
        embed_dim (int): 嵌入维度
        max_len (int): 最大序列长度，默认为5000
    
    Attributes:
        pe (torch.Tensor): 预计算的位置编码表
    """
    
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 生成位置索引 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        
        # 计算分母项：10000^(2i/d_model) 的对数形式
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, 1, embed_dim)
        
        # 偶数位置使用正弦函数
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        # 奇数位置使用余弦函数
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # 将pe注册为缓冲区，不会被优化器更新
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播：将位置编码加到输入上
        
        Args:
            x (torch.Tensor): 输入张量，形状 [seq_len, batch_size, embed_dim]
        
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerBlock(nn.Module):
    """
    Transformer基础块类
    
    实现Transformer的单个层，包含多头注意力子层和前馈神经网络子层，
    每个子层都有残差连接和层归一化。
    
    Args:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数量
        ff_dim (int): 前馈网络隐藏层维度
        dropout (float): dropout概率，默认为0.1
    
    Attributes:
        multi_head_attn (MultiHeadAttention): 多头注意力层
        ffn (nn.Sequential): 前馈神经网络
        norm1 (nn.LayerNorm): 第一个层归一化
        norm2 (nn.LayerNorm): 第二个层归一化
        dropout1 (nn.Dropout): 第一个dropout层
        dropout2 (nn.Dropout): 第二个dropout层
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # 多头注意力层
        self.multi_head_attn = MultiHeadAttention(embed_dim, num_heads)
        
        # 前馈神经网络：两层线性变换，中间GELU激活
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # 层归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状 [batch_size, seq_len, embed_dim]
            mask (torch.Tensor, optional): 掩码张量
        
        Returns:
            tuple: (输出张量, 注意力权重)
        """
        # 多头注意力子层 + 残差连接 + 层归一化
        attn_output, attn_weights = self.multi_head_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络子层 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class Transformer(nn.Module):
    """
    Transformer模型类
    
    完整的Transformer模型实现，包含词嵌入、位置编码、多个Transformer块和输出层。
    通过下三角mask实现自回归特性：每个位置只能关注前面的位置。
    
    Args:
        vocab_size (int): 词汇表大小
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数量
        ff_dim (int): 前馈网络隐藏层维度
        num_layers (int): Transformer块的数量
        max_len (int): 最大序列长度，默认为5000
        dropout (float): dropout概率，默认为0.1
    
    Attributes:
        embed_dim (int): 嵌入维度
        embedding (nn.Embedding): 词嵌入层
        positional_encoding (PositionalEncoding): 位置编码层
        transformer_blocks (nn.ModuleList): Transformer块列表
        fc (nn.Linear): 输出线性层
        dropout (nn.Dropout): dropout层
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        
        # 词嵌入层：将词索引转换为词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer块列表
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层：将嵌入维度映射到词汇表大小
        self.fc = nn.Linear(embed_dim, vocab_size)
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, seq_len):
        """
        生成下三角掩码
        
        创建一个下三角矩阵，对角线及以下为1（可见），对角线以上为0（遮掩）。
        实现"每个字只能看到前面的字"的自回归特性。
        
        示例（seq_len=4）：
        [[1, 0, 0, 0]
         [1, 1, 0, 0]
         [1, 1, 1, 0]
         [1, 1, 1, 1]]
        
        Args:
            seq_len (int): 序列长度
        
        Returns:
            torch.Tensor: 下三角掩码，形状 [seq_len, seq_len]
        """
        # torch.triu生成上三角矩阵（对角线以上为1），diagonal=1表示不包含主对角线
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        
        # 取反得到下三角矩阵：对角线及以下为1
        mask = 1 - mask
        
        return mask
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入序列，形状 [batch_size, seq_len]，包含词索引
        
        Returns:
            tuple: (输出张量, 所有层的注意力权重列表)
                - 输出张量形状: [batch_size, seq_len, vocab_size]
                - 注意力权重列表: 每个元素形状 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = x.size()
        
        # 生成下三角mask并移动到正确设备
        mask = self.generate_mask(seq_len).to(x.device)
        
        # 词嵌入 + 位置编码
        # 乘以sqrt(embed_dim)是为了平衡嵌入和位置编码的幅度
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 逐层通过Transformer块
        all_attn_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            all_attn_weights.append(attn_weights)
        
        # 输出层
        output = self.fc(x)
        
        return output