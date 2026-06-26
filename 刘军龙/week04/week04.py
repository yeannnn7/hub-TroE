#用pytorch实现一个transformer层。
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式API模块


class MultiHeadAttention(nn.Module):  # 定义多头注意力类，继承自nn.Module
    def __init__(self, d_model, num_heads):  # 初始化方法，参数：模型维度、头数
        super().__init__()  # 调用父类nn.Module的初始化方法
        assert d_model % num_heads == 0  # 确保模型维度能被头数整除
        self.d_k = d_model // num_heads  # 计算每个头的维度
        self.num_heads = num_heads  # 保存头数
        self.W_q = nn.Linear(d_model, d_model)  # 定义查询的线性变换层
        self.W_k = nn.Linear(d_model, d_model)  # 定义键的线性变换层
        self.W_v = nn.Linear(d_model, d_model)  # 定义值的线性变换层
        self.W_o = nn.Linear(d_model, d_model)  # 定义输出的线性变换层

    def scaled_dot_product_attention(self, Q, K, V, mask=None):  # 定义缩放点积注意力方法
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # 计算注意力分数并缩放
        if mask is not None:  # 如果有掩码
            scores = scores.masked_fill(mask == 0, -1e9)  # 用极小值填充掩码位置，避免注意力
        attention_weights = F.softmax(scores, dim=-1)  # 对最后一个维度做softmax得到注意力权重
        output = torch.matmul(attention_weights, V)  # 用注意力权重对值进行加权求和
        return output, attention_weights  # 返回输出和注意力权重

    def forward(self, Q, K, V, mask=None):  # 定义前向传播方法
        batch_size = Q.size(0)  # 获取批次大小
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 对Q做线性变换并分头
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 对K做线性变换并分头
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 对V做线性变换并分头
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)  # 计算多头注意力
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # 合并多头输出
        output = self.W_o(attn_output)  # 对合并后的输出做线性变换
        return output, attn_weights  # 返回最终输出和注意力权重


class FeedForward(nn.Module):  # 定义前馈网络类，继承自nn.Module
    def __init__(self, d_model, d_ff, dropout=0.1):  # 初始化方法，参数：模型维度、前馈网络维度、dropout率
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.linear1 = nn.Linear(d_model, d_ff)  # 定义第一个线性变换层
        self.linear2 = nn.Linear(d_ff, d_model)  # 定义第二个线性变换层
        self.dropout = nn.Dropout(dropout)  # 定义dropout层

    def forward(self, x):  # 定义前向传播方法
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # 线性变换→ReLU激活→dropout→线性变换


class TransformerLayer(nn.Module):  # 定义Transformer层类，继承自nn.Module
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  # 初始化方法，参数：模型维度、头数、前馈网络维度、dropout率
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 定义多头自注意力层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 定义前馈网络层
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二个层归一化
        self.dropout1 = nn.Dropout(dropout)  # 定义第一个dropout层
        self.dropout2 = nn.Dropout(dropout)  # 定义第二个dropout层

    def forward(self, x, mask=None):  # 定义前向传播方法
        attn_output, _ = self.self_attn(x, x, x, mask)  # 计算多头自注意力
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接+dropout+层归一化
        ff_output = self.feed_forward(x)  # 计算前馈网络输出
        x = self.norm2(x + self.dropout2(ff_output))  # 残差连接+dropout+层归一化
        return x  # 返回Transformer层的输出


# 测试代码
if __name__ == "__main__":  # 如果是直接运行此文件
    d_model = 512  # 设置模型维度为512
    num_heads = 8  # 设置头数为8
    d_ff = 2048  # 设置前馈网络维度为2048
    batch_size = 2  # 设置批次大小为2
    seq_len = 10  # 设置序列长度为10

    transformer_layer = TransformerLayer(d_model, num_heads, d_ff)  # 创建Transformer层实例
    x = torch.randn(batch_size, seq_len, d_model)  # 随机生成输入张量
    output = transformer_layer(x)  # 前向传播
    print(f"输入形状: {x.shape}")  # 打印输入形状
    print(f"输出形状: {output.shape}")  # 打印输出形状
