import math
import torch
import torch.nn as nn
import numpy as np

'''
手写transformer模型的训练流程
1. 构造模型
2. 构造数据集
3. 训练模型
4. 评估模型
5. 预测
'''

# gelu激活函数
def gelu(x):
    # 对x做tensor.detach().numpy()操作，得到numpy数组
    x = x.detach().numpy()
    x = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))
    return torch.tensor(x)  # 将numpy数组转换回tensor

#构造字符表
vocab = {
    "[pad]" : 0,
    "喜欢" : 1,
    "你好" : 2,
    "中国" : 3,
    "我" : 4,
    "[cls]" : 5,
    "[sep]" : 6,
    "[unk]":7
}

#构造数据集
input_text = [5, 4, 1, 3,  6, 2, 3, 6, 2, 4, 1, 3, 6] # token_ids
token = [5, 4, 1, 3,  6, 2, 3, 6, 2, 4, 1, 3, 6] # token_ids 
seg = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0] # segment_ids全为0
pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # 位置编码


class DiyTransformer:
    def __init__(self, input_text):
        self.num_attention_heads = 12 # 注意力头的数量
        self.hidden_size = 768 # transformer的隐藏层维度
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads) # 每个注意力头的维度是64
        self.num_layers = 2 # transformer层数
        self.embedding_forward(input_text, self.hidden_size) # 构造embedding层

    # 使用3层embedding层和位置编码层
    def embedding_forward(self, input_text, embedding_dim):
        '''三层embedding的处理'''
        vocab_size = len(input_text) # 词汇表大小
        token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # token embedding层
        segment_embedding = nn.Embedding(2, embedding_dim)  # segment embedding层
        position_embedding = nn.Embedding(512, embedding_dim)  # 位置编码层

        # embedding层相加
        token_ids = torch.LongTensor(token)  # 将输入文本转换为LongTensor
        segment_ids = torch.LongTensor(seg)  # 将segment_ids转换为LongTensor
        position_ids = torch.LongTensor(pos)  # 位置编码
        output = token_embedding(token_ids) + segment_embedding(segment_ids) + position_embedding(position_ids)

        # 对加和后的embedding进行归一化
        output = nn.LayerNorm(embedding_dim)(output)

        self.embedding = output

    # 多层transformer计算
    def all_transformer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_forward(x)
        return x

    # 单层transformer计算
    def single_transformer_forward(self, x):
        # 计算多头注意力
        attention_output = self.multi_head_attention_forward(x)

        # 残差连接
        x = nn.LayerNorm(self.hidden_size)(x + attention_output)

        # 计算前馈网络
        feed_forward_output = self.feed_forward_forward(x)

        # 残差连接
        output = nn.LayerNorm(self.hidden_size)(x + feed_forward_output)

        return output

    # 多头注意力计算
    def multi_head_attention_forward(self, x):
        # 计算Q、K、V矩阵
        Q = nn.Linear(len(x[0]), self.hidden_size)(x)
        K = nn.Linear(len(x[0]), self.hidden_size)(x)
        V = nn.Linear(len(x[0]), self.hidden_size)(x)

        # 将Q、K、V矩阵分成多个头
        Q = Q.view(-1, self.num_attention_heads, self.attention_head_size)
        K = K.view(-1, self.num_attention_heads, self.attention_head_size)
        V = V.view(-1, self.num_attention_heads, self.attention_head_size)

        # 计算注意力权重
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_head_size) 

        # 添加mask 上三角mask遮盖未来信息
        mask = torch.triu(torch.ones(attention_weights.size(-2), attention_weights.size(-1)), diagonal=1).bool() # 上三角mask
        attention_weights = attention_weights.masked_fill(mask, float('-inf')) # 将mask位置的权重设置为负无穷大

        attention_weights = nn.Softmax(dim=-1)(attention_weights) # 对注意力权重进行softmax归一化

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)

        # 将多个头的输出拼接起来
        attention_output = attention_output.view(-1, self.hidden_size)

        return attention_output
    
    # 前馈网络计算
    def feed_forward_forward(self, x):
        # 计算前馈网络的输出
        feed_forward_output = nn.Linear(self.hidden_size, self.hidden_size * 4)(x)

        feed_forward_output = gelu(feed_forward_output)

        feed_forward_output = nn.Linear(self.hidden_size * 4, self.hidden_size)(feed_forward_output)

        return feed_forward_output

    def forward(self, x):
        # 获取tansformer的输出
        transformer_output = self.all_transformer_forward(x)
        return transformer_output
    
# 构造模型
model = DiyTransformer(input_text)
# 获取模型的输出
output = model.forward(model.embedding)

# 使用softmax函数进行归一化
output = nn.Softmax(dim=-1)(output)   # (13, 768)

# 打印输出的向量和维度

print(output)
print(output.size())
