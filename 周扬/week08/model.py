"""
文本匹配模型 - BiEncoder
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


def load_bert_with_layers(model_name='bert-base-chinese', num_hidden_layers=3, config_name=None):
    """按指定层数加载 BERT，并允许配置来源与权重来源分离。"""
    config_source = config_name or model_name
    config = BertConfig.from_pretrained(config_source)
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    return BertModel.from_pretrained(model_name, config=config)


class BiEncoder(nn.Module):
    """

    用途：
      - 文本匹配、语义相似度计算
      - 支持 CosineEmbeddingLoss 训练
      - 支持 TripletLoss 训练

    参数：
      model_name : 预训练模型名称或路径
      hidden_size: BERT 输出维度（bert-base 为 768）
    """

    def __init__(self, model_name='bert-base-chinese', hidden_size=768, num_hidden_layers=3, config_name=None):
        super().__init__()

        # 默认只取前 3 层 Transformer，加快训练速度
        self.bert = load_bert_with_layers(
            model_name,
            num_hidden_layers=num_hidden_layers,
            config_name=config_name,
        )

        # 可选：添加一个线性层映射到更小的维度
        # self.projection = nn.Linear(hidden_size, 256)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        编码单个句子，返回句向量

        参数：
          input_ids:      [batch_size, seq_len]
          attention_mask: [batch_size, seq_len]
          token_type_ids: [batch_size, seq_len]

        返回：
          sentence_embedding: [batch_size, hidden_size]
        """
        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 方式1：使用 [CLS] 位置的向量作为句向量
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # pooler_output: [batch_size, hidden_size] (经过 tanh 激活的 [CLS])
        sentence_embedding = outputs.pooler_output

        # 方式2：mean pooling（取所有 token 的平均）
        # last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        # sentence_embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # 可选：通过投影层
        # sentence_embedding = self.projection(sentence_embedding)

        return sentence_embedding

    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a,
                      input_ids_b, attention_mask_b, token_type_ids_b):
        """
        前向传播：编码两个句子，计算相似度

        参数：
          input_ids_a, attention_mask_a, token_type_ids_a: 句子A的编码
          input_ids_b, attention_mask_b, token_type_ids_b: 句子B的编码

        返回：
          dict:
            - emb_a:      句子A的向量 [batch_size, hidden_size]
            - emb_b:      句子B的向量 [batch_size, hidden_size]
            - similarity: 余弦相似度 [batch_size]
        """
        # 编码两个句子
        emb_a = self.encode(input_ids_a, attention_mask_a, token_type_ids_a)
        emb_b = self.encode(input_ids_b, attention_mask_b, token_type_ids_b)

        # 计算余弦相似度
        similarity = self.cosine_similarity(emb_a, emb_b)

        return {
            'emb_a': emb_a,
            'emb_b': emb_b,
            'similarity': similarity
        }

    @staticmethod
    def cosine_similarity(a, b):
        """
        计算两个向量的余弦相似度

        参数：
          a: [batch_size, hidden_size]
          b: [batch_size, hidden_size]

        返回：
          similarity: [batch_size]
        """
        # 归一化
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)

        # 点积
        similarity = (a_norm * b_norm).sum(dim=-1)

        return similarity


class BiEncoderForClassification(nn.Module):
    """
    双塔编码器 + 分类头

    用途：
      - 直接预测两个句子是否匹配（二分类）
      - 使用 CrossEntropyLoss 训练

    参数：
      model_name : 预训练模型名称或路径
      hidden_size: BERT 输出维度
    """

    def __init__(self, model_name='bert-base-chinese', hidden_size=768, num_hidden_layers=3, config_name=None):
        super().__init__()

        self.encoder = BiEncoder(
            model_name,
            hidden_size,
            num_hidden_layers=num_hidden_layers,
            config_name=config_name,
        )

        # 分类头：接收两个句向量的拼接
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # 二分类
        )

    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a,
                      input_ids_b, attention_mask_b, token_type_ids_b,
                      labels=None):
        """
        前向传播

        返回：
          dict:
            - logits: 分类 logits [batch_size, 2]
            - loss:   如果提供 labels，返回交叉熵损失
        """
        # 获取两个句向量
        outputs = self.encoder(
            input_ids_a, attention_mask_a, token_type_ids_a,
            input_ids_b, attention_mask_b, token_type_ids_b
        )

        emb_a = outputs['emb_a']
        emb_b = outputs['emb_b']

        # 拼接两个句向量
        combined = torch.cat([emb_a, emb_b], dim=-1)  # [batch, hidden*2]

        # 分类
        logits = self.classifier(combined)  # [batch, 2]

        result = {'logits': logits}

        # 计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result['loss'] = loss_fn(logits, labels)

        return result


def main():
    """测试模型"""
    from transformers import BertTokenizer

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建模型
    model = BiEncoder()

    # 模拟输入
    texts_a = ["喜欢打篮球的男生喜欢什么样的女生", "我手机丢了，我想换个手机"]
    texts_b = ["爱打篮球的男生喜欢什么样的女生", "我想买个新手机，求推荐"]

    # 编码
    enc_a = tokenizer(texts_a, padding=True, truncation=True, max_length=64, return_tensors='pt')
    enc_b = tokenizer(texts_b, padding=True, truncation=True, max_length=64, return_tensors='pt')

    # 前向传播
    with torch.no_grad():
        outputs = model(
            enc_a['input_ids'], enc_a['attention_mask'], enc_a['token_type_ids'],
            enc_b['input_ids'], enc_b['attention_mask'], enc_b['token_type_ids']
        )

    print(f"emb_a shape: {outputs['emb_a'].shape}")
    print(f"emb_b shape: {outputs['emb_b'].shape}")
    print(f"similarity: {outputs['similarity']}")


if __name__ == '__main__':
    main()
