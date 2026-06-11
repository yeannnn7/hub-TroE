"""
BertNER（线性头）和 BertCRFNER（CRF头）两个模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import transformers


def _load_bert(bert_path: str) -> BertModel:
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path, local_files_only=True)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。

    前向过程：
      input_ids → BertModel → last_hidden_state (B, L, 768)
               → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)

    损失：CrossEntropy，ignore_index=-100 跳过特殊token和非首子词
    """
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor, 
                labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        返回 logits: [B, L, num_labels]，未经 softmax（交叉熵 loss 内部做）
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        seq_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(seq_output))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100)

        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    与 BertNER 的区别：
      - Linear 输出称为 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加转移矩阵，用 Viterbi 找全局最优序列
      - 损失：负对数似然（CRF 内部计算前向-后向）
      - 解码：self.crf.decode() 返回保证合法的标签序列
    """
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels


    def _get_emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        return emissions: [B, L, num_labels]，未经 softmax（CRF 内部做）
        """
        return self.classifier(self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True).last_hidden_state))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            # torchcrf 不支持 -100；padding 位填 0，由 mask 排除，不参与 loss
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
        return emissions, loss

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> list[list[int]]:
        """Viterbi 解码：在 emissions 上找全局最优 BIO 路径（比逐 token argmax 更合法）。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        return self.crf.decode(emissions, mask=attention_mask.bool())


def build_model(use_crf: bool, bert_path: str, num_labels: int, dropout: float = 0.1) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model
