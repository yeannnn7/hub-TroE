"""
基于Qwen的文本匹配模型

支持LoRA微调的大语言模型文本匹配
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class QwenForTextMatching(nn.Module):
    """基于Qwen的文本匹配模型"""
    
    def __init__(
        self,
        model_path: str = "pretrain_models/Qwen2___5-0___5B-Instruct",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_labels: int = 2
    ):
        """
        初始化模型
        
        Args:
            model_path: 模型路径
            use_lora: 是否使用LoRA
            lora_r: LoRA秩
            lora_alpha: LoRA alpha参数
            lora_dropout: LoRA dropout
            num_labels: 分类数量
        """
        super(QwenForTextMatching, self).__init__()
        
        self.num_labels = num_labels
        self.use_lora = use_lora
        
        # 加载预训练模型
        print(f"加载模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 应用LoRA
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen的attention层
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA配置: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            # 统计参数
            trainable_params, all_params = self._count_parameters()
            print(f"可训练参数: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.2f}%)")
        
        # 分类头（可选，用于分类任务）
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
    def _count_parameters(self):
        """统计参数数量"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        return trainable_params, all_params
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于训练）
        
        Returns:
            模型输出
        """
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_response(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 50):
        """
        生成响应
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
        
        Returns:
            生成的文本
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id
            )
        
        return outputs
    
    def save_pretrained(self, output_dir: str):
        """保存模型"""
        if self.use_lora:
            # 保存LoRA适配器
            self.model.save_pretrained(output_dir)
            print(f"保存LoRA适配器到: {output_dir}")
        else:
            # 保存完整模型
            self.model.save_pretrained(output_dir)
            print(f"保存模型到: {output_dir}")
    
    def load_lora_adapter(self, adapter_path: str):
        """加载LoRA适配器"""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"加载LoRA适配器: {adapter_path}")


def load_model_and_tokenizer(
    model_path: str = "pretrain_models/Qwen2___5-0___5B-Instruct",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1
):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        use_lora: 是否使用LoRA
        lora_r: LoRA秩
        lora_alpha: LoRA alpha参数
        lora_dropout: LoRA dropout
    
    Returns:
        model, tokenizer
    """
    # 加载分词器
    print(f"加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = QwenForTextMatching(
        model_path=model_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    return model, tokenizer
