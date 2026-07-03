import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


# 本地模型路径
model_path = "/Users/zhouyang/myworkspace/badou-nlp/model/Qwen2.5-0.5B-Instruct"


# 先判断用什么设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("当前设备:", device)


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)


# 加载模型
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    model = model.to(device)


# 提示词
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "介绍下自注意力机制"},
]

total_start_time = time.perf_counter()

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# 文本转成 token
model_inputs = tokenizer([text], return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}


# 开始推理
generate_start_time = time.perf_counter()
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
    )
generate_end_time = time.perf_counter()


# 
new_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]


# 解码成文字
answer = tokenizer.decode(new_ids, skip_special_tokens=True)
total_end_time = time.perf_counter()


print("\n模型回答：")
print(answer)
print("\n耗时统计：")
print(f"generate耗时: {generate_end_time - generate_start_time:.4f} 秒")
print(f"总耗时: {total_end_time - total_start_time:.4f} 秒")
