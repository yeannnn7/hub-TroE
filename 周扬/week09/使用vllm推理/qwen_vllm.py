from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# 本地模型路径
model_path = "/Users/zhouyang/myworkspace/badou-nlp/model/Qwen2.5-0.5B-Instruct"

# 先加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 推理提示词
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "介绍下自注意力机制"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=128,
)


# 加载 vLLM 模型
llm = LLM(
    model=model_path,
    trust_remote_code=True,
)


# 开始生成
outputs = llm.generate([prompt], sampling_params)


answer = outputs[0].outputs[0].text


print("提示词：")
print(prompt)

print("\n模型回答：")
print(answer)
