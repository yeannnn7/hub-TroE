启动VLLM服务
import os

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

cmd = f"""
python -m vllm.entrypoints.openai.api_server \
    --model {MODEL} \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
"""

os.system(cmd)


性能测试
import time
from openai import OpenAI

times = []

for i in range(10):

    start = time.time()

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.7,
        max_tokens=300
    )

    end = time.time()

    cost = end-start

    print(f"第{i+1}次耗时：{cost:.3f}s")

    times.append(cost)

print()

print("平均耗时：",sum(times)/len(times))


Transformers版本
import time

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer(prompt,return_tensors="pt").to(model.device)

times=[]

for i in range(10):

    start=time.time()

    output=model.generate(
        **inputs,
        max_new_tokens=300
    )

    end=time.time()

    t=end-start

    print(i+1,t)

    times.append(t)

print("平均：",sum(times)/10)


速度对比
方法         平均耗时(s)

Transformers    2.96
vLLM            1.34

速度提升：
(2.96-1.34)/2.96≈54.7%
