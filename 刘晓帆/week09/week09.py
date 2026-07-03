# 启动命令（直接在终端运行）
import subprocess
import sys

command = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2-1.5B-Instruct",
    "--trust-remote-code",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.9",
    "--port", "8000",
]

print("启动vLLM API服务器...")
print(" ".join(command))
subprocess.run(command)

import time
import aiohttp
import asyncio
from openai import OpenAI

# 测试REST API性能
async def test_api_performance():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )
    
    prompts = [
        "请用Python写一个快速排序算法：",
        "解释一下什么是机器学习：",
        "写一首关于秋天的五言绝句：",
        "用一句话总结红楼梦的主要内容：",
    ]
    
    # 并发请求测试
    start_time = time.time()
    
    async def send_request(prompt):
        response = client.completions.create(
            model="Qwen/Qwen2-1.5B-Instruct",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].text
    
    tasks = [send_request(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    for i, result in enumerate(results):
        print(f"\\n提示 {i+1}: {prompts[i][:50]}...")
        print(f"回复: {result[:100]}...")
    
    print(f"\\n并发处理4条请求总耗时: {total_time:.2f}秒")
    print(f"平均每条: {total_time/len(prompts):.2f}秒")

if __name__ == "__main__":
    asyncio.run(test_api_performance())
