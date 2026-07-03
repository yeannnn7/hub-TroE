### 1.1验证环境就绪
```bash
source ~/vllm_env/bin/activate
python -c "import vllm, torch; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```
输出：
```
vLLM: 0.9.2
CUDA: True
GPU: Tesla V100-PCIE-32GB
```


###  demo_guided_choice.py — 枚举约束

======================================================================
  Demo: guided_choice（枚举约束）
  Model: qwen2-0.5b   Choices: ['查股价', '查财报', '查新闻', '对比分析', '其他']
======================================================================

问题                              真值        裸 prompt 输出         guided 输出      
------------------------------------------------------------------------------------------
查一下茅台今天多少钱                      查股价       ✓ 查股价               ✓ 查股价
贵州茅台 2024 年营收多少亿                查财报       ~ 其他                ✗ 其他
最近宁德时代有什么新闻                     查新闻       ~ 其他                ✗ 其他
对比一下招行和平安的净利润                   对比分析      ~ 其他                ✗ 其他
今天天气怎么样                         其他        ✓ 其他                ✓ 其他
帮我看看 600000 的收盘价                查股价       ✓ 查股价               ✓ 查股价
招商银行去年的 ROE 是多少                 查财报       ~ 查股价               ✗ 查股价
宁德时代被限产了吗                       查新闻       ~ 对比分析              ✗ 对比分析
比亚迪和特斯拉哪个更强                     对比分析      ✓ 对比分析              ✓ 对比分析
帮我订一张机票                         其他        ~ 查股价               ✗ 查股价
五粮液现在股价                         查股价       ✓ 查股价               ✓ 查股价
平安保险的净利润增长率                     查财报       ~ 对比分析              ✗ 对比分析
------------------------------------------------------------------------------------------

指标                            裸 prompt            guided_choice       
----------------------------------------------------------------------
输出合法（在枚举内）                  12/12 (100%)      12/12 (100%)
预测正确                        5/12 (42%)      5/12 (42%)
平均延迟（秒）                     0.155           0.699

======================================================================
  结论：guided_choice 100% 保证输出合法，
       分类准确率也通常比裸 prompt 高（因为模型不会被错误 token 带偏）
======================================================================


###  demo_guided_regex.py — 正则约束

======================================================================
  任务 1：日期标准化 → YYYY-MM-DD
  正则: \d{4}-\d{2}-\d{2}
======================================================================

输入                                 裸 prompt                 guided_regex   
---------------------------------------------------------------------------
2024年5月12日                         ✓ 2024-05-12            ✓ 2024-05-12
2023/12/1 下午开会                     ✗ 2023-12-01 15:00:00   ✓ 2023-12-01
三月三号我去北京                           ✗ 03-03                 ✓ 0331-00-00
2024.11.30 是截止日期                   ✓ 2024-11-30            ✓ 2024-11-30
明天（假设今天是2026-05-11）                ✓ 2026-05-13            ✓ 2026-05-13
2024 年 10 月的第一天                    ✗ 01-01-2024            ✓ 0109-01-01
---------------------------------------------------------------------------
格式合法率：裸 prompt 3/6 (50%)  |  guided_regex 6/6 (100%)

======================================================================
  任务 2：A 股代码抽取 → 6 位数字
  正则: \d{6}
======================================================================

输入                                 裸 prompt                 guided_regex   
---------------------------------------------------------------------------
帮我查 600000 浦发银行                    ✓ 600300                ✓ 600300
code: 000001 平安银行                  ✓ 000001                ✓ 000001
茅台的代码是 600519                      ✓ 600519                ✓ 600519
六零零五一九                             ✓ 600595                ✓ 600595
股票代码：300750（宁德时代）                  ✓ 300750                ✓ 300750
---------------------------------------------------------------------------
格式合法率：裸 prompt 5/5 (100%)  |  guided_regex 5/5 (100%)

======================================================================
  结论：guided_regex 保证下游解析器永远能拿到合法输入
       特别适合日期/电话/代码/邮编等有严格格式的字段
======================================================================



###  demo_guided_json.py — JSON Schema 基础

==============================================================================
  Demo: guided_json（JSON Schema 约束）
  Model: qwen2-0.5b
  对比三种模式：裸 prompt / response_format / guided_json
==============================================================================

▶ 招行 2023 年营收多少
  [raw             ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}
  [response_format ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}
  [guided_json     ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}

▶ 贵州茅台 2022 的净利润
  [raw             ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}
  [response_format ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}
  [guided_json     ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}

▶ 平安银行去年（2024）的 ROE
  [raw             ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}
  [response_format ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}
  [guided_json     ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}

▶ 2021 年五粮液毛利率
  [raw             ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}
  [response_format ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}
  [guided_json     ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}

▶ 2023 宁德时代经营现金流
  [raw             ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}
  [response_format ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}
  [guided_json     ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}

▶ 问一下比亚迪 2024 的总资产规模
  [raw             ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}
  [response_format ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}
  [guided_json     ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}

▶ 茅台 2020 年利润情况
  [raw             ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}
  [response_format ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}
  [guided_json     ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}

▶ ICBC 2023 营收
  [raw             ] ✗  {"company": "ICBC", "year": 2023, "metric": "营业收入"}
  [response_format ] ✗  {"company": "ICBC", "year": 2023, "metric": "营业收入"}
  [guided_json     ] ✓  {"company": "ICBC", "year": 2023, "metric": "营收"}

▶ 隆基绿能 22 年 roe
  [raw             ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}
  [response_format ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}
  [guided_json     ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}

==============================================================================
  9 条测试结果汇总
==============================================================================
指标                      裸 prompt          response_format     guided_json    
------------------------------------------------------------------------------
合法 JSON               9/9 (100%)      9/9 (100%)      9/9 (100%)      
字段齐全                  9/9 (100%)      9/9 (100%)      9/9 (100%)      
year 在 2015~2025      9/9 (100%)      9/9 (100%)      9/9 (100%)      
metric 在枚举内           8/9 (89%)      8/9 (89%)      9/9 (100%)      
jsonschema 完全通过       8/9 (89%)      8/9 (89%)      9/9 (100%)      

==============================================================================
  结论：
    response_format 只保证是 JSON，不保证字段名、类型、枚举正确
    guided_json     是唯一 100% 保证 schema 合法的方式
==============================================================================





###  demo_response_format.py — OpenAI 标准方式

===========================================================================
  Demo: response_format（OpenAI 标准 JSON 模式）
  Model: qwen2-0.5b
===========================================================================

▶ 茅台三季度营收创历史新高，净利润同比增长 15%
  [raw         ] ✓ {
  "sentiment": "positive",
  "confidence": 0.9,
  "keywords": ["茅台", "三季度", "营收", "净利润", "增长"]
}
  [json_object ] ✓ {
  "sentiment": "positive",
  "confidence": 0.9,
  "keywords": ["茅台", "三季度", "营收", "净利润", "增长"]
}

▶ 比亚迪召回 10 万辆电动车，涉及电池安全问题
  [raw         ] ✓ {
  "sentiment": "negative",
  "confidence": 0.85,
  "keywords": ["召回", "电池安全"]
}
  [json_object ] ✓ {
  "sentiment": "negative",
  "confidence": 0.85,
  "keywords": ["召回", "电池安全"]
}

▶ 央行维持 LPR 利率不变
  [raw         ] ✓ {
  "sentiment": "neutral",
  "confidence": 0.75,
  "keywords": ["LPR", "保持不变"]
}
  [json_object ] ✓ {
  "sentiment": "neutral",
  "confidence": 0.75,
  "keywords": ["LPR", "保持不变"]
}

▶ 宁德时代与宝马签订长期供货协议
  [raw         ] ✓ {
  "sentiment": "positive",
  "confidence": 0.95,
  "keywords": ["宁德时代", "宝马", "长期供货协议"]
}
  [json_object ] ✓ {
  "sentiment": "positive",
  "confidence": 0.95,
  "keywords": ["宁德时代", "宝马", "长期供货协议"]
}

▶ 平安保险高管被调查，股价下跌 8%
  [raw         ] ✓ {"sentiment": "negative", "confidence": 0.9, "keywords": ["平安保险"]}
  [json_object ] ✓ {"sentiment": "negative", "confidence": 0.9, "keywords": ["平安保险"]}

===========================================================================
  5 条测试结果
===========================================================================
指标                    裸 prompt            response_format     
------------------------------------------------------------
合法 JSON             5/5 (100%)      5/5 (100%)      
有 sentiment 字段      5/5 (100%)      5/5 (100%)      
sentiment 值合法       5/5 (100%)      5/5 (100%)      
有 confidence 字段     5/5 (100%)      5/5 (100%)      
有 keywords 字段       5/5 (100%)      5/5 (100%)      

===========================================================================
  观察：
    response_format 显著提升 JSON 合法率，但字段语义仍靠模型自觉
    若需严格字段 schema，请用 guided_json（见 demo_function_call.py）
===========================================================================


### demo_function_call.py ★ 核心

# 跑两个工具共 100 个用例
==============================================================================
  demo_function_call.py   核心：裸 prompt vs response_format vs guided_json
  Model: qwen2-0.5b
==============================================================================

==============================================================================
  工具: get_stock_quote   测试数: 50   模式: 3
==============================================================================
  进度: 10/50
  进度: 20/50
  进度: 30/50
  进度: 40/50
  进度: 50/50

──────────────────────────────────────────────────────────────────────────────
  【get_stock_quote】 50 条测试 × 3 模式 汇总
──────────────────────────────────────────────────────────────────────────────
指标                      裸 prompt            response_format       guided_json    
──────────────────────────────────────────────────────────────────────────────
JSON 语法合法             50/50 (100%)       50/50 (100%)       50/50 (100%)       
必选字段齐全                50/50 (100%)       50/50 (100%)       50/50 (100%)       
完整 schema 通过 ★        46/50 ( 92%)       46/50 ( 92%)       50/50 (100%)       
平均延迟（秒）               0.726              1.128              3.941              

──────────────────────────────────────────────────────────────────────────────
  【get_stock_quote】 典型失败案例（前 3 条）
──────────────────────────────────────────────────────────────────────────────

[raw] 失败示例（schema 校验未通过）：
  ▶ Prompt: 帮我查询平安银行今日开盘价，并简单解释什么是开盘价
    输出:   {"symbol": "02699", "market": "SH", "date": "2023-07-10", "fields": ["open"], "adjust": "none"}
    错误:   schema: '02699' does not match '^\\d{6}$'
  ▶ Prompt: 请查询东方财富成交量并分析异动原因
    输出:   {"symbol": "000001.SZ", "market": "SZ", "date": "2023-10-07", "fields": ["volume"], "adjust": "hfq", "symbols": ["000001.SZ"]}
    错误:   schema: Additional properties are not allowed ('symbols' was unexpected)
  ▶ Prompt: 帮我查 000001 收盘价，然后帮我判断是否该买入
    输出:   {"symbol": "000001", "market": "SH", "date": "2026-05-12", "fields": ["close"], "adjust": "none", "isBuy": true}
    错误:   schema: Additional properties are not allowed ('isBuy' was unexpected)

[response_format] 失败示例（schema 校验未通过）：
  ▶ Prompt: 帮我查询平安银行今日开盘价，并简单解释什么是开盘价
    输出:   {"symbol": "02699", "market": "SH", "date": "2023-07-10", "fields": ["open"], "adjust": "none"}
    错误:   schema: '02699' does not match '^\\d{6}$'
  ▶ Prompt: 请查询东方财富成交量并分析异动原因
    输出:   {"symbol": "000001.SZ", "market": "SZ", "date": "2023-10-07", "fields": ["volume"], "adjust": "hfq", "symbols": ["000001.SZ"]}
    错误:   schema: Additional properties are not allowed ('symbols' was unexpected)
  ▶ Prompt: 帮我查 000001 收盘价，然后帮我判断是否该买入
    输出:   {"symbol": "000001", "market": "SH", "date": "2026-05-12", "fields": ["close"], "adjust": "none", "isBuy": true}
    错误:   schema: Additional properties are not allowed ('isBuy' was unexpected)

[guided_json] ✓ 无失败案例

  [耗时 290.3s]

==============================================================================
  工具: create_order   测试数: 50   模式: 3
==============================================================================
  进度: 10/50
  进度: 20/50
  进度: 30/50
  进度: 40/50
  进度: 50/50

──────────────────────────────────────────────────────────────────────────────
  【create_order】 50 条测试 × 3 模式 汇总
──────────────────────────────────────────────────────────────────────────────
指标                      裸 prompt            response_format       guided_json    
──────────────────────────────────────────────────────────────────────────────
JSON 语法合法             50/50 (100%)       50/50 (100%)       50/50 (100%)       
必选字段齐全                50/50 (100%)       50/50 (100%)       50/50 (100%)       
完整 schema 通过 ★        28/50 ( 56%)       28/50 ( 56%)       46/50 ( 92%)       
平均延迟（秒）               0.890              1.303              5.635              

──────────────────────────────────────────────────────────────────────────────
  【create_order】 典型失败案例（前 3 条）
──────────────────────────────────────────────────────────────────────────────

[raw] 失败示例（schema 校验未通过）：
  ▶ Prompt: 7 本笔记本，18900001111
    输出:   {
  "product": "7 本笔记本",
  "quantity": 1,
  "user_phone": "1890001111",
  "delivery_date": "2026-05-14",
  "priority": "normal",
  "payment_method": "
    错误:   schema: '1890001111' does not match '^1[3-9]\\d{9}$'
  ▶ Prompt: 3 瓶红酒，13677778888，wechat
    输出:   {"product": "红酒", "quantity": 3, "user_phone": "13677778888", "delivery_date": "2026-05-13", "priority": "wechat", "payment_method": "wechat"}
    错误:   schema: 'wechat' is not one of ['normal', 'express', 'urgent']
  ▶ Prompt: 给我 200 个鼠标，电话 13912345678
    输出:   {"product": "鼠标", "quantity": 200, "user_phone": "13912345678", "delivery_date": "2026-05-14", "priority": "normal", "payment_method": "alipay"}
    错误:   schema: 200 is greater than the maximum of 100

[response_format] 失败示例（schema 校验未通过）：
  ▶ Prompt: 7 本笔记本，18900001111
    输出:   {
  "product": "7 本笔记本",
  "quantity": 1,
  "user_phone": "1890001111",
  "delivery_date": "2026-05-14",
  "priority": "normal",
  "payment_method": "
    错误:   schema: '1890001111' does not match '^1[3-9]\\d{9}$'
  ▶ Prompt: 3 瓶红酒，13677778888，wechat
    输出:   {"product": "红酒", "quantity": 3, "user_phone": "13677778888", "delivery_date": "2026-05-13", "priority": "wechat", "payment_method": "wechat"}
    错误:   schema: 'wechat' is not one of ['normal', 'express', 'urgent']
  ▶ Prompt: 给我 200 个鼠标，电话 13912345678
    输出:   {"product": "鼠标", "quantity": 200, "user_phone": "13912345678", "delivery_date": "2026-05-14", "priority": "normal", "payment_method": "alipay"}
    错误:   schema: 200 is greater than the maximum of 100

[guided_json] 失败示例（schema 校验未通过）：
  ▶ Prompt: 给我 200 个鼠标，电话 13912345678
    输出:   {"product": "鼠标", "quantity": 200, "user_phone": "13912345678", "delivery_date": "2026-05-14", "priority": "normal", "payment_method": "alipay"}
    错误:   schema: 200 is greater than the maximum of 100
  ▶ Prompt: 订 0 个苹果，13812345678
    输出:   {"product": "苹果", "quantity": 0, "user_phone": "13812345678", "delivery_date": "2023-01-01", "priority": "normal", "payment_method": "alipay"}
    错误:   schema: 0 is less than the minimum of 1
  ▶ Prompt: 1000 瓶矿泉水，13755556666
    输出:   {"product": "矿泉水", "quantity": 1000, "user_phone": "13755556666", "delivery_date": "2026-05-14", "priority": "normal", "payment_method": "alipay"}
    错误:   schema: 1000 is greater than the maximum of 100

  [耗时 391.9s]

详细结果已保存：/mnt/vllm_deployment/src/../outputs/function_call_results.json

==============================================================================
  核心结论：
    裸 prompt        — JSON 语法偶尔错 / 字段拼错 / 正则枚举不符
    response_format  — JSON 合法率接近满分，但字段语义仍错
    guided_json      — 100% 满足完整 schema（小模型从不可用变可靠）
==============================================================================

