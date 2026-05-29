# 只跑 LSTM（无需预训练模型）
python train_lstm.py --epochs 10 --num_train 5000

# 只跑 BERT（需要 bert-base-chinese 模型）
python train_bert.py --epochs 3 --pool cls

# 只跑 Qwen2 SFT（需要 Qwen2-0.5B-Instruct 模型）
python train_llm.py --epochs 2 --lora_r 8

# 快速演示（少量数据）
python train_lstm.py --num_train 1000 --epochs 3