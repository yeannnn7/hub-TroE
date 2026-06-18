# 训练方式1:
python train_biencoder.py --data_dir ../data/lcqmc --loss cosine --pool mean --num_hidden_layers 3 --epochs 3

# 训练结果：
设备: cpu
实验名: lcqmc
Loss 类型: cosine  池化策略: mean  BERT 层数: 3  Epochs: 3

DataLoader 构建中...
  train :  18,847 条,   589 batch
  val   :   7,098 条,   222 batch
  test  :   8,074 条,   253 batch  (AFQMC test 无正样本，仅供参考)

构建模型...
Loading weights: 100%|█████████████████████████| 55/55 [00:00<00:00, 24346.88it/s]
模型: BiEncoder (pool=mean, layers=3)
参数量: 38.5M  (BERT 骨干: 38.5M)
总训练步数: 1767  Warmup 步数: 176
Epoch 1/3 | train_loss=0.2600 | val_acc=0.7010 val_f1=0.7002 threshold=0.92 | 573s
  ✓ 新最优模型已保存 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/lcqmc/outputs/lcqmc_checkpoints/biencoder_cosine_best.pt  (val_f1=0.7002)
Epoch 2/3 | train_loss=0.2100 | val_acc=0.7136 val_f1=0.7135 threshold=0.88 | 671s
  ✓ 新最优模型已保存 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/lcqmc/outputs/lcqmc_checkpoints/biencoder_cosine_best.pt  (val_f1=0.7135)
Epoch 3/3 | train_loss=0.1949 | val_acc=0.7102 val_f1=0.7100 threshold=0.91 | 698s

训练完成。最优 val_f1=0.7135
训练日志 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/lcqmc/outputs/lcqmc_logs/biencoder_cosine_log.json
最优 checkpoint → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/lcqmc/outputs/lcqmc_checkpoints/biencoder_cosine_best.pt

运行评估：python evaluate.py --model_type biencoder --ckpt /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/lcqmc/outputs/lcqmc_checkpoints/biencoder_cosine_best.pt
(py312) zhouyang@zy src % 