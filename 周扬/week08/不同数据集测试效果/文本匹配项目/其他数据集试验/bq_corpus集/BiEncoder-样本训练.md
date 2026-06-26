
# 训练方式1:
python train_biencoder.py --loss cosine --pool mean --num_hidden_layers 3 --epochs 3
# 因电脑3层训练也非常慢，所以我把数据集直接减少了 ，上手试验下过程
  训练集目前20000条
  测试集目前4143条
  验证集目前783条

# 训练过程

构建模型...
Loading weights: 100%|█████████████████████████| 55/55 [00:00<00:00, 14495.84it/s]
模型: BiEncoder (pool=mean, layers=3)
参数量: 38.5M  (BERT 骨干: 38.5M)
总训练步数: 1875  Warmup 步数: 187
Epoch 1/3 | train_loss=0.2744 | val_acc=0.7566 val_f1=0.7562 threshold=0.64 | 662s
  ✓ 新最优模型已保存 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_checkpoints/biencoder_cosine_best.pt  (val_f1=0.7562)
Epoch 2/3 | train_loss=0.2180 | val_acc=0.7796 val_f1=0.7796 threshold=0.68 | 527s
  ✓ 新最优模型已保存 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_checkpoints/biencoder_cosine_best.pt  (val_f1=0.7796)
Epoch 3/3 | train_loss=0.2017 | val_acc=0.7865 val_f1=0.7865 threshold=0.69 | 512s
  ✓ 新最优模型已保存 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_checkpoints/biencoder_cosine_best.pt  (val_f1=0.7865)

训练完成。最优 val_f1=0.7865
训练日志 → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_logs/biencoder_cosine_log.json
最优 checkpoint → /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_checkpoints/biencoder_cosine_best.pt

运行评估：python evaluate.py --model_type biencoder --ckpt /Users/zhouyang/myworkspace/badou-nlp/周扬/week08/不同数据集测试效果/文本匹配项目/bq_corpus/outputs/bq_corpus_checkpoints/biencoder_cosine_best.pt

