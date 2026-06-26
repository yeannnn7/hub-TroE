# 测试集结果

| 数据集 | 方法 | Accuracy | F1(weighted) | F1(pos) | 阈值 |
|---|---:|---:|---:|---:|---:|
| lcqmc:test | majority | 0.5000 | 0.3333 | 0.6667 |  |
| lcqmc:test | length_similarity | 0.5550 | 0.5160 | 0.6533 | 0.2517 |
| lcqmc:test | char_jaccard | 0.6955 | 0.6877 | 0.7371 | 0.5800 |
| lcqmc:test | tfidf_cosine | 0.6324 | 0.6180 | 0.6922 | 0.5100 |
| bq_corpus:test | majority | 0.4916 | 0.3241 | 0.0000 |  |
| bq_corpus:test | length_similarity | 0.5615 | 0.5603 | 0.5863 | 0.1700 |
| bq_corpus:test | char_jaccard | 0.6908 | 0.6908 | 0.6957 | 0.1950 |
| bq_corpus:test | tfidf_cosine | 0.6708 | 0.6704 | 0.6849 | 0.0700 |


