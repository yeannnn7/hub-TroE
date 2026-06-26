# 验证集结果

| 数据集 | 方法 | Accuracy | F1(weighted) | F1(pos) | 阈值 |
|---|---:|---:|---:|---:|---:|
| lcqmc:validation | majority | 0.5001 | 0.3335 | 0.6668 |  |
| lcqmc:validation | length_similarity | 0.5144 | 0.4631 | 0.6291 | 0.2517 |
| lcqmc:validation | char_jaccard | 0.6234 | 0.6129 | 0.6767 | 0.5800 |
| lcqmc:validation | tfidf_cosine | 0.6130 | 0.5963 | 0.6786 | 0.5100 |
| bq_corpus:validation | majority | 0.4978 | 0.3309 | 0.0000 |  |
| bq_corpus:validation | length_similarity | 0.5584 | 0.5571 | 0.5817 | 0.1700 |
| bq_corpus:validation | char_jaccard | 0.6870 | 0.6870 | 0.6874 | 0.1950 |
| bq_corpus:validation | tfidf_cosine | 0.6720 | 0.6718 | 0.6821 | 0.0700 |

