"""
微型问答语料库

包含一些简单的问答对，用于训练问答模型。
"""

import os

label = ["问题：","答案："]
def load_corpus_from_txt(filepath="corpus.txt"):
    """从txt文件加载语料库

    Args:
        filepath (str): 语料库文件路径

    Returns:
        list: 包含问答对的字典列表
    """
    corpus = []
    current_question = None
    current_answer = None

    if not os.path.exists(filepath):
        return get_corpus()
    n = 1
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if n%2 == 1:
                line = label[0] + line
            else:
                line = label[1] + line
            n = n + 1
            if line.startswith("问题："):
                if current_question and current_answer:
                    corpus.append({
                        "question": current_question,
                        "answer": current_answer
                    })
                current_question = line[3:]
                current_answer = None
            elif line.startswith("答案："):
                current_answer = line[3:]

        if current_question and current_answer:
            corpus.append({
                "question": current_question,
                "answer": current_answer
            })

    return corpus if corpus else get_corpus()


corpus = []


def get_corpus():
    """返回默认语料库"""
    return corpus


def get_corpus_from_file(filepath="corpus.txt"):
    """从文件加载语料库

    优先从corpus.txt加载，如果文件不存在则返回默认语料库。

    Args:
        filepath (str): 语料库文件路径

    Returns:
        list: 包含问答对的字典列表
    """
    return load_corpus_from_txt(filepath)


def get_question_answer_pairs():
    """返回问答对列表

    优先从corpus.txt加载。

    Returns:
        list: 包含(question, answer)元组的列表
    """
    corpus_data = get_corpus_from_file("corpus.txt")
    return [(item["question"], item["answer"]) for item in corpus_data]