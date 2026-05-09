import os
import random

from cycler import K

'''
把创建训练集的动作单独放在这个程序中，方便区分
'''
def create_samples(sp_num):
    '''
    创建训练集
    '''
    t_data = []
    for _ in range(sp_num):
        _w = create_chinese()
        _i = _w.index("你")
        t_data.append((_w, _i))
    return t_data

def create_chinese():
    #用unicode编码生成4个汉字，再添加一个"你"，随机打乱顺序
    chinese_sentences = [chr(random.randint(0x4e00, 0x9fa5)) for _ in range(4)]
    chinese_sentences.append("你")
    random.shuffle(chinese_sentences)
    #返回五个字，其中带你字
    return "".join(chinese_sentences)

#print(create_samples(100))