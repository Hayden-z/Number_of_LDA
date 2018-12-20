# -*- coding: utf-8 -*-
# @Time    : 17-8-4 上午9:26
# @Author  : 未来战士biubiu！！
# @FileName: test.py
# @Software: PyCharm Community Edition
# @Blog    ：http://blog.csdn.net/u010105243/article/
# Python3
import jieba


# jieba.load_userdict('userdict.txt')
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    jieba.load_userdict("/Users/haydenwong/PycharmProjects/LDA/input/newdict.txt")
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('/Users/haydenwong/PycharmProjects/topically-driven-language-model-python3/data/stop_words.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


inputs = open('/Users/haydenwong/PycharmProjects/LDA/input/input.txt', 'r', encoding='utf-8')
outputs = open('/Users/haydenwong/PycharmProjects/LDA/output/output.txt', 'w')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()