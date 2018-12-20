import codecs
from gensim import corpora, models, similarities
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

import gensim.models as g
import logging
import os
import matplotlib.pyplot as plt
import random
import math

f=open("/Users/haydenwong/PycharmProjects/LDA/output/topic_number.txt")
number=f.read()
MAX = 1000000
train = []  # 训练数据
fp = codecs.open(r'/Users/haydenwong/PycharmProjects/LDA/output/output.txt', 'r', encoding='utf-8')
for line in fp:
    line = line.split()
    train.append([w for w in line])

dictionary = corpora.Dictionary(train)  # 构造词典
corpus = [dictionary.doc2bow(text) for text in train]  # 每个text对应的稀疏向量
tfidf = models.TfidfModel(corpus)  # 统计tfidf
corpus_tfidf = tfidf[corpus]
#  alpha，eta即为LDA公式中的α和β，minimum_probability表示主题小于某个值（比如0.001）就舍弃此主题。
lda = models.LdaModel(corpus_tfidf, num_topics=13, id2word=dictionary, alpha='auto', eta='auto', minimum_probability=0.01,passes=20,eval_every=5,iterations=2000)
# 文档-主题
lda_doc=[]
for doc_topic in lda.get_document_topics(corpus_tfidf):
    lda_doc.append(doc_topic)
#print(lda_doc)

word2={}

#主题-词
with open(r'/Users/haydenwong/PycharmProjects/LDA/output/wordlistOutput.txt', 'w', encoding='utf-8') as f1:
    lda_topic1=[]
    for topic_id in range(13):
        print('Topic', topic_id)
        lda_topic=lda.get_topic_terms(topicid=topic_id)
        lda_topic1.append(lda_topic)
        #print(lda.get_topic_terms(topicid=topic_id))  # lda生成的主题中的词分布，默认显示10个
        print(lda.show_topic(topicid=topic_id))
        word_list = lda.show_topic(topicid=topic_id)
        for i in range(5):
            f1.write(word_list[i][0]+'\n')

# 文档-词
inputs = open(r'/Users/haydenwong/PycharmProjects/LDA/output/output.txt', 'r', encoding='utf-8')
text_list = inputs.readlines()
#print(text_list)
