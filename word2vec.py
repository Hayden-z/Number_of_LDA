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

MAX = 1000000
####word2vec
output_dir="word2vec"
input_doc="/Users/haydenwong/PycharmProjects/LDA/output/output.txt"

#main
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
docs = g.word2vec.LineSentence(input_doc)
m =g.Word2Vec(docs, size=200, alpha=0.025, window=5, min_count=2,sample=1e-5,workers=4, min_alpha=0.0001, sg=0, hs=0, negative=5, iter=1000)
m.wv.save_word2vec_format(output_dir + "/text.model.bin", binary=True)
model = g.KeyedVectors.load_word2vec_format(output_dir + "/text.model.bin", binary=True)
#for i in len(open(input_doc).readline()):

word2vec=model.vectors

length = len(word2vec)
print(length)
tsne=TSNE()
word2vec1=tsne.fit_transform(word2vec)
print(word2vec1 )
np.savetxt(output_dir+"/word2vec.txt", word2vec1)

#x=[]
#y=[]
#for i in range(length):
   # x.append(word2vec1[i][0])
    #y.append(word2vec1[i][1])

#T = np.arctan2(y,x)
#plt.scatter(x, y, s=75, c=T, alpha=.5)

#plt.xlim(-10, 10)
#plt.xticks(())  # ignore xticks
#plt.ylim(-10, 10)
#plt.yticks(())  # ignore yticks

#plt.show()
c=[]
d=[]

for i in word2vec1:
    #print(i)
    a=i[0]
    b=i[1]

    c.append(a)
    d.append(b)
plt.scatter(c, d, s=75, alpha=.5)
plt.show()