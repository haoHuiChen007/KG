# 抓取平凡的世界各章节文本
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import os

if os.path.exists(r'worldofplainness.txt'):
    os.remove(r'worldofplainness.txt')
for j in range(3):
    print("第%d部" % (j + 1))
    for i in range(54):
        print("第%d章" % (i + 1))
        url = 'https://www.xstt5.com/mingzhu/1520/' + str(110319 + i + 54 * j) + '.html'
        data = requests.get(url)
        data.encoding = 'utf-8'
        soup = BeautifulSoup(data.text, "lxml")
        step1 = soup.find("div", attrs={'class': 'zw'})
        remove = re.compile(r'<.*?>|\r|\n|\t', re.S)  # 去掉标签等符号
        text = re.sub(remove, '', str(step1))
        with open("worldofplainness.txt", 'a', encoding='utf-8') as f:
            f.write("第%d部" % (j + 1) + "第%d章" % (i + 1))
            f.write(text)
            f.write('\n')
    f.close()

# 切词
if os.path.exists(r'output.txt'):
    with open("output.txt", "w") as f:
        f.write("")
    f.close()  # 清空txt文件

import thulac

thu1 = thulac.thulac(seg_only=True)
thu1.cut_f("worldofplainness.txt", "output.txt")


# 文本预处理
def process(data):
    split_chars = " …《》，、。？！；：“”‘’'\n\r-=—()（）.【】"
    for char in split_chars:
        data = " ".join(data.split(char))  ##将标点符号转化为空格分隔

    # 设置停用词
    stopwords = []
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                stopwords.append(line.strip())
    # 手动添加停用词
    stopwords.extend(['xiaoshuotxT', 'xt', 'NET', 'tXt天＿堂', '小＿',
                      'xiaoshuotx', 'net', 'wＷw．xiＡoshＵotx', 'Txt',
                      'Txt天', 'Txt天堂', '小说T', '＿天', '天堂', '小说'])

    words = []
    for word in data.split(' '):
        if len(word) > 1:  # 不考虑单字
            if word not in stopwords:
                words.append(word)
    return words


corpus = []
with open("output.txt", 'r', encoding='utf-8') as f:
    for line in f:
        corpus.append(process(line))

    # 词频统计
words = []
for i in range(len(corpus)):
    words.extend(corpus[i])

word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1
items = list(word_count.items())
items.sort(key=lambda x: x[1], reverse=True)

from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# word2vec
model = Word2Vec(list(set(words)), min_count=50, sg=1)
keys = model.wv.vocab.keys()
wordvector = []
for key in keys:
    wordvector.append(model[key])

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 手肘法确定k值
sse = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=100)
    km.fit_predict(wordvector)
    sse.append(km.inertia_)
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('K')
plt.ylabel('SSE')

# 轮廓系数法确定k值
plt.rcParams['font.sans-serif'] = ['SimHei']
Scores = []  # 存放轮廓系数
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=100)  # 构造聚类器
    km.fit_predict(wordvector)
    Scores.append(silhouette_score(wordvector, km.labels_, metric='euclidean'))
X = range(2, 11)
plt.xlabel('K')
plt.ylabel('轮廓系数')
plt.plot(X, Scores, 'o-')
plt.show()

# K-Means聚类
from time import time

print("clustering keywords ...")
t = time()

clf = KMeans(n_clusters=3)
s = clf.fit_predict(wordvector)
res = []
for i in range(3):
    label_i = []
    for j in range(0, len(s)):
        if s[j] == i:
            label_i.append(words[j])
    res.append(label_i)
    # 打印每个簇的关键词
    print('label_' + str(i) + ':' + str(label_i))
print("done in {0} seconds".format(time() - t))

# PCA对word2vec降维并可视化
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
decomposition_data = pca.fit_transform(wordvector)
x = []
y = []
for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])
ax = plt.axes()
plt.scatter(x, y, c=clf.labels_, marker="x")
plt.xticks(())
plt.yticks(())

" ".join(list(set(res[0])))

" ".join(list(set(res[1])))

" ".join(list(set(res[2])))