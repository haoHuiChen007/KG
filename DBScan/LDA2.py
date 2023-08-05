# -------------计算困惑度-------------------------------------
import encodings

import gensim
import pandas as pd
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
from gensim.models import Word2Vec
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis.gensim_models

# 以下几行将train列表的每个元素都生成一个列表 形成列表嵌套
# 构建数据
model = Word2Vec.load(r"D:\py\KG\data_util\data\embedding_word2vec1.model")
words = model.wv.index_to_key

df = pd.read_excel(r"D:\py\KG\data\tfidf_clean.xlsx")
word = list(df[df.columns[0]].values)

keys = []
train_y = []
for i in tqdm(word):
    if i in words:
        keys.append(i)
        train_y.append(model.wv[i])

train = []
for i in range(len(keys)):
    train1 = []
    train1.append(keys[i])
    train.append(train1)

dictionary = corpora.Dictionary(train)  # 构建 document-term matrix
corpus = [dictionary.doc2bow(text) for text in train]
Lda = gensim.models.ldamodel.LdaModel


def perplexity(num_topics):
    print("======0========")
    ldamodel = Lda(corpus, num_topics=num_topics, id2word=dictionary, passes=50)  # passes为迭代次数，次数越多越精准
    print(ldamodel.print_topics(num_topics=num_topics, num_words=7))  # num_words为每个主题下的词语数量
    print(ldamodel.log_perplexity(corpus))
    print("======start========")
    return ldamodel.log_perplexity(corpus)


# 绘制困惑度折线图
print("======1========")
x = range(1, 10)  # 主题范围数量
print("======2========")
y = [perplexity(i) for i in x]
print("======3========")
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('困惑度大小')
print("======4========")
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
print("======5========")
plt.title('主题-困惑度变化情况')
plt.show()
plt.savefig('1.png')
print("ovesr aaa")






