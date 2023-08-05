import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from tqdm import tqdm

model = Word2Vec.load("../embedding/embedding.model")
words = model.wv.index_to_key

df = pd.read_excel("../data/tfidf_all.xlsx")
word = list(df[df.columns[0]].values)

keys = []
train_y = []
for i in tqdm(word):
    if i in words:
        keys.append(i)
        train_y.append(model.wv[i])

print("关键字数目{}".format(len(keys)))

# Z-score标注化
# 计算点与点直接的相似度


model_k = KMeans(
    n_clusters=4,
    random_state=0,
).fit(train_y)

# train_label = model_k.labels_
# train_cluster = model_k.cluster_centers_
# print(train_label[:10])
# print(train_cluster[0])
# 查看聚类结果
# pd.value_counts(train_label)
# 找出簇质心连续性变量坐标
# centroid_cluster_inversescale = pd.DataFrame(train_cluster).copy().iloc[:, :5]
# centroid_cluster_inversescale.columns = []
# metrics.silhouette_score()
# print("轮廓系数(Silhouette Coefficient): %0.4f" % metrics.silhouette_score(train_y, train_label))

for i in range(2, 10):
    model_k = KMeans(
        n_clusters=i,
        random_state=0,
    ).fit(train_y)
    train_label = model_k.labels_
    print("===========")
    print("分为" + str(i) + "类")
    print("轮廓系数(Silhouette Coefficient): %0.4f " % metrics.silhouette_score(train_y, train_label))
