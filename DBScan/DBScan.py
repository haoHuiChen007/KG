import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec

from sklearn import cluster, datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
#
# np.random.seed(0)
#
# # 构建数据
# n_samples = 1500
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
#
# # noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
# # noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
# # blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#
# data_sets = [
#     (
#         train_y,
#         {
#             "eps": 0.3,
#             "min_samples": 5
#         }
#     )
# ]
#
#
# colors = ["#377eb8"]
#
# plt.figure(figsize=(15, 5))
#
# for i_dataset, (dataset, algo_params) in enumerate(data_sets):
#     # 模型参数
#     params = algo_params
#
#     # 数据
#     X, y = dataset
#     X = StandardScaler().fit_transform(X)
#
#     # 创建DBSCAN
#     dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params['min_samples'])
#
#     # 训练
#     dbscan.fit(X)
#
#     # 预测
#     y_pred = dbscan.labels_.astype(int)
#
#     y_pred_colors = []
#
#     for i in y_pred:
#         y_pred_colors.append(colors[i])
#
#     plt.subplot(1, 3, i_dataset + 1)
#
#     plt.scatter(X[:, 0], X[:, 1], color=y_pred_colors)
#
# plt.show()
#
# # 迭代不同的eps值


# import numpy as np
# import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from gensim.models import Word2Vec
# from tqdm import tqdm
#
# start = time.time()
# model = Word2Vec.load("../embedding/embedding.model")
# words = model.wv.index_to_key
#
# df = pd.read_excel("../data/tfidf_all.xlsx")
# word = list(df[df.columns[0]].values)
#
# keys = []
# train_y = []
# for i in tqdm(word):
#     if i in words:
#         keys.append(i)
#         train_y.append(model.wv[i])
#
# print("关键字数目{}".format(len(keys)))
# 读取数据
# read_df = pd.read_csv('../../datas/cluster_500-10_7.csv')
#
# target = read_df.iloc[:, -1]
# data = read_df.iloc[:, 1:-1]
#
# k = 7
# n = data.shape[0]
#
# # 初始化dis矩阵
# dis = np.zeros([n, n])
# # 求两两簇（点）之间的距离
# for i in range(n - 1):
#     for j in range(i + 1, n):
#         dis[j][i] = ((data.iloc[j] - data.iloc[i]) ** 2).sum()
#     print("初始化dis矩阵进度：{}/{}".format(i + 1, n))
# # 下三角复制到上三角
# i_lower = np.triu_indices(n, 0)
# dis[i_lower] = dis.T[i_lower]
# print("初始化dis矩阵进度：{}/{}".format(n, n))
#
#
# def Exactitude(pre_target, c_num):
#     """
#     Exactitude的相关定义放在了完整的项目代码中（文末查看）此处不影响使用
#     完全预测正确返回0
#     """
#     pass
#
#
# ######### 以下是重中之重 #########
#
# def regionQuery(p, dis, Eps):
#     """
#     返回点p的密度直达点
#     """
#     neighbors = np.where(dis[:, p] <= Eps ** 2)[0]
#     return neighbors
#
#
# def growCluster(dis, pre_target, labels, p, Eps, MinPts):
#     """
#     寻找p点的所有密度可达点，形成最终一个簇
#     输入：距离矩阵、预测标签、初始点p、是否被遍历过的标签、邻域半径、邻域中数据对象数目阈值
#     """
#
#     # 如果该点已经经过遍历，结束对该点的操作
#     if labels[p] == -1:
#         return labels, pre_target
#
#     # p的密度直达点
#     NeighborPts = regionQuery(p, dis, Eps)
#
#     # 遍历p的密度直达点
#     i = 0
#     while i < len(NeighborPts):
#         Pn = NeighborPts[i]
#         # 找出Pn的密度直达点
#         PnNeighborPts = regionQuery(Pn, dis, Eps)
#         # 如果此时的点是核心点
#         if len(PnNeighborPts) >= MinPts:
#             # 将点Pn的新的密度直达点加入点簇
#             Setdiff1d = np.setdiff1d(PnNeighborPts, NeighborPts)  # 在PnNeighborPts不在NeighborPts中
#             NeighborPts = np.hstack((NeighborPts, Setdiff1d))
#         # 否则，说明为边界点，什么也不需要做
#         # NeighborPts = NeighborPts
#         i += 1
#
#     # 将点p密度可达各点归入p所在簇
#     pre_target[NeighborPts] = pre_target[p]
#     labels[NeighborPts] = -1
#     return labels, pre_target
#
#
# def DBSCAN(n, k, dis, Eps, MinPts, mode=2):
#     """
#     输入：距离矩阵、邻域半径、邻域中数据对象数目阈值
#     输出：mode==1:预测值准确性（平均标准差），运行时间;mode==2:预测值
#     """
#     temp_start = int(round(time.time() * 1000000))
#
#     p = 0
#     labels = np.zeros(n)  # 有两个可能的值：-1：完成遍历的；0：这个点还没经历过遍历，初始均为0
#     pre_target = np.arange(n)
#
#     if mode == 2:
#         print("开始循环迭代")
#
#     # 从第一个点开始遍历
#     while p < n:
#         # 寻找当前点的密度可达点，形成一个簇
#         labels, pre_target = growCluster(dis, pre_target, labels, p, Eps, MinPts)
#         # 此时的簇数
#         c_num = len(np.unique(pre_target))
#         if mode == 2:
#             print("循环迭代次数：{}，此时有{}个簇".format(p + 1, c_num))
#         # 分成小于k簇直接跳出循环（说明分得有问题）
#         # 分成正好k簇也跳出循环，直接去检查有没有分对
#         if c_num <= k:
#             break
#         p += 1
#
#     if mode == 2:
#         print("结束循环迭代")
#
#     temp_stop = int(round(time.time() * 1000000))
#
#     if mode == 1:
#         return Exactitude(pre_target, c_num), temp_stop - temp_start
#     elif mode == 2:
#         return pre_target
#
#
# ######### 以上是重中之重 #########
#
# # 经过观察，Eps=4.0,MinPts=29可作为参数传入，
# # 准确率100%
# # 再次提示，测试、参数调整过程及可视化所用相关在文末完整项目中提供
# # pre_target = DBSCAN(n=n, k=k, dis=dis, Eps=4.0, MinPts=29, mode=1)
#
# pre_target = DBSCAN(n=n, k=k, dis=dis, Eps=4.0, MinPts=29)
#
# # pca降维
#
# pca = PCA(n_components=2)
# newData = pca.fit_transform(data)
# newData = pd.DataFrame(newData)
#
# # 可视化
#
# x = np.array(newData.iloc[:, 0])
# y = np.array(newData.iloc[:, 1])
#
# # 原数据
# plt.subplot(2, 1, 1)
# plt.scatter(x, y, c=np.array(target))
# # 预测数据
# plt.subplot(2, 1, 2)
# plt.scatter(x, y, c=pre_target)
# plt.show()
#
# end = time.time()
# print(end - start)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn import metrics

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

# model_d = DBSCAN(eps=1.5, min_samples=4).fit(train_y)
# train_label = model_d.labels_
#
# df = pd.value_counts(train_label)
# print(df)
# plt.figure(figsize=(5, 5))
# df.plot.bar(rot=0)
# print("over!")
# metrics.silhouette_score(train_y[train_label > -1], train_label[train_label > -1])
# 最小簇的个数
# min_samples_grid = [1, 2, 3, 4]
# # 簇数
# cluster_number = []
# # 轮廓系数
# slt_score = []
# # 噪声点
# noise_count = []
# for item in min_samples_grid:
#     for eps in tqdm([0.25, 0.5, 0.75]):
#         model_d = DBSCAN(eps=eps, min_samples=item).fit(train_y)
#         cluster_number.append(len(np.unique(model_d.labels_)) - 1)
#         slt_score.append(metrics.silhouette_score(train_y[model_d.labels_ > -1],
#                                                   model_d.labels_[model_d.labels_ > -1]))
#         noise_count.append((model_d.labels_ == -1).sum())
#
# plt.plot(min_samples_grid, cluster_number, 'r-*', linewidth=2)
# plt.xlabel('最小样本数')
# plt.ylabel('簇的个数')
# plt.title('不同最小样本数下聚类的簇的个数')



# DBSCAN
res = []
# 迭代不同的eps值
for eps in tqdm([0.25, 0.5, 0.75]):
    # 迭代不同的min_samples值
    for min_samples in range(2, 10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # 模型拟合
        dbscan.fit(train_y)
        # 统计各参数组合下的聚类个数（-1表示异常点）
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # 异常点的个数
        outlines = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
        # 统计每个簇的样本个数
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        res.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outlines': outlines,
                    'stats': stats})

# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)
# 根据条件筛选合理的参数组合
print(df.loc[df.n_clusters == 3, :])
#
