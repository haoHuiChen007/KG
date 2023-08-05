from matplotlib import colors, pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
# 导入数据预处理工具
# 对红酒数据集进行预处理
scaler = StandardScaler()
X = wine.data
y = wine.target
X_scaled = scaler.fit_transform(X)
# 处理后的数据集形态
print(X_scaled.shape)

# 导入PCA
# 设置主成分数量为 以便我们进行可视化
pca = PCA(n_components=0.8)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
# 打印主成分提取后的数据形态
print(X_pca.shape)

X0 = X_pca[wine.target == 0]
X1 = X_pca[wine.target == 1]
X2 = X_pca[wine.target == 2]
plt.scatter(X0[:, 0], X0[:, 1], c='b', s=60, edgecolor='k')
plt.scatter(X1[:, 0], X1[:, 1], c='g', s=60, edgecolor='k')
plt.scatter(X2[:, 0], X2[:, 1], c='r', s=60, edgecolor='k')

plt.legend(wine.target_names, loc='best')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()

# 使用主成分绘制热度图
plt.matshow(pca.components_, cmap='plasma')
# 纵轴为主成分数
plt.yticks([0, 1], ['component 1', 'component 2'])
plt.colorbar()
# 横轴为原始特征数量
plt.xticks(range(len(wine.feature_names)), wine.feature_names, rotation=60, ha='left')
plt.show()
