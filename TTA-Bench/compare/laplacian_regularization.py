import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']        # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
# 生成模拟数据
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2) * 1.5

# 创建非线性可分数据 (螺旋分布)
theta = np.linspace(0, 4*np.pi, n_samples)
r = np.linspace(0.5, 2, n_samples)
X = np.column_stack([r*np.cos(theta) + np.random.randn(n_samples)*0.3,
                    r*np.sin(theta) + np.random.randn(n_samples)*0.3])
labels = (theta < 2*np.pi).astype(int)  # 两类标签

# 计算相似度矩阵 (动态带宽)
pairwise_dists = squareform(pdist(X))
sigma = np.median(pairwise_dists)
W = np.exp(-pairwise_dists**2 / (2 * sigma**2))

# 标签传播算法 (模拟正则化效果)
def label_propagation(W, labels, max_iter=20):
    smoothed = labels.copy().astype(float)
    for _ in range(max_iter):
        smoothed = W @ smoothed / np.sum(W, axis=1)
    return np.clip(smoothed, 0, 1)

smoothed_probs = label_propagation(W, labels)

# 生成决策边界网格
def generate_decision_boundary(X, y):
    clf = LogisticRegression().fit(X, y)
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    return xx, yy, Z.reshape(xx.shape)

# 可视化设置
plt.figure(figsize=(12, 6), dpi=100)
plt.suptitle("拉普拉斯正则化用于保持局部一致性", y=0.96, fontsize=22)
cmap = plt.cm.viridis

# 左图：无正则化
plt.subplot(1, 2, 1)
xx, yy, Z = generate_decision_boundary(X, labels)
plt.contourf(xx, yy, Z, levels=30, cmap=cmap, alpha=0.4)
sc = plt.scatter(X[:,0], X[:,1], c=labels, cmap=cmap,
                edgecolors='k', s=50, vmin=0, vmax=1)
plt.title("正则化前", pad=12, fontsize=20)  # 标题字号也调大
plt.xlabel("特征 1", fontsize=18)
plt.ylabel("特征 2", fontsize=18)
plt.tick_params(axis='both', labelsize=16)  # 刻度字号调大

# 右图：应用拉普拉斯正则化
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, levels=30, cmap=cmap, alpha=0.4)  # 保持相同决策边界
sc = plt.scatter(X[:,0], X[:,1], c=smoothed_probs, cmap=cmap,
                edgecolors='k', s=50, vmin=0, vmax=1)
plt.title("正则化后", pad=12, fontsize=20)
plt.xlabel("特征 1", fontsize=18)
plt.ylabel("特征 2", fontsize=18)  # 补充右图纵坐标标签
plt.tick_params(axis='both', labelsize=16)  # 刻度字号调大

# 添加连接线表示高相似性
top_edges = np.argsort(W.flatten())[-20:]  # 选择相似度最高的20条边
for edge_idx in top_edges:
    i, j = edge_idx // n_samples, edge_idx % n_samples
    if i < j:  # 避免重复绘制
        plt.subplot(1, 2, 1)
        plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]],
                'grey', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.subplot(1, 2, 2)
        plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]],
                'white', linestyle='-', linewidth=1.2, alpha=0.8)

# 添加颜色条
cax_ax = plt.axes([0.9, 0.15, 0.02, 0.7])
cbar=plt.colorbar(sc, cax=cax_ax)
cbar.set_label('分类置信度', fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()