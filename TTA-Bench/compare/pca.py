import matplotlib
matplotlib.use('TkAgg')
# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 配置参数
id_list = ['5300-1', '5300-2', '5300-3']
class_idx = 0  # 选择要可视化的类别
n_subcarriers = 5  # 要显示的子载波数量

# # 创建画布
# fig, axes = plt.subplots(len(id_list), 1, figsize=(15, 10))
#
# for i, id in enumerate(id_list):
#     # 加载数据（假设数据已经预处理为numpy格式）
#     x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy').reshape(-1, 1, 2000, 30)
#     y_train = np.load(f'E:/wifi感知/{id}_npy/y_train.npy')
#
#     # 找到目标类别的第一个样本
#     sample_idx = np.where(y_train == class_idx)[0][0]
#     sample = x_train[sample_idx].squeeze()  # 形状 (2000, 30)
#
#     # 绘制前n个子载波
#     for sc in range(n_subcarriers):
#         axes[i].plot(sample[:, sc], label=f'Subcarrier {sc + 1}', alpha=0.7)
#
#     axes[i].set_title(f'Dataset {id} - Class {class_idx}')
#     axes[i].set_xlabel('Time Step')
#     axes[i].set_ylabel('Amplitude')
#     if i == 0:
#         axes[i].legend()
#
# plt.tight_layout()
# plt.show()

# # 同一类别热力图
# fig, axes = plt.subplots(1, 3, figsize=(20, 5))
#
# for i, id in enumerate(id_list):
#     x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy').reshape(-1, 1, 2000, 30)
#     y_train = np.load(f'E:/wifi感知/{id}_npy/y_train.npy')
#
#     sample_idx = np.where(y_train == class_idx)[0][0]
#     sample = x_train[sample_idx].squeeze().T  # 转置为(30, 2000)
#
#     im = axes[i].imshow(sample, aspect='auto', cmap='viridis',
#                         vmin=np.percentile(sample, 5),  # 统一颜色范围
#                         vmax=np.percentile(sample, 95))
#     axes[i].set_title(f'Dataset {id} - Class {class_idx}')
#     axes[i].set_xlabel('Time Step')
#     axes[i].set_ylabel('Subcarrier Index')
#     fig.colorbar(im, ax=axes[i])
#
# plt.tight_layout()
# plt.show()

# # 方差
# plt.figure(figsize=(12, 6))
#
# for id in id_list:
#     x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy').reshape(-1, 1, 2000, 30)
#
#     # 计算每个样本的方差（时间维度）
#     temporal_vars = []
#     for sample in x_train:
#         sample_2d = sample.squeeze()  # (2000, 30)
#         temporal_var = np.var(sample_2d, axis=0).mean()  # 计算每个子载波的方差后取平均
#         temporal_vars.append(temporal_var)
#
#     # 绘制分布曲线
#     plt.hist(temporal_vars, bins=50, alpha=0.5, label=id)
#
# plt.xlabel('Temporal Variance')
# plt.ylabel('Frequency')
# plt.title('Temporal Variance Distribution Comparison')
# plt.legend()
# plt.show()
#
# 特征空间
from sklearn.decomposition import PCA

# 合并所有数据集的特征
all_features = []
all_labels = []  # 用数字标签区分数据集

for idx, id in enumerate(id_list):
    x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy').reshape(-1, 1, 2000, 30)

    # 提取特征：每个样本的统计量
    features = []
    for sample in x_train:
        sample_2d = sample.squeeze()
        features.append([
            np.mean(sample_2d),  # 全局均值
            np.var(sample_2d),  # 全局方差
            np.max(sample_2d) - np.min(sample_2d)  # 动态范围
        ])
    all_features.extend(features)
    all_labels.extend([idx] * len(x_train))

# PCA降维
pca = PCA(n_components=2)
transformed = pca.fit_transform(all_features)

# 可视化
plt.figure(figsize=(10, 8))
for idx in range(len(id_list)):
    mask = np.array(all_labels) == idx
    plt.scatter(transformed[mask, 0], transformed[mask, 1],
                label=id_list[idx], alpha=0.6)
plt.xlabel('PCA 成分 1')
plt.ylabel('PCA 成分 2')
plt.title('特征空间分布 (PCA)')
plt.legend()
plt.show()