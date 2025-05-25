

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
id_list = ['5300-1', '5300-2', '5300-3']
n_classes = 7
class_names = ['无动作', '跳跃', '拾取', '跑步', '坐下', '行走', '挥手']

# 创建子图布局
fig, axes = plt.subplots(n_classes, len(id_list), figsize=(22, 32))
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.08, top=0.92, hspace=0.15, wspace=0.15)

# ===== 动态计算列标题位置 =====
# 获取子图区域的左右边界
plot_area = fig.subplotpars
left_bound = plot_area.left
right_bound = plot_area.right

# 计算每列的中心位置
num_cols = len(id_list)
col_centers = np.linspace(
    left_bound + (right_bound - left_bound) / (2 * num_cols),
    right_bound - (right_bound - left_bound) / (2 * num_cols),
    num_cols
)

# 添加列标题
for idx, (id_name, x_pos) in enumerate(zip(id_list, col_centers)):
    fig.text(
        x=x_pos,
        y=0.94,  # 标题垂直位置（靠近顶部）
        s=f"数据集 {id_name}",
        ha='center',  # 水平居中
        va='center',
        fontsize=18,
        fontweight='bold'
    )

# ===== 行标签设置（左侧）=====
for row in range(n_classes):
    fig.text(
        x=0.09,  # 水平位置调整，靠近子图左边界
        y=0.88 - row * 0.125,
        s=class_names[row],
        ha='right',
        va='center',
        fontsize=18,
        transform=fig.transFigure
    )

# ===== 主绘图部分（保持不变）=====
# 计算全局颜色范围
global_min = []
global_max = []
for id in id_list:
    x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy')
    global_min.append(np.percentile(x_train, 5))
    global_max.append(np.percentile(x_train, 95))
vmin, vmax = np.min(global_min), np.max(global_max)

# 绘制热力图
for row in range(n_classes):
    for col in range(len(id_list)):
        id = id_list[col]
        ax = axes[row, col]

        # 加载数据
        x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy').reshape(-1, 1, 2000, 30)
        y_train = np.load(f'E:/wifi感知/{id}_npy/y_train.npy')

        # 获取样本
        class_samples = np.where(y_train == row)[0]
        if len(class_samples) == 0:
            ax.axis('off')
            continue

        sample = x_train[class_samples[0]].squeeze().T

        # 绘制热力图
        im = ax.imshow(sample, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        # 设置刻度
        ax.tick_params(axis='both', which='both', length=0)
        if col == 0:  # 最左列显示子载波刻度
            ax.set_yticks([0, 10, 20, 29])
            ax.set_yticklabels(['0', '10', '20', '29'], fontsize=20)
        else:
            ax.set_yticks([])

        if row == n_classes - 1:  # 最底行显示时间步刻度
            ax.set_xticks([0, 500, 1000, 1500, 1999])
            ax.set_xticklabels(['0', '500', '1000', '1500', '2000'], fontsize=20)
        else:
            ax.set_xticks([])

# ===== 全局坐标标签 =====
fig.text(
    x=left_bound * 0.7,  # 纵坐标标签位置
    y=0.5,
    s='子载波索引',
    rotation=90,
    va='center',
    ha='center',
    fontsize=18
)

fig.text(
    x=left_bound + (right_bound - left_bound) / 2,  # 横坐标标签居中
    y=0.03,
    s='时间步',
    va='center',
    ha='center',
    fontsize=18
)

# ===== 颜色条 =====
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('信号幅度', fontsize=22)
cbar.ax.tick_params(labelsize=20)

plt.show()