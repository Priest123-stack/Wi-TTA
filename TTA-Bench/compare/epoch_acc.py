import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# G-SDFA算法准确率数据
accuracies_g_sdfa = [30.29, 31.28, 30.29, 30.01, 28.71, 28.01, 27.34, 26.79, 26.13, 25.56,
                     24.87, 23.92, 22.79, 21.72, 21.01, 19.90, 19.66, 16.41, 15.56, 15.90,
                     15.52, 15.50, 15.72, 15.72, 15.70, 15.68, 15.70, 15.58, 15.58, 15.54]

# 3C-GAN算法准确率数据
accuracies_3c_gan = [36.41, 14.63, 15.62, 14.97, 25.99, 24.13, 22.19, 25.20, 32.03, 28.89,
                     30.17, 37.77, 42.11, 43.73, 45.83, 47.57, 46.19, 44.57, 42.35, 40.13,
                     40.63, 37.72, 35.66, 33.46, 30.85, 31.26, 28.51, 28.35, 28.59, 26.31]

# SHOT算法准确率数据
accuracies_shot = [72.56, 73.17, 74.42, 76.12, 75.05, 74.68, 72.50, 74.38, 74.34, 74.24,
                   74.26, 75.91, 75.39, 74.08, 73.97, 74.90, 74.08, 72.98, 74.28, 73.37,
                   74.14, 72.24, 72.44, 73.09, 72.64, 73.23, 72.38, 70.50, 71.97, 72.44]

# 训练轮数
epochs = list(range(1, len(accuracies_g_sdfa) + 1))

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 绘制G-SDFA算法折线图
plt.plot(epochs, accuracies_g_sdfa, marker='o', linestyle='-', label='G-SDFA算法')

# 绘制3C-GAN算法折线图
plt.plot(epochs, accuracies_3c_gan, marker='s', linestyle='--', label='3C-GAN算法')

# 绘制SHOT算法折线图
plt.plot(epochs, accuracies_shot, marker='^', linestyle='-.', label='SHOT算法')

# 添加标题和标签
plt.title('TTDA类训练轮数与准确率的关系', fontsize=8)
plt.xlabel('训练轮数')
# plt.xticks(rotation=0)
plt.ylabel('准确率 (%)')
# 添加垂直虚线
# 添加水平虚线
y_position = 14.29
plt.axhline(y=y_position, color='r', linestyle='--')

# 添加标注
plt.text(plt.xlim()[0] + 0.1, y_position + 0.1, 'SHOT仅用伪标签交叉熵损失', fontsize=5)

# 设置网格线
plt.grid(True)

# 添加图例
plt.legend()
plt.tight_layout()
# 显示图形
plt.show()