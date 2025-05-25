import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# import matplotlib.pyplot as plt

# 数据
algorithms = ['SHOT', 'G-SFDA', '3C-GAN', 'MEMO', 'GeOS', 'LAME', 'T3A', 'CoTTA', 'Tent', 'ETA', 'EcoTTA']
inference_times = [15.2, 18.7, 24.5, 22.1, 12.3, 3.2, 4.1, 16.9, 4.8, 5.3, 6.7]
parameter_ratios = [32.5, 41.8, 95.6, 100.0, 15.4, 0.0, 0.8, 25.3, 0.7, 0.9, 7.2]
baseline_time = 3.0

# 创建画布和第一个坐标轴
fig, ax1 = plt.subplots()

# 绘制单样本平均推理时间的折线图，添加圆圈标记
color = 'tab:red'
ax1.set_xlabel('算法',fontsize=14)
ax1.set_ylabel('单样本平均推理时间（ms）', color=color, fontsize=14)
line1 = ax1.plot(algorithms, inference_times, color=color, label='单样本平均推理时间（ms）', marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# 添加基线水平虚线
ax1.axhline(y=baseline_time, color='gray', linestyle='--', label='基线原始 CNN - 5 模型')

# 创建第二个坐标轴，共享 x 轴
ax2 = ax1.twinx()

# 绘制可训练参数量占比的折线图，添加三角形标记
color = 'tab:blue'
ax2.set_ylabel('可训练参数量占比（%）', color=color, fontsize=14)
line2 = ax2.plot(algorithms, parameter_ratios, color=color, label='可训练参数量占比（%）', marker='^')
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
lines = line1 + line2
all_lines = lines + [ax1.axhline(y=baseline_time, color='gray', linestyle='--')]
labels = [l.get_label() for l in lines] + ['原始CNN-5模型']
ax1.legend(all_lines, labels, loc='upper left')

# 设置标题
plt.title('算法性能指标对比')

# 显示图形
plt.show()
