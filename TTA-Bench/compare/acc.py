import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("D:/Python programs/formatted_accuracy.csv")

# ================ 关键修改：修正分类、颜色和线型 ================
# 生成 Source-Target 列（确保数据类型一致）
df["Target"] = df["Target"].str.replace("Target", "").astype(int)
df["Source"] = df["Source"].astype(int)
df["Source-Target"] = df["Source"].astype(str) + "-" + df["Target"].astype(str)

# 修正算法分类映射（独立"未适应时"）
algorithm_types = {
    # TTDA组
    "3C-GAN": "TTDA",
    "SHOT": "TTDA",
    "G-SDFA": "TTDA",
    # TTBA组
    "MEMO": "TTBA",
    "GeOS": "TTBA",
    # OTTA组
    "Tent": "OTTA",
    "EcoTTA": "OTTA",
    "CoTTA": "OTTA",
    "ETA": "OTTA",
    "LAME": "OTTA",
    "T3A": "OTTA",
    # 独立分类
    "未适应时": "Baseline"  # 新增独立分类
}
df["Category"] = df["Algorithm"].map(algorithm_types)

# 修正颜色映射（降低"未适应时"显著性）
algorithm_colors = {
    "3C-GAN": "#1f77b4",   # 蓝色
    "SHOT": "#aec7e8",     # 浅蓝
    "G-SDFA": "#ff7f0e",   # 橙色
    "MEMO": "#2ca02c",     # 绿色
    "GeOS": "#d62728",     # 红色
    "Tent": "#9467bd",     # 紫色
    "EcoTTA": "#8c564b",   # 棕色
    "CoTTA": "#e377c2",    # 粉色
    "ETA": "#7f7f7f",      # 灰色
    "LAME": "#17becf",     # 青色
    "T3A": "#bcbd22",      # 橄榄绿
    "未适应时": "black"   # 中灰色（原黑色改为低显著性）
}

# 修正线型定义（新增Baseline虚线）
line_styles = {
    "TTDA": (5, 5),        # 稀疏虚线
    "TTBA": (1, 5),        # 密集虚线
    "OTTA": (None, None),  # 实线
    "Baseline": (3, 3)     # 未适应时专用虚线
}

# 定义分类对应的标记符号
marker_map = {
    "TTDA": "o",
    "TTBA": "s",
    "OTTA": "^",
    "Baseline": "X"        # 特殊标记
}

# ================ 同一场景迁移图 ================
# plt.figure(figsize=(11, 6))
# ax1 = plt.gca()
#
# # 一次性绘制所有算法（包含未适应时）
# sns.lineplot(
#     data=df[df["Source"] == df["Target"]],
#     x="Source-Target",
#     y="Accuracy",
#     hue="Algorithm",
#     style="Category",
#     dashes=line_styles,
#     markers=marker_map,
#     palette=algorithm_colors,
#     linewidth=1.5,      # 统一线宽
#     markersize=8,
#     sort=False,
#     legend=False,
#     ax=ax1
# )
# ax1.tick_params(axis='both', labelsize=20)
# # 新增代码：设置坐标轴标签字号
# ax1.set_xlabel("迁移场景（Source-Target）", fontsize=20)
# ax1.set_ylabel("准确率（%）", fontsize=20)
# plt.ylim(80, 100)
# plt.title("同一场景迁移准确率对比（Source=Target）", fontsize=20)
# plt.grid(linestyle="--", alpha=0.6)
#
# ================ 跨场景迁移图 ================
plt.figure(figsize=(11, 6))
ax2 = plt.gca()

sns.lineplot(
    data=df[df["Source"] != df["Target"]],
    x="Source-Target",
    y="Accuracy",
    hue="Algorithm",
    style="Category",
    dashes=line_styles,
    markers=marker_map,
    palette=algorithm_colors,
    linewidth=1.5,
    markersize=8,
    sort=False,
    legend=False,
    ax=ax2
)
ax2.tick_params(axis='both', labelsize=20)
# 新增代码：设置坐标轴标签字号
ax2.set_xlabel("迁移场景（Source-Target）", fontsize=20)
ax2.set_ylabel("准确率（%）", fontsize=20)
plt.ylim(0, 100)
plt.title("跨场景迁移准确率对比（Source≠Target）", fontsize=20)
plt.grid(linestyle="--", alpha=0.6)

# ================ 统一图例（增强可读性） ================
legend_elements = [
    # 线型图例
    Line2D([0], [0], color="gray", lw=2, dashes=line_styles["TTDA"], label="TTDA（稀疏虚线）"),
    Line2D([0], [0], color="gray", lw=2, dashes=line_styles["TTBA"], label="TTBA（密集虚线）"),
    Line2D([0], [0], color="gray", lw=2, dashes=line_styles["OTTA"], label="OTTA（实线）"),
    Line2D([0], [0], color="gray", lw=2, dashes=line_styles["Baseline"], label="未适应时（虚线）"),
    # 算法颜色图例（动态生成）
    *[Line2D([0], [0],
            marker=marker_map[algorithm_types.get(algo, "OTTA")],  # 根据分类选择标记
            color=algorithm_colors[algo],
            label=algo,
            linestyle="None",
            markersize=8)
      for algo in algorithm_colors.keys()]
]

# 添加图例（统一位置和样式）
# for ax in [ax1, ax2]:
#     ax.legend(
#         handles=legend_elements,
#         bbox_to_anchor=(1.35, 1),
#         loc="upper left",
#         frameon=False,
#         ncol=1,
#         fontsize=16
#     )
# for ax in [ax1, ax2]:
# ax1.legend(
#     handles=legend_elements,
#     bbox_to_anchor=(1, 1),
#     loc="upper left",
#     frameon=False,
#     ncol=1,
#     fontsize=16
# )
    # # 为第二个图添加右侧图例（新增代码）
ax2.legend(
    handles=legend_elements,
    bbox_to_anchor=(1, 1),  # 调整右侧位置
    loc="upper left",  # 定位到右上角
    frameon=False,
    ncol=1,
    fontsize=16  # 统一字号
)

plt.tight_layout()
plt.show()