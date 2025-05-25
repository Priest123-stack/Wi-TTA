import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("D:/Python programs/formatted_accuracy.csv")

# 强制指定算法顺序（按原始表格顺序）
original_algo_order = [
    "未适应时", "SHOT", "G-SDFA", "3C-GAN",
    "MEMO", "GeOS", "LAME", "T3A",
    "ETA", "CoTTA", "EcoTTA", "Tent"
]

# -------------------------------- 数据处理 --------------------------------
df = df.copy()
df["Source"] = df["Source"].astype(int)
df["Target"] = df["Target"].str.replace("Target", "").astype(int)
df["Std"] = df["Std"].astype(float)

# 筛选跨场景数据
cross_df = df.loc[df["Source"] != df["Target"], :].copy()
cross_df.loc[:, "Source-Target"] = cross_df["Source"].astype(str) + "-" + cross_df["Target"].astype(str)
custom_order = ["1-2", "1-3", "2-1", "2-3", "3-1", "3-2"]
cross_df.loc[:, "Source-Target"] = pd.Categorical(
    cross_df["Source-Target"],
    categories=custom_order,
    ordered=True
)
cross_df = cross_df.sort_values("Source-Target")

# -------------------------------- 可视化设置 --------------------------------
algorithm_style = {
    "SHOT": {"hatch": "////", "edgecolor": "darkred", "linewidth": 1.5},
    "G-SDFA": {"hatch": "////", "edgecolor": "darkred", "linewidth": 1.5},
    "3C-GAN": {"hatch": "////", "edgecolor": "darkred", "linewidth": 1.5},
    "MEMO": {"hatch": "....", "edgecolor": "navy", "linewidth": 1.5},
    "GeOS": {"hatch": "....", "edgecolor": "navy", "linewidth": 1.5},
    "default": {"hatch": "", "edgecolor": "black", "linewidth": 1}
}

palette = sns.color_palette("husl", n_colors=6)
global_mean = cross_df["Std"].mean()

# -------------------------------- 绘图逻辑 --------------------------------
fig, axes = plt.subplots(6, 2, figsize=(15, 20))
plt.subplots_adjust(bottom=0.08, right=0.9,left=0.1, hspace=0.3, wspace=0.25)  # 关键调整：right=0.8

for idx, algo in enumerate(original_algo_order):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    algo_data = cross_df[cross_df["Algorithm"] == algo]

    if not algo_data.empty:
        bars = ax.bar(
            x=range(len(custom_order)),
            height=algo_data.sort_values("Source-Target")["Std"],
            color=palette,
            edgecolor='black'
        )

        style = algorithm_style.get(algo, algorithm_style["default"])
        for bar in bars:
            bar.set_hatch(style["hatch"])
            bar.set_edgecolor(style["edgecolor"])
            bar.set_linewidth(style["linewidth"])

        ax.axhline(
            y=global_mean,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            zorder=0
        )

        if row < 5:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(6))
            ax.set_xticklabels(custom_order, fontsize=20)
            ax.tick_params(axis='x', which='both', length=3)

        ax.set_ylim(0, cross_df["Std"].max() * 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        # 在绘图循环内部添加以下代码（在ax.grid之后）
        ax.tick_params(axis='y', labelsize=18)  # 纵坐标刻度字号设为18

    ax.set_title(f"{algo}", fontsize=15, pad=2)
    ax.set_ylabel("方差 (%)" if col == 0 else "", fontsize=15)

# 全局横坐标标签
fig.text(0.5, 0.03, "源域-目标域",
         ha='center', va='center',
         fontsize=20, fontweight='bold')

# =============== 图例系统 ===============
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='darkred',
              hatch="////", linewidth=1.5, label='TTDA (SHOT/G-SDFA/3C-GAN)'),
    Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='navy',
              hatch="....", linewidth=1.5, label='TTBA (MEMO/GeOS)'),
    Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black',
              linewidth=1, label='OTTA (其他算法)'),
    plt.Line2D([], [], color='gray', linestyle='--', label='全局平均方差')
]

fig.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.02, 0.9),  # 调整后的精准定位
    loc="upper left",
    title="图例说明",
    title_fontsize=10,
    fontsize=20,
    frameon=False
)

plt.suptitle("不同TTA算法跨场景方差分析", y=0.995, fontsize=20)
plt.show()