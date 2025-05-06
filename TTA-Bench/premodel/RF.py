import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ptflops import get_model_complexity_info

# 正确加载数据
x_train = np.load('E:/wifi感知/5300-2_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-2_npy/y_train.npy')  # 假设这是正确的标签文件
x_test = np.load('E:/wifi感知/5300-2_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-2_npy/y_test.npy')  # 假设这是正确的标签文件

x_train = x_train.reshape(len(x_train), 2000 * 30)  # 重塑为二维
x_test = x_test.reshape(len(x_test), 2000 * 30)

# 确保 y_train 和 y_test 是一维数组
y_train = y_train.ravel()
y_test = y_test.ravel()

# 定义分类器
rf = RandomForestClassifier(n_estimators=100)  # n_estimators 森林中决策树的数量

# 训练分类器
rf.fit(x_train, y_train)

# 估算参数数量
total_params = 0
for tree in rf.estimators_:
    # 每个内部节点需要一个特征索引和一个阈值，每个叶节点需要一个类别标签
    total_params += tree.tree_.node_count * (1 + 1 + 1)

params_M = total_params / 1e6
print(f"Params (M): {params_M:.2f}")

# 估算FLOPs
total_flops = 0
num_samples = len(x_train)
for tree in rf.estimators_:
    # 每个样本在一棵决策树中的FLOPs近似为树的深度
    total_flops += num_samples * tree.tree_.max_depth

flops_M = total_flops / 1e6
print(f"FLOPs (M): {flops_M:.2f}")

print('训练精度：', rf.score(x_train, y_train))  # 精度
print('测试精度：', rf.score(x_test, y_test))