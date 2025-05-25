import numpy as np
import torch
from sklearn import svm
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
# 修正加载标签数据
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
# 修正加载标签数据
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')
#print(x_train.shape)#(3959,2000,30)

x_train = x_train.reshape(len(x_train),2000*30)#重塑 二维
x_test = x_test.reshape(len(x_test),2000*30)
# print(x_train.shape)


# 定义分类器，clf 意为 classifier，是分类器的传统命名
clf = svm.SVC(C=0.1, kernel='linear', gamma=0.6) # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
# 训练分类器
clf.fit(x_train, y_train)# 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 cls 里）
print('训练精度：', clf.score(x_train, y_train))  # 精度
y_train_true = clf.predict(x_train)#训练集原始标签

# clf.fit(x_test, y_test)
print('测试精度：', clf.score(x_test, y_test))
y_test_true = clf.predict(x_test)