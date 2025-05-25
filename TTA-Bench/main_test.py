import matplotlib
matplotlib.use('TkAgg')

import torch
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import logging
import math
import OTTA.cotta as cotta
import OTTA.EATA as EATA
import OTTA.tent as tent
import OTTA.ecotta as ecotta
import OTTA.norm as norm
import TTDA.shot as shot
from TTBA.GeoS_1 import GEOS
from TTDA import gsdfa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载目标域数据集
target_id = '5300-1'
x_train_target = np.load('E:/wifi感知/' + target_id + '_npy/x_train.npy')
y_train_target = np.load('E:/wifi感知/' + target_id + '_npy/y_train.npy')
x_test_target = np.load('E:/wifi感知/' + target_id + '_npy/x_test.npy')
y_test_target = np.load('E:/wifi感知/' + target_id + '_npy/y_test.npy')

# 加载源域数据集
source_id = '5300-3'  # 假设源域数据集为 5300-1
x_train_source = np.load('E:/wifi感知/' + source_id + '_npy/x_train.npy')
y_train_source = np.load('E:/wifi感知/' + source_id + '_npy/y_train.npy')
x_test_source = np.load('E:/wifi感知/' + source_id + '_npy/x_test.npy')
y_test_source = np.load('E:/wifi感知/' + source_id + '_npy/y_test.npy')

# 转换为张量
x_train_target = torch.tensor(x_train_target.reshape(len(x_train_target), 1, 2000, 30)).float()
y_train_target = torch.tensor(y_train_target)
x_test_target = torch.tensor(x_test_target.reshape(len(x_test_target), 1, 2000, 30)).float()
y_test_target = torch.tensor(y_test_target)

x_train_source = torch.tensor(x_train_source.reshape(len(x_train_source), 1, 2000, 30)).float()
y_train_source = torch.tensor(y_train_source)
x_test_source = torch.tensor(x_test_source.reshape(len(x_test_source), 1, 2000, 30)).float()
y_test_source = torch.tensor(y_test_source)

# 批量化
train_dataset_target = TensorDataset(x_train_target, y_train_target)
train_loader_target = DataLoader(dataset=train_dataset_target, batch_size=30, shuffle=True)
test_dataset_target = TensorDataset(x_test_target, y_test_target)
test_loader_target = DataLoader(dataset=test_dataset_target, batch_size=30, shuffle=True)
combined_dataset_target = torch.utils.data.ConcatDataset([train_dataset_target, test_dataset_target])
combined_loader_target = DataLoader(dataset=combined_dataset_target, batch_size=30, shuffle=True)

train_dataset_source = TensorDataset(x_train_source, y_train_source)
train_loader_source = DataLoader(dataset=train_dataset_source, batch_size=30, shuffle=True)
test_dataset_source = TensorDataset(x_test_source, y_test_source)
test_loader_source = DataLoader(dataset=test_dataset_source, batch_size=30, shuffle=True)
combined_dataset_source = torch.utils.data.ConcatDataset([train_dataset_source, test_dataset_source])
combined_loader_source = DataLoader(dataset=combined_dataset_source, batch_size=30, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(128 * 125 * 1, 256)
        self.Dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.Dropout(x)
        out = self.fc2(x)
        return out

    def extract_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x


def test_time_adaptation(net, test_loader_target, test_loader_source):
    net.train()
    correct = 0
    test_loss = 0
    total = 0
    all_outputs_target = []
    all_features_target = []
    all_features_source = []

    # 提取目标域特征
    for i, data in enumerate(test_loader_target):
        if (i % 50 == 0):
            print(i)
        x_test, y_test = data
        x_test = x_test.float()
        x_test, y_test = x_test.to(device), y_test.to(device)
        total += y_test.size(0)
        outputs = net(x_test)
        if hasattr(net, 'model'):
            features = net.model.extract_features(x_test)
        else:
            features = net.extract_features(x_test)
        all_features_target.append(features.cpu().detach())
        all_outputs_target.append(outputs.cpu().detach())
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_test.long()).sum().item()
    test_acc = correct / total
    print("test_acc{}".format(test_acc))
    all_outputs_target = torch.cat(all_outputs_target, dim=0)
    all_features_target = torch.cat(all_features_target, dim=0)

    # 提取源域特征（只提取一次）
    if not all_features_source:
        for i, data in enumerate(test_loader_source):
            x_test, y_test = data
            x_test = x_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            if hasattr(net, 'model'):
                features = net.model.extract_features(x_test)
            else:
                features = net.extract_features(x_test)
            all_features_source.append(features.cpu().detach())
        all_features_source = torch.cat(all_features_source, dim=0)

    return all_outputs_target, all_features_target, all_features_source


def setup_tent(model):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = optim.Adam(params,
                           lr=1e-5,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    tent_model = tent.Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def plot_feature_distribution(features_target, features_source, title):
    # 使用 PCA 将特征降维到 2 维
    pca = PCA(n_components=2)
    features_target_2d = pca.fit_transform(features_target.numpy())
    features_source_2d = pca.transform(features_source.numpy())

    plt.scatter(features_target_2d[:, 0], features_target_2d[:, 1], s=1, c='blue', label='目标域数据集')
    plt.scatter(features_source_2d[:, 0], features_source_2d[:, 1], s=1, c='red', label='源域数据集')
    plt.title(title)
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    base_model = CNN()
    model_path = 'D:/model/5300-3.pth'
    base_model.load_state_dict(torch.load(model_path, weights_only=True))
    base_model.to(device)

    # 提取调整前的特征
    _, pre_adapt_features_target, source_features = test_time_adaptation(base_model, combined_loader_target,
                                                                         combined_loader_source)
    plot_feature_distribution(pre_adapt_features_target, source_features,
                              '自适应前特征分布')

    tent_model = setup_tent(base_model)
    adjusted_data, post_adapt_features_target, _ = test_time_adaptation(tent_model, combined_loader_target,
                                                                        combined_loader_source)
    print("调整后的数据形状:", adjusted_data.shape)

    # 提取调整后的特征
    plot_feature_distribution(post_adapt_features_target, source_features,
                              '自适应前特征分布')