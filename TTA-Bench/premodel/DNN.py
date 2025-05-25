import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from thop import profile
import time

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
# 修正加载标签数据
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
# 修正加载标签数据
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')

x_train = torch.Tensor(x_train)
x_train = x_train.reshape(x_train.size(0), -1)
y_train = torch.Tensor(y_train).long()

x_test = torch.Tensor(x_test)
x_test = x_test.reshape(x_test.size(0), -1)
y_test = torch.Tensor(y_test).long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

class DNN(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(n_feature, n_hidden1)
        self.batch1 = nn.BatchNorm1d(n_hidden1, momentum=0.9)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.batch2 = nn.BatchNorm1d(n_hidden2, momentum=0.9)
        self.layer3 = nn.Linear(n_hidden2, n_hidden3)
        self.batch3 = nn.BatchNorm1d(n_hidden3, momentum=0.9)
        self.layer4 = nn.Linear(n_hidden3, n_hidden4)
        self.batch4 = nn.BatchNorm1d(n_hidden4, momentum=0.9)
        self.drop = nn.Dropout(0.5)
        self.layer5 = nn.Linear(n_hidden4, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(self.batch1(x))
        x = self.layer2(x)
        x = F.relu(self.batch2(x))
        x = self.layer3(x)
        x = F.relu(self.batch3(x))
        x = self.layer4(x)
        x = F.relu(self.batch4(x))
        x = self.drop(x)
        x = self.layer5(x)
        return x

def training(module):
    module.float()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(module.parameters(), lr=0.0001)
    epoches = 100
    # 收敛判断参数
    convergence_threshold = 0.01  # 损失变化阈值
    consecutive_epochs = 5  # 连续 epoch 数
    loss_history = []

    start_time = time.time()  # 记录训练开始时间
    for epoch in range(epoches):
        train_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            x_train, y_train = data
            x_train = x_train.float()
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            output = module(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += y_train.size(0)
            correct += (predicted == y_train).sum().item()
            train_loss += loss.item()

        train_acc = correct / total
        avetrain_loss = train_loss / total
        print("epoch:{}  train_loss:{}  train_acc:{} ".format(epoch, avetrain_loss, train_acc))

        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(test_loader):
                module.eval()
                x_test, y_test = data
                x_test = x_test.float()
                x_test, y_test = x_test.to(device), y_test.to(device)
                outputs = module(x_test)
                _, predicted = torch.max(outputs.data, 1)
                total += y_test.size(0)
                correct += (predicted == y_test).sum().item()

        print('Accuracy of the network on the test :', correct / total)

        loss_history.append(avetrain_loss)
        if len(loss_history) >= consecutive_epochs:
            losses = loss_history[-consecutive_epochs:]
            max_diff = max(losses) - min(losses)
            if max_diff < convergence_threshold:
                print(f"Model converged at epoch {epoch}.")
                break

    end_time = time.time()  # 记录训练结束时间
    convergence_time = end_time - start_time  # 计算收敛时间
    print(f"Model convergence time: {convergence_time} seconds")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    net = DNN(60000, 600, 300, 100, 50, 7)
    net = net.to(device)

    # 计算 FLOPs 和参数量
    input_tensor = torch.randn(1, 60000).to(device)
    flops, params = profile(net, inputs=(input_tensor,))
    flops_m = flops / 1e6  # 转换为 M FLOPs
    params_m = params / 1e6  # 转换为 M Params
    print(f"FLOPs (M): {flops_m}")
    print(f"Params (M): {params_m}")

    training(net)