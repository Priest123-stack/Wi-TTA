import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from ptflops import get_model_complexity_info

# 数据加载和预处理（保持与CNN相同的数据源）
x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')

# 展平数据为MLP需要的形状 (samples, features)
x_train = torch.tensor(x_train.reshape(len(x_train), -1))  # [N, 2000*30]
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test.reshape(len(x_test), -1))  # [N, 2000*30]
y_test = torch.tensor(y_test)

print(x_train.shape)  # 应显示 torch.Size([N, 60000])
print(x_test.shape)  # 应显示 torch.Size([M, 60000])

# 创建数据加载器
train_dataset = TensorDataset(x_train.float(), y_train)
test_dataset = TensorDataset(x_test.float(), y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class MLP(nn.Module):
    def __init__(self, input_dim=2000 * 30, num_classes=7):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, epochs=50, lr=0.0001):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} Acc: {acc:.2f}%')


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型
    mlp = MLP().to(device)

    # 计算模型复杂度
    macs, params = get_model_complexity_info(mlp, (2000 * 30,), as_strings=False)
    print(f"FLOPs: {macs * 2 / 1e6:.2f}M")
    print(f"Parameters: {params / 1e6:.2f}M")

    # # 训练和测试
    train(mlp, train_loader, epochs=50)
    test_acc = test(mlp, test_loader)
