import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
class CNN_geos(nn.Module):
    def __init__(self):
        super(CNN_geos, self).__init__()
        # 特征提取器（卷积层）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 类似定义 layer2, layer3, layer4...
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 辅助模块（全连接层）
        self.aux_block = nn.Sequential(
            nn.Linear(128*125*1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.aux_block(x)
        return x
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = CNN_geos().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据加载（保持与原始代码一致）
    x_train = np.load('E:/wifi感知/5300-2_npy/x_train.npy')
    y_train = np.load('E:/wifi感知/5300-2_npy/y_train.npy')
    x_train = torch.tensor(x_train.reshape(len(x_train), 1, 2000, 30)).float()
    y_train = torch.tensor(y_train).long()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f} Acc: {acc:.4f}")

    # 保存模型参数
    torch.save(model.state_dict(), "D:/model/CNN_geos_2.pth")


if __name__ == '__main__':
    train_model()