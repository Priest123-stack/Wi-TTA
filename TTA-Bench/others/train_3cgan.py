# train.py
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据加载（根据实际路径修改）
def load_data(data_path='E:/wifi感知/5300-3_npy/'):
    x_train = np.load(data_path + 'x_train.npy')
    y_train = np.load(data_path + 'y_train.npy')

    # 转换为PyTorch Tensor
    x_train = torch.tensor(x_train.reshape(-1, 1, 2000, 30)).float()
    y_train = torch.tensor(y_train).long()

    print(f"Training data shape: {x_train.shape}")
    return TensorDataset(x_train, y_train)


# 模型定义（与CNN.py完全一致）
class SourceCNN(nn.Module):
    def __init__(self):
        super(SourceCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (16, 1000, 15)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (32, 500, 7)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (64, 250, 3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (128, 125, 1)
        )
        self.fc1 = nn.Linear(128 * 125 * 1, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def extract_features(self, x):
        """用于3C-GAN的特征提取方法"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

    def classify(self, features):
        """用于3C-GAN的分类头"""
        x = F.relu(self.fc1(features))
        return self.fc2(x)


# 训练函数
def pretrain_model(model, train_loader, epochs=30, save_path="source_model.pth"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print("\nStart training...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 打印训练信息
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    # 1. 加载数据
    train_dataset = load_data()  # 修改为实际数据路径

    # 2. 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2
    )

    # 3. 初始化模型
    model = SourceCNN().to(device)
    print("Model architecture:\n", model)

    # 4. 开始预训练
    pretrain_model(
        model=model,
        train_loader=train_loader,
        epochs=30,
        save_path="D:/model/3cgan_3.pth"
    )