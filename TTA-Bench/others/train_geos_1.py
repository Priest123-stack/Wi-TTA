import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class GeOS_PreTrainer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 主特征提取器（ResNet风格）
        self.primary_net = nn.Sequential(
            nn.Conv2d(1, 64, (7, 3), (2, 1), (3, 1)),  # [64, 1000, 30]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 1), (1, 1)),  # [64, 500, 30]
            self._make_residual_layer(64, 2)
        )

        # 主分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def _make_residual_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.primary_net(x)
        return self.classifier(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv_block(x))


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    x_train = np.load('E:/wifi感知/5300-3_npy/x_train.npy')
    y_train = np.load('E:/wifi感知/5300-3_npy/y_train.npy')
    x_train = torch.tensor(x_train.reshape(-1, 1, 2000, 30)).float()
    y_train = torch.tensor(y_train).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=32, shuffle=True,
                              pin_memory=True if device == 'cuda' else False)

    # 模型初始化
    model = GeOS_PreTrainer(num_classes=len(torch.unique(y_train))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(100):
        model.train()
        total_loss, correct = 0.0, 0

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
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch [{epoch + 1}/50] Loss: {total_loss / len(x_train):.4f} "
              f"Acc: {correct / len(x_train):.4f}")

    # 保存主干网络参数（为后续GeOS扩展保留接口）
    torch.save({
        'primary_net': model.primary_net.state_dict(),
        'classifier': model.classifier.state_dict(),
        'num_classes': len(torch.unique(y_train))  # 新增元数据
    }, "D:/model/CNN_GPU_3.pth")


if __name__ == '__main__':
    train_model()