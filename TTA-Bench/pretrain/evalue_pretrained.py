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



def evaluate_pretrained():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载测试数据 ===
    x_test = np.load('E:/wifi感知/5300-3_npy/x_test.npy')
    y_test = np.load('E:/wifi感知/5300-3_npy/y_test.npy')
    test_dataset = TensorDataset(
        torch.tensor(x_test.reshape(-1, 1, 2000, 30)).float(),
        torch.tensor(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === 加载预训练模型 ===
    checkpoint = torch.load("D:/model/CNN_GPU_3.pth")
    model = GeOS_PreTrainer(num_classes=checkpoint['num_classes']).to(device)

    # === 正确加载方式：分开加载primary_net和classifier ===
    model.primary_net.load_state_dict(checkpoint['primary_net'])  # 直接加载整个primary_net
    model.classifier.load_state_dict(checkpoint['classifier'])  # 直接加载整个classifier

    # === 评估 ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"原始预训练模型测试准确度: {correct / total:.4f}")


if __name__ == '__main__':
    evaluate_pretrained()