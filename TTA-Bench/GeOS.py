import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GeOS_Adapt(nn.Module):
    def __init__(self, pretrained_path, num_aux_classes=3):
        super().__init__()
        checkpoint = torch.load(pretrained_path)

        # === 关键修改：正确推断分类器维度 ===
        # 获取分类器所有键名
        classifier_keys = list(checkpoint['classifier'].keys())
        # 过滤出权重参数键（假设最后一层是Linear）
        weight_keys = [k for k in classifier_keys if 'weight' in k]
        if not weight_keys:
            raise ValueError("Classifier has no weight parameters")
        last_weight_key = weight_keys[-1]
        num_classes = checkpoint['classifier'][last_weight_key].shape[0]

        # 主网络结构（需与预训练时一致）
        self.primary_net = nn.Sequential(
            nn.Conv2d(1, 64, (7, 3), (2, 1), (3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 1), (1, 1)),
            self._make_residual_layer(64, 2)
        )
        self.primary_net.load_state_dict(checkpoint['primary_net'])
        for param in self.primary_net.parameters():
            param.requires_grad = False

        # 主分类器结构（严格匹配预训练结构）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)  # 使用推断的类别数
        )
        self.classifier.load_state_dict(checkpoint['classifier'])
        for param in self.classifier.parameters():
            param.requires_grad = False

        # 可训练的辅助网络
        self.aux_net = nn.Sequential(
            self._make_residual_layer(64, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_aux_classes)
        )

    def _make_residual_layer(self, channels, num_blocks):
        return nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x, aug_x=None):
        main_feat = self.primary_net(x)
        aux_out = None

        if aug_x is not None:
            with torch.no_grad():
                aug_feat = self.primary_net(aug_x)
            aux_out = self.aux_net(aug_feat)

        return self.classifier(main_feat), aux_out


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


def apply_augmentation(x, method):
    """数据增强实现"""
    if method == 'noise':
        return x + torch.randn_like(x) * 0.1
    elif method == 'shift':
        return torch.roll(x, shifts=np.random.randint(-100, 100), dims=2)
    elif method == 'scale':
        return x * torch.FloatTensor(1).uniform_(0.8, 1.2).to(x.device)
    return x


def load_data():
    """加载并合并数据集（保持标签但训练时不使用）"""
    x_train = np.load('E:/wifi感知/5300-2_npy/x_train.npy')
    y_train = np.load('E:/wifi感知/5300-2_npy/y_train.npy')
    x_test = np.load('E:/wifi感知/5300-2_npy/x_test.npy')
    y_test = np.load('E:/wifi感知/5300-2_npy/y_test.npy')

    # 转换为Tensor并保持标签
    train_dataset = TensorDataset(
        torch.tensor(x_train.reshape(-1, 1, 2000, 30)).float(),
        torch.tensor(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test.reshape(-1, 1, 2000, 30)).float(),
        torch.tensor(y_test).long()
    )
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    return (
        DataLoader(train_dataset, batch_size=16, shuffle=True),
        DataLoader(test_dataset, batch_size=16, shuffle=False),
        DataLoader(combined_dataset, batch_size=16, shuffle=True)
    )


def adapt_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, combined_loader = load_data()
    # 初始化模型
    model = GeOS_Adapt("D:/model/CNN_GPU_3.pth").to(device)
    optimizer = optim.Adam(model.aux_net.parameters(), lr=1e-4)

    # 无监督自适应训练
    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for batch in combined_loader:
            x = batch[0].to(device)  # 只取特征，忽略标签

            # 生成自监督标签
            aux_labels = torch.randint(0, 3, (x.size(0),)).to(device)

            # 应用数据增强
            aug_x = torch.stack([
                apply_augmentation(x[i], ['noise', 'shift', 'scale'][label.item()])
                for i, label in enumerate(aux_labels)
            ])

            # 前向传播
            _, aux_out = model(x, aug_x)

            # 计算损失
            loss = F.cross_entropy(aux_out, aux_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        # 输出epoch信息
        print(f"Adapt Epoch [{epoch + 1}] Loss: {total_loss / len(combined_loader.dataset):.4f}")

    # 最终评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs, _ = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    # print(f"Final Test Accuracy: {correct / total:.4f}")
    test_acc = correct / total
    print("test_acc{}".format(test_acc))
    # 可视化部分（修改版）
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_data(sample, title, ax):
        """绘制数据热力图（自动适应数据范围）"""
        ax.imshow(sample, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # 设置随机种子确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 获取样本数据
    def get_sample(loader):
        batch, _ = next(iter(loader))
        return batch[0].unsqueeze(0).to(device)  # [1,1,2000,30]

    # 获取源域和目标域样本
    source_sample_tensor = get_sample(train_loader)
    target_sample_tensor = get_sample(test_loader)

    # 应用链式增强（噪声→位移→缩放）
    def apply_combined_augmentation(x):
        x = apply_augmentation(x, 'noise')  # 1. 添加噪声
        x = apply_augmentation(x, 'shift')  # 2. 随机位移
        x = apply_augmentation(x, 'scale')  # 3. 随机缩放
        return x

    # 生成增强数据并归一化
    with torch.no_grad():
        # 应用三次增强
        aug_tensor = apply_combined_augmentation(target_sample_tensor)

        # 执行归一化 (x - μ)/σ
        # aug_normalized = (aug_tensor - aug_tensor.mean()) / aug_tensor.std()

    # 转换为numpy数组
    source_sample = source_sample_tensor.squeeze().cpu().numpy()
    target_sample = target_sample_tensor.squeeze().cpu().numpy()
    aug_sample = aug_tensor.squeeze().cpu().numpy()

    # 创建对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

    # 绘制对比数据
    plot_data(source_sample, "Source Domain (Raw)", axes[0])
    plot_data(target_sample, "Target Domain (Original)", axes[1])
    plot_data(aug_sample, "Target Domain (Combined Augmented + Normalized)", axes[2])

    plt.tight_layout()
    plt.savefig('GeOS_combined_aug.png', bbox_inches='tight')
    plt.show()

    # 打印数据统计信息
    print("[数据统计]")
    print(f"源域数据   | 均值: {source_sample.mean():.2f} 标准差: {source_sample.std():.2f}")
    print(f"目标域原始 | 均值: {target_sample.mean():.2f} 标准差: {target_sample.std():.2f}")
    print(f"增强后数据 | 均值: {aug_sample.mean():.2f} 标准差: {aug_sample.std():.2f}")

if __name__ == '__main__':
    adapt_and_evaluate()
