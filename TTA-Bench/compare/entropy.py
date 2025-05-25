import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
id = '5300-3'
x_train = np.load(f'E:/wifi感知/{id}_npy/x_train.npy')
y_train = np.load(f'E:/wifi感知/{id}_npy/y_train.npy')
x_test = np.load(f'E:/wifi感知/{id}_npy/x_test.npy')
y_test = np.load(f'E:/wifi感知/{id}_npy/y_test.npy')

# 转换为PyTorch Dataset
x_combined = np.concatenate([x_train, x_test], axis=0)
y_combined = np.concatenate([y_train, y_test], axis=0)
combined_data = torch.tensor(x_combined.reshape(-1, 1, 2000, 30)).float().to(device)
combined_labels = torch.tensor(y_combined).to(device)
combined_dataset = TensorDataset(combined_data, combined_labels)
combined_loader = DataLoader(combined_dataset, batch_size=30, shuffle=False)


# 定义CNN模型
class WiFiCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((100, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 100, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def entropy_loss(predictions):
    probs = torch.softmax(predictions, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy.mean()


def diversity_loss(predictions):
    probs = torch.softmax(predictions, dim=1)
    mean_probs = torch.mean(probs, dim=0)
    uniform = torch.ones_like(mean_probs) / probs.size(1)
    div_loss = torch.sum(mean_probs * torch.log(mean_probs / uniform))
    return div_loss


def train_model(model, loader, loss_fn, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x = x.to(device)
            preds = model(x)
            loss = loss_fn(preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def get_probs(model, loader):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x)
            probs = torch.softmax(preds, dim=1).cpu()  # 将结果移回CPU
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0).detach().numpy()


# 原始模型预测（无损失优化）
model_raw = WiFiCNN().to(device)
probs_raw = get_probs(model_raw, combined_loader)

# 仅熵最小化
model_ent = WiFiCNN().to(device)
model_ent = train_model(model_ent, combined_loader, entropy_loss)
probs_ent = get_probs(model_ent, combined_loader)

# 仅多样性最大化
model_div = WiFiCNN().to(device)
model_div = train_model(model_div, combined_loader, diversity_loss)
probs_div = get_probs(model_div, combined_loader)

# 混合损失
model_hybrid = WiFiCNN().to(device)
hybrid_loss = lambda pred: 0.6 *entropy_loss(pred) +  diversity_loss(pred)
model_hybrid = train_model(model_hybrid, combined_loader, hybrid_loss)
probs_hybrid = get_probs(model_hybrid, combined_loader)


def plot_tsne(probs, labels, title):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(probs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.6)
    # 新增字号设置
    plt.xlabel("t-SNE 1", fontsize=16)  # 横轴标签字号16
    plt.ylabel("t-SNE 2", fontsize=16)  # 纵轴标签字号16
    plt.tick_params(axis='both', labelsize=14)  # 刻度字号14

    # 可选：调整图例和标题字号
    plt.legend(*scatter.legend_elements(), title="Classes", fontsize=12, title_fontsize=14)
    plt.title(title, fontsize=18)  # 标题字号18
    # plt.legend(*scatter.legend_elements(), title="Classes")
    # plt.title(title)
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    plt.show()


# 原始数据分布
plot_tsne(probs_raw, combined_labels, "Original Predictions (No Loss)")

# 仅熵最小化
plot_tsne(probs_ent, combined_labels, "Entropy Minimization (L_ent)")

# 仅多样性最大化
plot_tsne(probs_div, combined_labels, "Diversity Maximization (L_div)")

# 混合损失
plot_tsne(probs_hybrid, combined_labels, "Combined Loss (L_ent + L_div)")