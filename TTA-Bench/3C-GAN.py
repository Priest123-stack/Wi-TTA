# main.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from pretrain.train_3cgan import SourceCNN  # 从train.py导入模型定义

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数配置
EPOCHS = 1
BATCH_SIZE = 10
FEAT_DIM = 128 * 125 * 1  # 特征维度必须与模型输出一致
NUM_CLASSES = 7
LAMBDA_S = 3  # 语义相似性权重
LAMBDA_W = 0  # 权重正则化权重
LAMBDA_CLU = 0  # 聚类正则化权重


# 数据加载函数（修改为你的测试数据路径）
def load_target_data(data_path='E:/wifi感知/5300-3_npy/'):
    # 加载训练集数据
    x_train = np.load(data_path + 'x_train.npy')
    y_train = np.load(data_path + 'y_train.npy')

    # 加载测试集数据
    x_test = np.load(data_path + 'x_test.npy')
    y_test = np.load(data_path + 'y_test.npy')

    # 合并训练集和测试集数据
    x_combined = np.concatenate((x_train, x_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)

    # 转换为PyTorch Tensor
    x_combined = torch.tensor(x_combined.reshape(-1, 1, 2000, 30)).float()
    y_combined = torch.tensor(y_combined).long()

    print(f"Combined target data shape: {x_combined.shape}")
    return TensorDataset(x_combined, y_combined)


# 3C-GAN组件定义
class ConditionalGenerator(nn.Module):
    """带类别条件的生成器"""

    def __init__(self, input_dim=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        return self.net(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    """域判别器（带谱归一化）"""

    def __init__(self, input_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 3C-GAN训练函数
def adapt_3cgan(source_model, target_loader):
    # 初始化组件
    G = ConditionalGenerator().to(device)
    D = Discriminator().to(device)

    # 关键修复：获取分类头的参数（fc1和fc2）
    classifier_params = [
        {"params": source_model.fc1.parameters()},
        {"params": source_model.fc2.parameters()}
    ]

    # 优化器
    opt_g = optim.Adam(G.parameters(), lr=1e-5, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=1e-5, betas=(0.5, 0.999))
    opt_c = optim.Adam(classifier_params, lr=1e-5)  # 使用修正后的参数

    # 冻结特征提取器（除fc1和fc2外的所有层）
    for name, param in source_model.named_parameters():
        if "fc1" not in name and "fc2" not in name:
            param.requires_grad = False

    # 保存原始模型参数（用于权重正则化）
    source_params = {n: p.detach().clone() for n, p in source_model.named_parameters()}

    # 训练循环
    for epoch in range(EPOCHS):
        for batch_idx, (x_real, _) in enumerate(target_loader):
            x_real = x_real.to(device)
            batch_size = x_real.size(0)

            # ==================== 训练判别器 ====================
            with torch.no_grad():
                real_feat = source_model.extract_features(x_real)
                labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
                z = torch.randn(batch_size, FEAT_DIM).to(device)
                fake_feat = G(z, labels)

            # 计算判别器损失
            real_loss = F.binary_cross_entropy(D(real_feat), torch.ones(batch_size, 1).to(device))
            fake_loss = F.binary_cross_entropy(D(fake_feat.detach()), torch.zeros(batch_size, 1).to(device))
            d_loss = (real_loss + fake_loss) / 2

            # 更新判别器
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ==================== 训练生成器和分类器 ====================
            # 生成新特征
            z = torch.randn(batch_size, FEAT_DIM).to(device)
            labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
            fake_feat = G(z, labels)

            # 对抗损失
            g_loss_adv = F.binary_cross_entropy(D(fake_feat), torch.ones(batch_size, 1).to(device))

            # 语义相似性损失
            logits = source_model.classify(fake_feat)
            sem_loss = F.cross_entropy(logits, labels)

            # 总生成器损失
            g_loss = g_loss_adv + LAMBDA_S * sem_loss

            # 分类器损失
            with torch.no_grad():
                real_feat = source_model.extract_features(x_real)
                pseudo_labels = torch.argmax(source_model.classify(real_feat), dim=1)

            trans_logits = source_model.classify(fake_feat)
            c_loss = F.cross_entropy(trans_logits, pseudo_labels)

            # 权重正则化
            w_reg = 0.0
            for n, p in source_model.named_parameters():
                if 'classify' in n:
                    w_reg += torch.norm(p - source_params[n])

            # 聚类正则化（条件熵）
            real_logits = source_model.classify(real_feat)
            entropy_loss = -torch.mean(torch.sum(F.softmax(real_logits, dim=1) * F.log_softmax(real_logits, dim=1)))

            # 总分类器损失
            total_c_loss = LAMBDA_W * w_reg + LAMBDA_CLU * entropy_loss + c_loss

            # 更新生成器和分类器
            opt_g.zero_grad()
            opt_c.zero_grad()
            (g_loss + total_c_loss).backward()
            opt_g.step()
            opt_c.step()

            # 打印训练信息
            # print('epoch:',epoch)
            # if (epoch + 1) % 10 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} C_loss: {total_c_loss.item():.4f}")

    return source_model


# 评估函数
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    # 1. 加载目标域数据
    target_dataset = load_target_data()
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 2. 加载预训练源模型
    source_model = SourceCNN().to(device)
    source_model.load_state_dict(torch.load('D:/model/3cgan_3.pth', map_location=device))
    print("Source model loaded successfully")

    # 3. 执行3C-GAN自适应
    print("\nStarting 3C-GAN adaptation...")
    adapted_model = adapt_3cgan(source_model, target_loader)

    # 4. 评估目标域性能
    test_acc = evaluate(adapted_model, target_loader)
    print("test_acc: {:.4f}".format(test_acc))

    # # 5. 保存适配后的模型（可选）
    # torch.save(adapted_model.state_dict(), 'D:/model/3cgan_adapted.pth')
    # print("Adapted model saved")