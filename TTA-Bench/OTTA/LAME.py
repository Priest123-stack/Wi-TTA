import torch
import torch.nn as nn
import torch.nn.functional as F


class LAMEWrapper(nn.Module):
    def __init__(self, model, alpha=0.5):
        super().__init__()
        self.model = model
        self.model.eval()

        self.alpha = alpha  # 拉普拉斯正则化权重

    def forward(self, x):
        """普通推理"""
        return self.model(x)

    def compute_laplacian_loss(self, features, logits):
        """
        features: 特征向量 [batch, feat_dim]
        logits: 模型输出 [batch, num_classes]
        """
        # 计算特征相似度矩阵（cosine similarity）
        features = F.normalize(features, dim=1)
        sim_matrix = torch.mm(features, features.t())  # [B, B]

        # 构建邻接矩阵 A
        A = F.relu(sim_matrix)  # 只保留正相似度作为邻接关系

        # 计算度矩阵 D
        D = torch.diag(A.sum(1))

        # Laplacian矩阵 L = D - A
        L = D - A

        # softmax输出
        prob = F.softmax(logits, dim=1)

        # 计算拉普拉斯正则项：tr(P^T L P)
        laplacian_loss = torch.trace(torch.mm(prob.t(), torch.mm(L, prob))) / prob.shape[0]

        return laplacian_loss

    def update(self, x):
        """
        Test-Time Adaptation入口，参数-free
        """
        x = x.to(next(self.model.parameters()).device)

        # 提取特征 + logits
        feat = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        feat = feat.view(x.size(0), -1)  # 展平
        logits = self.model.fc2(F.relu(self.model.fc1(feat)))

        # laplacian loss
        laplacian_loss = self.compute_laplacian_loss(feat, logits)

        # 原本的 logits softmax输出
        prob = F.softmax(logits, dim=1)

        # 直接用 laplacian loss 反向调整 logits，不更新模型参数
        logits_adjusted = logits - self.alpha * laplacian_loss

        return logits_adjusted
