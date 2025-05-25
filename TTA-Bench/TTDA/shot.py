# import torch
# import torch.nn.functional as F
# import torch.nn as nn
#
# class SHOTWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.model.eval()
#
#         # Freeze classifier
#         for param in self.model.fc2.parameters():
#             param.requires_grad = False
#
#         self.params = [p for n, p in self.model.named_parameters() if p.requires_grad]
#         self.optimizer = torch.optim.Adam(self.params, lr=1e-5)
#
#     def forward(self, x):
#         return self.update(x)
#
#     def update(self, x):
#         x = x.to(next(self.model.parameters()).device)
#         x.requires_grad = True
#         feat = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
#         feat = feat.view(x.size(0), -1)
#         logits = self.model.fc2(F.dropout(F.relu(self.model.fc1(feat)), p=0.5, training=True))
#         softmax_out = F.softmax(logits, dim=1)
#
#         # entropy minimization
#         entropy = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-6), dim=1))
#
#         # diversity maximization
#         mean_softmax = torch.mean(softmax_out, dim=0)
#         diversity = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-6))
#
#         # cosine-based pseudo-labeling
#         pseudo_preds = softmax_out.argmax(1)
#         num_classes = softmax_out.shape[1]
#         centroids = []
#         for c in range(num_classes):
#             feats_c = feat[pseudo_preds == c]
#             if feats_c.shape[0] > 0:
#                 centroids.append(F.normalize(feats_c.mean(0, keepdim=True), dim=1).squeeze())
#             else:
#                 centroids.append(torch.zeros(feat.size(1), device=feat.device))
#         centroids = torch.stack(centroids)
#
#         feat_norm = F.normalize(feat, dim=1)
#         sim = torch.mm(feat_norm, centroids.t())
#         pseudo_labels = sim.argmax(dim=1)
#
#         loss_ce = F.cross_entropy(logits, pseudo_labels)
#
#         loss = entropy + diversity + 0.3 * loss_ce
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         return logits
#
#     # 重命名自定义的 train 方法
#     def train_model(self, data_loader):
#         for i, data in enumerate(data_loader):
#             if (i % 50 == 0):
#                 print(i)
#             x, _ = data
#             self.update(x)
#         return self.model, self.model.fc2

import torch
import torch.nn.functional as F
import torch.nn as nn

class SHOTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

        # Freeze classifier
        for param in self.model.fc2.parameters():
            param.requires_grad = False

        # 确保需要更新的参数的 requires_grad 为 True
        self.params = [p for n, p in self.model.named_parameters() if p.requires_grad and 'fc2' not in n]
        self.optimizer = torch.optim.Adam(self.params, lr=1e-5)

    def forward(self, x):
        return self.update(x)

    def update(self, x):
        x = x.to(next(self.model.parameters()).device)
        x.requires_grad = True
        feat = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        feat = feat.view(x.size(0), -1)
        logits = self.model.fc2(F.dropout(F.relu(self.model.fc1(feat)), p=0.5, training=True))
        softmax_out = F.softmax(logits, dim=1)

        # entropy minimization
        entropy = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-6), dim=1))

        # diversity maximization
        mean_softmax = torch.mean(softmax_out, dim=0)
        diversity = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-6))

        # cosine-based pseudo-labeling
        pseudo_preds = softmax_out.argmax(1)
        num_classes = softmax_out.shape[1]
        centroids = []
        for c in range(num_classes):
            feats_c = feat[pseudo_preds == c]
            if feats_c.shape[0] > 0:
                centroids.append(F.normalize(feats_c.mean(0, keepdim=True), dim=1).squeeze())
            else:
                centroids.append(torch.zeros(feat.size(1), device=feat.device))
        centroids = torch.stack(centroids)

        feat_norm = F.normalize(feat, dim=1)
        sim = torch.mm(feat_norm, centroids.t())
        pseudo_labels = sim.argmax(dim=1)

        loss_ce = F.cross_entropy(logits, pseudo_labels)

        loss = entropy + diversity + 0.3 * loss_ce

        self.optimizer.zero_grad()
        loss.backward()
        # 验证梯度是否为 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: {param.grad.sum()}")

        self.optimizer.step()

        return logits

    # 重命名自定义的 train 方法
    def train_model(self, data_loader):
        for i, data in enumerate(data_loader):
            if (i % 50 == 0):
                print(i)
            x, _ = data
            self.update(x)
        return self.model, self.model.fc2