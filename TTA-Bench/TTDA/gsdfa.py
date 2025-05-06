# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # class GSDFAWrapper(nn.Module):
# #     def __init__(self, model, mask_old=None):
# #         super(GSDFAWrapper, self).__init__()
# #         self.model = model
# #         self.device = next(self.model.parameters()).device
# #         self.mask_old = mask_old  # 历史mask
# #         self.mask = None  # 当前mask
# #         self.hook_registered = False
# #
# #     def register_feature_hook(self):
# #         def feature_hook(module, input, output):
# #             if self.mask_old is not None:
# #                 if self.mask_old.shape[0] != output.shape[1]:
# #                     # print(
# #                     #     f"[警告] Mask channels ({self.mask_old.shape[0]}) != Output channels ({output.shape[1]}), 自动适配")
# #                     if self.mask_old.shape[0] > output.shape[1]:
# #                         adjusted_mask = self.mask_old[:output.shape[1]]
# #                     else:
# #                         adjusted_mask = F.pad(self.mask_old, (0, output.shape[1] - self.mask_old.shape[0]), "constant",
# #                                               1.0)
# #                 else:
# #                     adjusted_mask = self.mask_old
# #                 mask = adjusted_mask.view(1, -1, 1, 1)
# #                 return output * mask
# #             else:
# #                 return output
# #         # 自动挂到 model.feature_layer，如果没有需要用户在模型里补充
# #         if hasattr(self.model, 'feature_layer'):
# #             self.model.feature_layer.register_forward_hook(feature_hook)
# #         else:
# #             raise AttributeError("Model has no attribute 'feature_layer'. Please set 'feature_layer' manually.")
# #
# #     def extract_features(self, x):
# #         if not self.hook_registered:
# #             self.register_feature_hook()
# #             self.hook_registered = True
# #         return self.model.extract_features(x)
# #
# #     def forward(self, x):
# #         feat = self.extract_features(x)
# #         feat = feat.view(feat.size(0), -1)  # <-- 🔥重要，展平！！
# #         feat = F.relu(self.model.fc1(feat))
# #         logits = self.model.fc2(feat)
# #         return logits
# #
# #     def update(self, dataloader, epochs=5, k=5, lr=1e-4):
# #         self.model.train()
# #         optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
# #         for epoch in range(epochs):
# #             for batch_idx, (x, _) in enumerate(dataloader):
# #                 x = x.to(self.device)
# #                 feat = self.extract_features(x)
# #                 logits = self.model.classify(feat)
# #                 entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1).mean()
# #                 loss = entropy
# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 optimizer.step()
# #
# # def setup_gsdfa(model, mask_old=None):
# #     return GSDFAWrapper(model, mask_old)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
#
# class SDFAAdapt:
#     def __init__(self, model, classifier, mask_old=None, reg_lambda=0.001, max_iter=100, k=3):
#         self.model = model  # 源模型（特征提取器）
#         self.classifier = classifier  # 分类器（fc1 + fc2）
#         self.mask_old = mask_old  # 初始掩码（可选）
#         self.reg_lambda = reg_lambda  # 正则化权重
#         self.max_iter = max_iter  # 最大迭代次数
#         self.k = k  # 邻域数量
#         # 冻结特征提取器的参数
#         for param in self.model.parameters():
#             param.requires_grad = False
#         # 确保分类器参数可训练
#         for param in self.classifier.parameters():
#             param.requires_grad = True  # 添加此行
#
#         # 冻结特征提取器的参数
#         for param in self.model.parameters():
#             param.requires_grad = False
#
#         # 仅优化分类器参数
#         self.optimizer = optim.SGD(self.classifier.parameters(), lr=1e-6)
#         self.device = next(self.model.parameters()).device
#
#         # 动态计算 fc1 输入尺寸
#         self._to_linear = None
#         self._calculate_flattened_size()
#
#     def _calculate_flattened_size(self):
#         with torch.no_grad():
#             x = torch.randn(1, 1, 2000, 30).to(self.device)
#             features = self.model.extract_features(x)
#             self._to_linear = features.numel()
#
#     def compute_maximization_loss(self, features):
#         feat_mean = features.mean(dim=0)
#         feat_diff = features - feat_mean
#         feat_loss = feat_diff.pow(2).sum(dim=1).mean()
#         return feat_loss
#
#     def generate_dynamic_mask(self, features):
#         # 动态生成掩码（示例：基于特征均值生成）
#         embedding = features.mean(dim=[2, 3])  # 全局平均池化
#         mask = torch.sigmoid(100 * embedding)  # 生成接近二值的掩码
#         return mask
#
#     def apply_gradient_mask(self):
#         if self.mask is None:
#             return
#         # 应用动态生成的掩码到分类器梯度
#         for name, param in self.classifier.named_parameters():
#             if 'weight' in name and param.grad is not None:
#                 param.grad *= (1.0 - self.mask.view(-1, 1).to(self.device))
#             elif 'bias' in name and param.grad is not None:
#                 param.grad *= (1.0 - self.mask.to(self.device))
#
#     def train(self, dset_loader_target):
#         self.model.eval()  # 特征提取器始终在评估模式
#         self.classifier.train()  # 分类器可训练
#
#         # 初始化特征库和得分库
#         num_sample = len(dset_loader_target.dataset)
#         fea_bank = torch.randn(num_sample, self._to_linear).to(self.device)
#         score_bank = torch.randn(num_sample, self.classifier[-1].out_features).to(self.device)
#
#         # 预填充特征库和得分库
#         with torch.no_grad():
#             for batch_idx, (x, _) in enumerate(dset_loader_target):
#                 x = x.to(self.device)
#                 features = self.model.extract_features(x)
#                 features_flat = features.view(features.size(0), -1)
#                 logits = self.classifier(features_flat)
#                 probs = F.softmax(logits, dim=1)
#
#                 start = batch_idx * dset_loader_target.batch_size
#                 end = start + x.size(0)
#                 fea_bank[start:end] = features_flat
#                 score_bank[start:end] = probs
#
#         for iter_num in range(self.max_iter):
#             total_loss_value = 0
#             for batch_idx, (x, _) in enumerate(dset_loader_target):
#                 x = x.to(self.device)
#
#                 # 前向传播
#                 with torch.no_grad():
#                     features = self.model.extract_features(x)
#                 features_flat = features.view(features.size(0), -1)
#                 logits = self.classifier(features_flat)
#                 probs = F.softmax(logits, dim=1)
#
#                 # 计算信息最大化损失
#                 im_loss = self.compute_maximization_loss(features)
#
#                 # 计算邻域一致性损失
#                 start = batch_idx * dset_loader_target.batch_size
#                 end = start + x.size(0)
#                 fea_bank[start:end].fill_(-1)  # 排除当前批次
#                 distance = features_flat @ fea_bank.T
#                 _, idx_near = torch.topk(distance, self.k, dim=1)
#                 score_near = score_bank[idx_near].permute(0, 2, 1)
#                 output_re = probs.unsqueeze(1)
#                 loss_const = -torch.log(torch.bmm(output_re, score_near)).sum(-1).mean()
#
#                 # 计算KL散度损失（类别分布对齐）
#                 probs_mean = torch.mean(probs, dim=0)
#                 uniform_dist = torch.ones_like(probs_mean) / probs.size(1)
#                 kl_loss = F.kl_div(torch.log(probs_mean + 1e-6), uniform_dist, reduction='batchmean')
#
#                 # 总损失
#                 total_loss = self.reg_lambda * im_loss + 0.1 * loss_const + 0.1 * kl_loss
#
#                 # 反向传播
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#
#                 # 应用动态梯度掩码
#                 self.mask = self.generate_dynamic_mask(features)
#                 self.apply_gradient_mask()
#
#                 # 梯度裁剪
#                 torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
#                 self.optimizer.step()
#
#                 total_loss_value += total_loss.item() * x.size(0)
#
#             avg_loss = total_loss_value / num_sample
#             print(f"Iteration {iter_num + 1}/{self.max_iter} - Loss: {avg_loss:.4f}")
#
#         return self.model, self.classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SDFAAdapt:
    def __init__(self, model, classifier, mask_old=None, reg_lambda=0.001, max_iter=100):
        self.model = model
        self.classifier = classifier
        self.mask_old = mask_old
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4)
        self.device = next(self.model.parameters()).device
        self._to_linear = None
        self._calculate_flattened_size()

    def _calculate_flattened_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 2000, 30).to(self.device)
            features = self.model.extract_features(x)
            self._to_linear = features.numel()

    def compute_kl_loss(self, probs, smoothed_labels):
        softmax_out = F.softmax(probs, dim=1)
        kl_loss = F.kl_div(torch.log(softmax_out + 1e-6), smoothed_labels, reduction='batchmean')
        return kl_loss

    def compute_maximization_loss(self, features):
        feat_mean = features.mean(dim=0)
        feat_diff = features - feat_mean
        feat_loss = feat_diff.pow(2).sum(dim=1).mean()
        return feat_loss

    def apply_gradient_mask(self):
        if self.mask_old is None:
            return
        for name, param in self.model.named_parameters():
            if 'fc1.weight' in name and param.grad is not None:
                param.grad *= (1.0 - self.mask_old.view(-1, 1).to(self.device))
            if 'fc1.bias' in name and param.grad is not None:
                param.grad *= (1.0 - self.mask_old.to(self.device))
            if 'fc2.weight' in name and param.grad is not None:
                param.grad *= (1.0 - self.mask_old.view(1, -1).to(self.device))

    def train(self, dset_loader_target):
        self.model.train()
        total_loss_value = 0
        total_samples = 0

        for iter_num in range(self.max_iter):
            for x_target, _ in dset_loader_target:  # 忽略标签
                x_target = x_target.to(self.device)

                # 前向传播
                features = self.model.extract_features(x_target)
                features_flat = features.view(features.size(0), -1)
                logits = self.classifier(features_flat)
                probs = F.softmax(logits, dim=1)

                # 生成伪标签（基于预测结果）
                _, pseudo_labels = torch.max(probs.detach(), dim=1)

                # 创建平滑标签（无监督版本）
                smoothed_labels = torch.full_like(probs, fill_value=0.1 / probs.size(1))
                smoothed_labels.scatter_(1, pseudo_labels.unsqueeze(1), 0.9)

                # 计算损失
                kl_loss = self.compute_kl_loss(probs, smoothed_labels)
                im_loss = self.compute_maximization_loss(features)
                total_loss = kl_loss + self.reg_lambda * im_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.apply_gradient_mask()
                self.optimizer.step()

                total_loss_value += total_loss.item() * x_target.size(0)
                total_samples += x_target.size(0)

            avg_loss = total_loss_value / total_samples
            print(f"Iteration {iter_num + 1}/{self.max_iter} - Loss: {avg_loss:.4f}")

        return self.model, self.classifier