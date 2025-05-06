import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


class WeakTimeSeriesAugmentation:
    def __init__(self, noise_std=0.01, time_shift=5, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.scale_range = scale_range

    def __call__(self, x):
        # 确保输入张量在原始设备上
        device = x.device

        # 加高斯噪声
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise  # 自动保留梯度

        # 随机缩放（在GPU上生成缩放因子）
        scale = torch.empty(1, device=device).uniform_(*self.scale_range)
        x_scaled = x_noisy * scale

        # 随机时间平移（使用 torch.roll）
        shift = torch.randint(-self.time_shift, self.time_shift + 1, (1,), device=device).item()
        x_shifted = torch.roll(x_scaled, shifts=shift, dims=1)

        return x_shifted




class GEOS(nn.Module):
    def __init__(self, model, device='cuda', lr=1e-7, lambda_consistency=0.5):
        super(GEOS, self).__init__()
        self.model = model.to(device)
        self.model.train()  # Test-time 需要保持 BN 统计
        self.device = device
        # print("=== 冻结前参数状态 ===")
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        # 冻结特征提取器参数，解冻辅助模块参数
        for name, param in self.model.named_parameters():
            if "feature_extractor" in name:  # 特征提取器参数冻结
                param.requires_grad = False
            elif "aux_block" in name:  # 辅助模块参数解冻
                param.requires_grad = True
        # # 打印冻结后的参数状态
        # print("\n=== 冻结后参数状态 ===")
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")


        self.augmenter = WeakTimeSeriesAugmentation()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_consistency = lambda_consistency
        # 仅优化需要梯度的参数
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def forward(self, x):
        # return self(x)
        return self.adapt_one_sample(x)

    def adapt_one_sample(self, x):
        self.model.train()  # 强制启用训练模式
        x = x.to(self.device)

        # 生成增强样本（确保在GPU上且保留梯度）
        x_aug = torch.stack([self.augmenter(img).to(self.device) for img in x])
        x_aug.requires_grad_(True)  # 显式启用梯度

        # 前向传播（确保梯度追踪）
        logits_orig = self.model(x)
        logits_aug = self.model(x_aug)

        # # 检查梯度是否可追踪
        # print(f"logits_orig.requires_grad: {logits_orig.requires_grad}")
        # print(f"logits_aug.requires_grad: {logits_aug.requires_grad}")

        # 计算损失
        prob_orig = F.softmax(logits_orig, dim=1)  # 不移除梯度
        log_prob_aug = F.log_softmax(logits_aug, dim=1)
        consistency_loss = F.kl_div(log_prob_aug, prob_orig, reduction='batchmean')
        loss = self.lambda_consistency * consistency_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 检查解冻参数的梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"警告：参数 {name} 的梯度为 None")

        self.optimizer.step()
        return logits_orig
        # return logits_orig.detach()  # 返回时切断梯度




