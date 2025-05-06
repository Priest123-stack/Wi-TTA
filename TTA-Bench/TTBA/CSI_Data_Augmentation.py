import torch
import torch.nn.functional as F
import random
import numpy as np
class CSIDataAugmentation:
    def __init__(
        self,
        prob_rotation=1.0,
        prob_crop=1.0,
        prob_amplitude=1.0,
        prob_contrast=1.0
    ):
        self.prob_rotation = prob_rotation
        self.prob_crop = prob_crop
        self.prob_amplitude = prob_amplitude
        self.prob_contrast = prob_contrast
        self.augmentations = [
            self.random_circular_rotation,
            self.random_resized_crop,
            self.random_amplitude,
            self.random_contrast,
        ]

    def __call__(self, x):
        if random.random() < self.prob_rotation:
            x = self.random_circular_rotation(x)
        if random.random() < self.prob_crop:
            x = self.random_resized_crop(x)
        if random.random() < self.prob_amplitude:
            x = self.random_amplitude(x)
        if random.random() < self.prob_contrast:
            x = self.random_contrast(x)
        return x

    def random_circular_rotation(self, x):
        batch_size = x.size(0)
        rotated = []
        for i in range(batch_size):
            n = random.randint(1, 2000)
            rotated_sample = torch.cat((x[i:i+1, :, -n:, :], x[i:i+1, :, :-n, :]), dim=2)
            rotated.append(rotated_sample)
        return torch.cat(rotated, dim=0)

    def random_resized_crop(self, x):
        batch_size = x.size(0)
        cropped_resized = []
        for i in range(batch_size):
            orig = x[i:i+1]
            crop_len = random.randint(1000, 2000)
            start_idx = random.randint(0, 2000 - crop_len)
            cropped = orig[:, :, start_idx:start_idx+crop_len, :]
            resized = F.interpolate(cropped, size=(2000, 30), mode='bilinear', align_corners=False)
            cropped_resized.append(resized)
        return torch.cat(cropped_resized, dim=0)

    def random_amplitude(self, x):
        batch_size = x.size(0)
        _, _, _, c = x.shape
        scale = torch.empty(batch_size, 1, 1, c).uniform_(0.75, 1.25).to(x.device)
        return x * scale

    def random_contrast(self, x):
        batch_size = x.size(0)
        _, _, _, c = x.shape
        mean = x.mean(dim=2, keepdim=True)
        alpha = torch.empty(batch_size, 1, 1, c).uniform_(0.75, 1.25).to(x.device)
        return (x - mean) * alpha + mean



    def augmix_csi(self, x_orig):
        """对 CSI 数据执行 AugMix 风格增强"""
        w = torch.tensor(np.random.dirichlet([1.0] * 3), dtype=torch.float32, device=x_orig.device)
        m = np.random.beta(1.0, 1.0)

        mix = torch.zeros_like(x_orig)
        for i in range(3):
            x_aug = x_orig.clone()
            for _ in range(np.random.randint(1, 4)):
                aug_func = random.choice(self.augmentations)
                x_aug = aug_func(x_aug)
            mix += w[i] * x_aug

        mixed = m * x_orig + (1 - m) * mix
        return mixed

# import torch
# import random
# import numpy as np
# from scipy.interpolate import interp1d
# import torch.nn.functional as F


# class CSIDataAugmentation:
#     def __init__(
#             self,
#             # 原始参数
#             prob_rotation=0.5,
#             prob_crop=0.5,
#             prob_amplitude=0.8,
#             prob_contrast=0.8,
#             # 新增环境干扰参数
#             prob_multipath=0.7,
#             prob_gaussian_noise=0.9,
#             prob_impulse_noise=0.3,
#             prob_attenuation=0.5,
#             prob_fluctuation=0.4,
#             prob_subcarrier_drop=0.6,
#             prob_time_warp=0.3,
#             # 增强参数配置
#             multipath_params={'num_paths': 3, 'delay_range': 10, 'alpha': 0.3},
#             noise_params={'gaussian_std_range': (0.01, 0.1), 'impulse_magnitude': 5.0},
#             fluctuation_params={'freq_range': (0.1, 2.0), 'amplitude': 0.2}
#     ):
#         # 概率参数
#         self.prob_rotation = prob_rotation
#         self.prob_crop = prob_crop
#         self.prob_amplitude = prob_amplitude
#         self.prob_contrast = prob_contrast
#         self.prob_multipath = prob_multipath
#         self.prob_gaussian_noise = prob_gaussian_noise
#         self.prob_impulse_noise = prob_impulse_noise
#         self.prob_attenuation = prob_attenuation
#         self.prob_fluctuation = prob_fluctuation
#         self.prob_subcarrier_drop = prob_subcarrier_drop
#         self.prob_time_warp = prob_time_warp
#
#         # 增强参数
#         self.multipath_params = multipath_params
#         self.noise_params = noise_params
#         self.fluctuation_params = fluctuation_params
#         self.augmentations = [
#                 self.random_circular_rotation,
#                 self.random_resized_crop,
#                 self.random_amplitude,
#                 self.random_contrast,
#             add_multipath
#
#                ]
#
#     def __call__(self, x):
#         # 基础增强
#         if random.random() < self.prob_rotation:
#             x = self.random_circular_rotation(x)
#         if random.random() < self.prob_crop:
#             x = self.random_resized_crop(x)
#         if random.random() < self.prob_amplitude:
#             x = self.random_amplitude(x)
#         if random.random() < self.prob_contrast:
#             x = self.random_contrast(x)
#
#         # 环境干扰增强
#         if random.random() < self.prob_multipath:
#             x = self.add_multipath(x)
#         if random.random() < self.prob_gaussian_noise:
#             x = self.add_gaussian_noise(x)
#         if random.random() < self.prob_impulse_noise:
#             x = self.add_impulse_noise(x)
#         if random.random() < self.prob_attenuation:
#             x = self.apply_distance_attenuation(x)
#         if random.random() < self.prob_fluctuation:
#             x = self.add_dynamic_fluctuation(x)
#         if random.random() < self.prob_subcarrier_drop:
#             x = self.subcarrier_dropout(x)
#         if random.random() < self.prob_time_warp:
#             x = self.time_warp(x)
#
#         return x
#
#     # ----------------- 原始增强方法 -----------------
#     def random_circular_rotation(self, x):
#         batch_size = x.size(0)
#         rotated = []
#         for i in range(batch_size):
#             n = random.randint(1, 2000)
#             rotated_sample = torch.cat((x[i:i + 1, :, -n:, :], x[i:i + 1, :, :-n, :]), dim=2)
#             rotated.append(rotated_sample)
#         return torch.cat(rotated, dim=0)
#
#     def random_resized_crop(self, x):
#         batch_size = x.size(0)
#         cropped_resized = []
#         for i in range(batch_size):
#             orig = x[i:i + 1]
#             crop_len = random.randint(1000, 2000)
#             start_idx = random.randint(0, 2000 - crop_len)
#             cropped = orig[:, :, start_idx:start_idx + crop_len, :]
#             resized = F.interpolate(cropped, size=(2000, 30), mode='bilinear', align_corners=False)
#             cropped_resized.append(resized)
#         return torch.cat(cropped_resized, dim=0)
#
#     def random_amplitude(self, x):
#         batch_size = x.size(0)
#         _, _, _, c = x.shape
#         scale = torch.empty(batch_size, 1, 1, c).uniform_(0.75, 1.25).to(x.device)
#         return x * scale
#
#     def random_contrast(self, x):
#         batch_size = x.size(0)
#         _, _, _, c = x.shape
#         mean = x.mean(dim=2, keepdim=True)
#         alpha = torch.empty(batch_size, 1, 1, c).uniform_(0.75, 1.25).to(x.device)
#         return (x - mean) * alpha + mean
#
#     # ----------------- 新增环境干扰增强方法 -----------------
#     def add_multipath(self, x):
#         """多径效应模拟"""
#         batch, _, seq_len, subcarriers = x.shape
#         multipath = torch.zeros_like(x)
#         for _ in range(self.multipath_params['num_paths']):
#             delay = np.random.randint(1, self.multipath_params['delay_range'])
#             scaled = self.multipath_params['alpha'] * torch.roll(x, shifts=delay, dims=2)
#             scaled[:, :, :delay, :] = 0
#             multipath += scaled
#         return x + multipath
#
#     def add_gaussian_noise(self, x):
#         """高斯噪声注入"""
#         noise_std = torch.empty(1).uniform_(*self.noise_params['gaussian_std_range']).to(x.device)
#         return x + torch.randn_like(x) * noise_std
#
#     def add_impulse_noise(self, x):
#         """脉冲噪声注入"""
#         mask = torch.rand_like(x) < self.noise_params.get('impulse_prob', 0.02)
#         return x + mask.float() * torch.randn_like(x) * self.noise_params['impulse_magnitude']
#
#     def apply_distance_attenuation(self, x):
#         """距离衰减模拟"""
#         attenuation = torch.empty(1).uniform_(0.5, 1.5).to(x.device)
#         return x * attenuation
#
#     def add_dynamic_fluctuation(self, x):
#         """动态波动干扰"""
#         batch, _, seq_len, subcarriers = x.shape
#         freq = torch.empty(1).uniform_(*self.fluctuation_params['freq_range']).item()
#         t = torch.arange(seq_len) / seq_len
#         fluctuation = self.fluctuation_params['amplitude'] * torch.sin(2 * np.pi * freq * t)
#         return x + fluctuation.view(1, 1, -1, 1).to(x.device)
#
#     def subcarrier_dropout(self, x):
#         """子载波随机丢弃"""
#         mask = torch.rand(x.size(-1)).ge(self.prob_subcarrier_drop).float().to(x.device)
#         return x * mask.view(1, 1, 1, -1)
#
#     def time_warp(self, x):
#         """时间轴非线性扭曲"""
#         device = x.device
#         x_np = x.cpu().numpy()
#         warped = np.zeros_like(x_np)
#
#         for b in range(x_np.shape[0]):
#             for s in range(x_np.shape[3]):
#                 orig = x_np[b, 0, :, s]
#                 warp_strength = np.random.uniform(-0.2, 0.2)
#                 new_indexes = np.linspace(0, 1 + warp_strength, len(orig))
#                 f = interp1d(np.linspace(0, 1, len(orig)), orig, fill_value="extrapolate")
#                 warped[b, 0, :, s] = f(new_indexes[:len(orig)])
#
#         return torch.from_numpy(warped).to(device)
#
#     def augmix_csi(self, x_orig):
#         """对 CSI 数据执行 AugMix 风格增强"""
#         w = torch.tensor(np.random.dirichlet([1.0] * 3), dtype=torch.float32, device=x_orig.device)
#         m = np.random.beta(1.0, 1.0)
#
#         mix = torch.zeros_like(x_orig)
#         for i in range(3):
#             x_aug = x_orig.clone()
#             for _ in range(np.random.randint(1, 4)):
#                 aug_func = random.choice(self.augmentations)
#                 x_aug = aug_func(x_aug)
#             mix += w[i] * x_aug
#
#         mixed = m * x_orig + (1 - m) * mix
#         return mixed



