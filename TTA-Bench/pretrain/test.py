
import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class CSIDataAugmentation:
    def __init__(
            self,
            # 概率控制（默认降低概率）
            prob_rotation=0.3,
            prob_crop=0.3,
            prob_amplitude=0.3,
            prob_contrast=0.3,
            prob_multipath=0.2,
            prob_gaussian_noise=0.2,

            # 增强强度参数（可调节）
            rotation_params={'shift_range': (50, 200)},  # 位移范围
            crop_params={'crop_len_range': (1500, 1800)},  # 裁剪范围
            amplitude_params={'scale_range': (0.7, 1.3)},  # 幅度缩放
            contrast_params={'alpha_range': (0.8, 1.2)},  # 对比度
            multipath_params={'num_paths': 2, 'delay_range': 30, 'alpha': 0.1},  # 多径参数
            noise_params={'gaussian_std_range': (0.01, 0.05)}  # 噪声强度
    ):
        # 概率参数
        self.prob_rotation = prob_rotation
        self.prob_crop = prob_crop
        self.prob_amplitude = prob_amplitude
        self.prob_contrast = prob_contrast
        self.prob_multipath = prob_multipath
        self.prob_gaussian_noise = prob_gaussian_noise

        # 增强参数
        self.rotation_params = rotation_params
        self.crop_params = crop_params
        self.amplitude_params = amplitude_params
        self.contrast_params = contrast_params
        self.multipath_params = multipath_params
        self.noise_params = noise_params

    def __call__(self, x):
        # 基础增强（概率降低）
        if random.random() < self.prob_rotation:
            x = self.random_circular_rotation(x)
        if random.random() < self.prob_crop:
            x = self.random_resized_crop(x)
        if random.random() < self.prob_amplitude:
            x = self.random_amplitude(x)
        if random.random() < self.prob_contrast:
            x = self.random_contrast(x)

        # 环境干扰增强（更低概率）
        if random.random() < self.prob_multipath:
            x = self.add_multipath(x)
        if random.random() < self.prob_gaussian_noise:
            x = self.add_gaussian_noise(x)

        return x

    # ========== 修改后的增强方法 ==========
    def random_circular_rotation(self, x):
        shift_min, shift_max = self.rotation_params['shift_range']
        batch_size = x.size(0)
        rotated = []
        for i in range(batch_size):
            n = random.randint(shift_min, shift_max)  # 使用可调节参数
            rotated_sample = torch.cat((x[i:i + 1, :, -n:, :], x[i:i + 1, :, :-n, :]), dim=2)
            rotated.append(rotated_sample)
        return torch.cat(rotated, dim=0)

    def random_resized_crop(self, x):
        crop_min, crop_max = self.crop_params['crop_len_range']
        batch_size = x.size(0)
        cropped_resized = []
        for i in range(batch_size):
            crop_len = random.randint(crop_min, crop_max)
            start_idx = random.randint(0, 2000 - crop_len)
            cropped = x[i:i + 1, :, start_idx:start_idx + crop_len, :]
            resized = F.interpolate(cropped, size=(2000, 30), mode='bilinear', align_corners=False)
            cropped_resized.append(resized)
        return torch.cat(cropped_resized, dim=0)

    def random_amplitude(self, x):
        scale_min, scale_max = self.amplitude_params['scale_range']
        scale = torch.empty(x.size(0), 1, 1, x.size(3)).uniform_(scale_min, scale_max).to(x.device)
        return x * scale

    def random_contrast(self, x):
        alpha_min, alpha_max = self.contrast_params['alpha_range']
        mean = x.mean(dim=2, keepdim=True)
        alpha = torch.empty(x.size(0), 1, 1, x.size(3)).uniform_(alpha_min, alpha_max).to(x.device)
        return (x - mean) * alpha + mean

    def add_multipath(self, x):
        """修正后的多径效应"""
        batch, _, seq_len, subcarriers = x.shape
        multipath = torch.zeros_like(x)
        for _ in range(self.multipath_params['num_paths']):
            delay = np.random.randint(5, self.multipath_params['delay_range'])
            scaled = self.multipath_params['alpha'] * torch.roll(x, shifts=delay, dims=2)
            scaled[:, :, :delay, :] = 0
            multipath += scaled
        return x + multipath  # 移除之前的*0操作

    def add_gaussian_noise(self, x):
        """修正后的噪声添加"""
        noise_min, noise_max = self.noise_params['gaussian_std_range']
        noise_std = torch.empty(1).uniform_(noise_min, noise_max).to(x.device)
        return x + torch.randn_like(x) * noise_std  # 移除*0操作


# ...（可视化函数和主程序保持不变）


# ================== 可视化函数 ==================
def plot_comparison(original, augmented):
    plt.figure(figsize=(15, 6))

    # 动态计算颜色范围
    vmin = min(original.min(), augmented.min())
    vmax = max(original.max(), augmented.max())

    # 热力图对比
    plt.subplot(2, 1, 1)
    plt.imshow(original.T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.title('Original CSI Data (Time × Subcarriers)')
    plt.xlabel('Time Samples')
    plt.ylabel('Subcarriers')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(augmented.T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.title('Augmented CSI Data (Time × Subcarriers)')
    plt.xlabel('Time Samples')
    plt.ylabel('Subcarriers')
    plt.colorbar()

    # 时间序列对比（随机选择5个子载波）
    plt.figure(figsize=(15, 10))
    for i in range(5):
        subcarrier = np.random.randint(0, 30)
        plt.subplot(5, 1, i + 1)
        plt.plot(original[:, subcarrier], label='Original', alpha=0.7)
        plt.plot(augmented[:, subcarrier], label='Augmented', alpha=0.7)
        plt.title(f'Subcarrier {subcarrier} Comparison')
        plt.legend()
    plt.tight_layout()
    plt.show()


# ================== 主程序 ==================
if __name__ == "__main__":
    # 示例数据加载（需替换为真实数据）
    x_train = np.load('E:/wifi感知/5300-3_npy/x_train.npy')
    y_train = np.load('E:/wifi感知/5300-3_npy/y_train.npy')
    x_test = np.load('E:/wifi感知/5300-3_npy/x_test.npy')
    y_test = np.load('E:/wifi感知/5300-3_npy/y_test.npy')

    # 为了使数据能让CNN处理转换为图格式数据
    x_train = torch.tensor(x_train.reshape(len(x_train), 1, 2000, 30))
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test.reshape(len(x_test), 1, 2000, 30))
    y_test = torch.tensor(y_test)
    print(x_train.shape)
    # 批量化
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(dataset=combined_dataset, batch_size=1, shuffle=True)

    # 初始化增强器
    augmenter = CSIDataAugmentation()

    # 获取样本数据
    sample_batch = next(iter(combined_loader))[0]
    original_sample = sample_batch[0].squeeze().numpy()  # [2000, 30]

    # 应用增强
    augmented_batch = augmenter(sample_batch)
    augmented_sample = augmented_batch[0].squeeze().numpy()

    # 打印统计信息
    print(f"[原始数据] 均值: {original_sample.mean():.2f} 标准差: {original_sample.std():.2f}")
    print(f"[增强数据] 均值: {augmented_sample.mean():.2f} 标准差: {augmented_sample.std():.2f}")

    # 可视化对比
    plot_comparison(original_sample, augmented_sample)