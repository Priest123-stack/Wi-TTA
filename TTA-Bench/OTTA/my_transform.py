import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda
# ColorJitter 类用于对图像的颜色属性进行随机调整，包括亮度、对比度、饱和度和色调
# Compose 类允许将多个图像变换操作组合成一个序列。它接受一个变换列表作为输入，然后依次对图像应用这些变换。
# Lambda 类允许用户自定义一个匿名函数作为图像变换操作。它接受一个 Python 函数作为参数，该函数将被应用到输入的图像上。
from numpy import random

class GaussianNoise(torch.nn.Module):
    # 向输入图像添加高斯噪声
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std #标准差
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        # torch.randn(img.size()) 会生成一个与 img 尺寸相同的张量，服从（0，1）正态分布
        noise = noise.to(img.device)
        # 将生成的噪声张量移动到与输入图像 img 相同的设备，（如 CPU 或 GPU）上
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        # 返回的字符串 GaussianNoise(mean=0.0, std=1.0)
class Clip(torch.nn.Module):
    # 对输入的图像张量进行裁剪操作
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    #     裁剪操作的最小值和最大值

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)
    # 张量中的元素值限制在指定的最小值和最大值之间
    # 如果元素的值小于 self.min_val，则将其替换为 self.min_val
    # 如果元素的值大于 self.max_val，则将其替换为 self.max_val。

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""
    # 用于对图像的颜色属性进行随机调整
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')
#通过调用 self._check_input 方法对 gamma参数进行检查和处理，确保其符合特定的要求，并将处理后的结果保存为self.gamma

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            # 检查 brightness 参数是否被提供，如果提供了，如(0.5, 1.5)：
            brightness_factor = random.uniform(brightness[0], brightness[1])
            # 在(0.5, 1.5)随机生成一个亮度调整因子
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
            #transforms 是一个存储图像变换操作的列表
            #Lambda 定义了一个匿名函数，该函数接受一个图像 img 作为输入，并使用 F.adjust_brightness 函数来调整图像的亮度。
            #将这个自定义的亮度调整操作添加到 transforms 列表中，以便后续对图像进行处理。
        if contrast is not None:#对比度
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:#饱和度
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:#色调
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            # 检查 gamma 参数是否已被设置，进而决定是否执行与伽马校正相关的操作
            # gamma 值小于 1 时会使图像变亮，大于 1 时会使图像变暗
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)#transforms 列表中的元素顺序会被随机打乱
        transform = Compose(transforms)#将多个图像变换操作组合成一个序列。

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        # 使用 PyTorch 库生成一个长度为 5 的0~n-1随机排列的整数张量，如tensor([3, 1, 4, 0, 2])
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)
            # torch.tensor(1.0)：创建一个值为 1.0 的单元素张量
            # uniform_()从指定的亮度调整因子范围中随机选取一个值，然后使用该值对图像的亮度进行调整F.adjust_brightness()。

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string