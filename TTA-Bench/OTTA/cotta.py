from copy import deepcopy
# deepcopy 函数用于创建对象的深拷贝。深拷贝会递归地复制对象及其所有嵌套的对象，
# 也就是说，深拷贝会创建一个新的对象，并且新对象的所有子对象也都是新创建的，与原对象及其子对象在内存中是完全独立的。
import torch
import torch.nn as nn
import torch.jit
# torch.jit 是 PyTorch 提供的一个即时编译模块，主要用于将 PyTorch 模型转换为一种可以被高效执行的中间表示形式。
#  torch.jit.script 函数，可以将 Python 函数或者 torch.nn.Module 子类转换为 TorchScript 代码。
# torch.jit.trace 函数，能对一个 PyTorch 模型进行追踪
import PIL
# 用于导入 Python Imaging Library（PIL）的派生库 Pillow，其提供了丰富的图像处理功能
import torchvision.transforms as transforms
# torchvision.transforms 模块提供了一系列用于图像预处理和数据增强的工具
import OTTA.my_transform as my_transforms
from time import time
import logging


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    img_shape = (2000, 30, 1)
    # 输入图像的形状，这里表示图像的尺寸为 2000x30，通道数为 1
    n_pixels = img_shape[0]
    # n_pixels 提取了图像的第一个维度（高度）的值，即 2000

    clip_min, clip_max = 0.0, 1.0
    # 图像像素值的裁剪范围，将像素值限制在 [0.0, 1.0] 之间

    p_hflip = 0.5
    # 定义了随机水平翻转的概率，这里设置为 0.5，表示有 50% 的概率对图像进行水平翻转

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        # 自定义的裁剪变换类
        my_transforms.ColorJitterPro(
            # 自定义的颜色抖动变换类
            # soft 参数控制了变换的强度，当 soft 为 False 时，变换范围较大[0.6, 1.4]。
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        #transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            #  PyTorch 内置的随机仿射变换类，用于对图像进行随机旋转、平移、缩放和错切
            degrees=[-8, 8] if soft else [-15, 15],
            # 控制旋转角度范围
            translate=(1 / 16, 1 / 16),
            # 控制平移范围
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            # scale 控制缩放比例范围
            shear=None,
            # 当 shear=None 时，意味着不进行错切变换。也就是说，在这次随机仿射变换过程中，图像不会发生倾斜的效果
            interpolation=transforms.InterpolationMode.BILINEAR,
            # 指定插值方法为双线性插值
            # fillcolor=None
            fill=None
        #     fill 用于填充变换后的空白区域。
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        # 用于对图像进行高斯模糊处理，kernel_size 是卷积核的大小，sigma 是高斯核的标准差范围
        #transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        # 随机水平翻转变换类
        my_transforms.GaussianNoise(0, gaussian_std),
        # 自定义的高斯噪声变换类，用于向图像中添加均值为 0、标准差为 gaussian_std 的高斯噪声
        my_transforms.Clip(clip_min, clip_max)
    #     再次使用 my_transforms.Clip 类将图像的像素值裁剪到 [clip_min, clip_max] 范围内。
    ])
    return tta_transforms
# 函数返回创建好的变换组合 tta_transforms，通过调整 soft 和 gaussian_std 参数可以控制变换的强度和噪声水平。


def update_ema_variables(ema_model, model, alpha_teacher):
    # 函数的主要功能是更新指数移动平均（ EMA）模型的参数
    # EMA 是一种常用的技术，它通过对模型参数进行加权平均，得到一个平滑的参数版本，有助于提高模型的泛化能力和稳定性。
    # ema_model：指数移动平均模型，它的参数会根据当前模型的参数进行更新。
    # model：当前正在训练的模型，其参数用于更新 EMA 模型。
    # alpha_teacher：加权系数，用于控制 EMA 模型参数更新的平滑程度
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # 遍历模型参数，分别返回 EMA 模型和当前模型的所有可训练参数
        # zip() 函数将两个模型的参数一一对应地组合成元组，然后通过 for 循环遍历这些元组，依次处理每个对应的参数对
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    #   EMA 参数的更新公式
    #  新的 EMA 参数 = alpha_teacher * 旧的 EMA 参数 + (1 - alpha_teacher) * 当前模型参数
    # 当 alpha_teacher 接近 1 时，EMA 模型的参数更新较慢，更倾向于保留旧的参数值，模型更加平滑；反之参数更新更快
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
        CoTTA（协同测试时自适应）在测试过程中通过最小化熵来适配模型。
    一旦进行了测试时自适应（tented），模型会在每次前向传播时自我更新以进行适配。
    """

    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        # 使用 assert 语句检查 steps 是否大于 0，如果不满足条件，会抛出 AssertionError 并显示错误信息
        # 确保 CoTTA 至少有 1 步来进行前向传播和更新。
        self.episodic = episodic
        # 后续判断是否采用情节式更新策略。

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        # 复制当前模型和优化器的状态，并将结果保存为类的实例属性
        # self.model_ema：可能是模型的指数移动平均版本
        # self.model_anchor：可能是模型的锚点版本
        self.transform = get_tta_transforms()
        # 获取用于测试时增强的变换（前面已定义），并将其保存为类的实例属性
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()
        #     根据 episodic 标志决定是否重置模型状态

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        #     指定的步数内对输入数据进行前向传播和自适应更新，最后返回模型的输出结果。

        return outputs

    def reset(self):
        # 不接受额外的参数，因为它主要是对类内部保存的模型和优化器状态进行操作。
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        # 检查状态是否保存检查状态是否保存
        # 如果这两个状态中的任何一个为 None，说明没有保存模型和优化器的初始状态，抛出异常
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # 将保存的模型状态 self.model_state 和优化器状态 self.optimizer_state
        # 加载到当前的模型 self.model 和优化器 self.optimizer 中。这样就实现了模型和优化器状态的重置。
        # Use this line to also restore the teacher model
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
    #     再次调用 copy_model_and_optimizer 函数，重新复制当前模型和优化器的状态

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    # Python 装饰器，确保被装饰的函数在执行时，即使处于无梯度（torch.no_grad()）的上下文环境中，也能够进行梯度计算。
    # 在深度学习里，测试阶段一般会使用 torch.no_grad() 来禁用梯度计算，以此减少内存消耗并加快计算速度。
    # 不过，在某些情形下，像测试时模型自适应调整这类操作，就需要在测试阶段计算梯度。
    def forward_and_adapt(self, x, model, optimizer):
        # 用于实现模型在测试阶段进行自适应调整的核心方法。
        # 结合了教师模型（model_ema 和 model_anchor）和学生模型（model）
        # 通过一系列操作（如增强平均预测、损失计算、参数更新等）来调整模型参数，以适应测试数据。
        outputs = self.model(x)
        # 前向传播得到学生模型输出
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        # 使用锚点模型 self.model_anchor 对输入数据 x 进行前向传播。
        # 通过 torch.nn.functional.softmax 函数将输出转换为概率分布，dim=1 表示在类别维度上进行 softmax 操作。
        # 接着取每个样本概率分布中的最大值，得到 anchor_prob，用于后续的阈值判断。
        # .max(1)[0]用于找出张量在指定维度（这里是维度 1）上的最大值，并返回这些最大值构成的新张量。
        standard_ema = self.model_ema(x)
        # 使用指数移动平均模型 self.model_ema 对输入数据 x 进行前向传播，得到标准的 EMA 预测结果 standard_ema。
        # Augmentation-averaged Prediction
        N = 32
        # 进行 N = 32 次增强预测
        outputs_emas = []
        for i in range(N):
            outputs_ = self.model_ema(x).detach()
            # 每次使用指数移动平均模型 self.model_ema 对输入数据 x 进行前向传播，
            # 并使用 detach() 方法将输出从计算图中分离，避免后续梯度计算影响该输出。
            outputs_emas.append(outputs_)
        #     将每次的预测结果添加到列表 outputs_emas 中
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0) < self.ap:
            #     计算 anchor_prob 的均值，并与阈值 self.ap 进行比较
            outputs_ema = torch.stack(outputs_emas).mean(0)

            # 如果均值小于阈值，则将 outputs_emas 列表中的所有预测结果通过
            # torch.stack 堆叠成一个张量，然后在第 0 维上求均值，得到增强平均预测结果 outputs_ema。
        else:
            outputs_ema = standard_ema
        #     否则，直接使用标准的 EMA 预测结果 standard_ema 作为 outputs_ema。
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        # 计算学生模型输出 outputs 和教师模型预测结果 outputs_ema 之间的熵损失
        loss.backward()
        optimizer.step()
        # 更新学生模型的参数
        optimizer.zero_grad()
        # 清零优化器中的梯度，以便下一次迭代
        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
        # Stochastic restore，对学生模型的每个模块中的可训练参数（权重和偏置）进行随机恢复操作
        if True:
            for nm, m in self.model.named_modules():
                # 依次访问模型中的每个模块，nm 是模块的名称，m 是对应的模块对象
                for npp, p in m.named_parameters():
                    # 依次访问每个模块中的每个参数，npp 是参数的名称，p 是对应的参数张量
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        # 检查参数的名称是否为权重（weight）和偏置（bias）
                        # p.requires_grad 检查该参数是否需要计算梯度，只有需要计算梯度的参数才会参与后续的随机恢复操作。
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        # torch.rand(p.shape) 会生成一个与参数 p 形状相同的张量，其中的元素是从 [0, 1) 区间内均匀随机采样得到的。
                        # torch.rand(p.shape) < self.rst对生成的随机张量中的每个元素进行比较，返回True or False
                        # .float() 将布尔类型的张量转换为浮点类型的张量
                        # .cuda() 将张量移动到 GPU 上进行计算
                        # self.rst是随机恢复的参数
                        # 生成掩码

                        with torch.no_grad():
                            # 使用 torch.no_grad() 上下文管理器，避免在恢复过程中计算梯度。
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
                    # model_state[f"{nm}.{npp}"]从初始状态中获取参数的初始值，p是当前参数的值
                    # 根据掩码 mask 对参数 p 进行更新，以一定概率将参数恢复到初始状态，其中值为 1.0 恢复为初始状态。
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    # x：通常是模型输出的对数概率（logits）
    # x_ema：同样是对数概率（logits），可能是模型的指数移动平均（EMA）版本的输出。
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
# 对 x_ema 在维度 1 上应用 softmax 函数。softmax 函数的作用是将对数概率转换为概率分布
# 对 x 在维度 1 上应用 log_softmax 函数。log_softmax 是先对输入应用 softmax 函数，再取对数的操作。
# 返回熵公式-sum pi * log(qi)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    # 收集模型中所有可训练的参数，并将这些参数及其对应的名称存储在列表中返回。
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:  # isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    # 判断是否为权重和偏置且需要计算梯度
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    # 返回一个字典，包含了模型中所有可学习参数（如权重和偏置）的当前状态。
    # 这里使用 deepcopy 是为了确保 model_state 是一个独立的副本，修改 model_state 不会影响原始模型的状态。
    model_anchor = deepcopy(model)
    #  deepcopy 函数复制整个模型对象，得到 model_anchor。
    #  model_anchor 可以作为模型的一个锚点版本，在后续的操作中可能用于对比或参考。
    optimizer_state = deepcopy(optimizer.state_dict())
    # 返回一个字典，包含了优化器的当前状态，如学习率、动量等。
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    #  将参数从计算图中分离出来，即在 EMA 模型中的参数不再参与梯度计算
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    # 函数的主要功能是将之前保存的模型和优化器的状态恢复到当前的模型和优化器中。
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    # 将指定的状态字典加载到模型中
    # strict=True 表示严格匹配状态字典中的键和模型中的参数，否则会报错
    optimizer.load_state_dict(optimizer_state)
#     将指定的状态字典加载到优化器中。


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # 将模型设置为训练模式。
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # 禁用模型的梯度计算。为了后续有选择地重新启用需要更新的参数的梯度计算。
    # enable all trainable
    for m in model.modules():
        # model.modules() 会返回一个迭代器，用于遍历模型中的所有模块
        if isinstance(m, nn.BatchNorm2d):
            # 检查对象 m 是否为 nn.BatchNorm2d 类的实例
            m.requires_grad_(True)
            #  nn.BatchNorm2d 模块的参数的 requires_grad 属性设置为 True，
            #  启用这些参数的梯度计算，以便在自适应过程中对其进行更新。
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            # 强制 nn.BatchNorm2d 模块在训练和评估模式下都使用当前批次的统计信息（均值和方差），而不是使用之前累积的全局统计信息。
            m.running_mean = None
            m.running_var = None
        #     确保不使用之前保存的全局统计信息。
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    # 检查模型是否处于训练模式
    """Check model for compatability with tent."""
    is_training = model.training
    # model.training 是一个布尔属性，用于表示模型当前是否处于训练模式。
    assert is_training, "tent needs train mode: call model.train()"
    # assert 语句用于检查 is_training 是否为 True。如果为 False，说明模型处于评估模式，报错绿字信息
    param_grads = [p.requires_grad for p in model.parameters()]
    # 遍历模型的所有参数是否需要计算梯度的属性，并存储在列表 param_grads 中
    has_any_params = any(param_grads)
    # 使用 any 函数检查列表中是否至少有一个元素为 True，即模型是否至少有一个可训练的参数。
    has_all_params = all(param_grads)
    # 使用 all 函数检查列表中的所有元素是否都为 True，即模型的所有参数是否都可训练。
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    # 检查模型是否包含批量归一化层
    assert has_bn, "tent needs normalization for its optimization"