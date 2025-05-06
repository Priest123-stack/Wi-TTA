"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# current_model_probs：它表示模型概率的移动平均值，记录了模型在之前所有样本上预测概率的平均情况
# 这个值是通过不断更新得到的，在处理新样本时，会结合新样本的预测概率来更新
# 它是一个动态变化的值，用于衡量模型在整个自适应过程中的平均预测概率分布
# outputs：outputs 是模型对当前输入数据 x 进行前向传播后得到的输出结果
# 它代表了模型对当前这一批输入数据的即时预测结果。




class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05):
        # model：需要进行自适应调整的基础模型。
        # optimizer：用于更新模型参数的优化器。
        # fishers：Fisher 正则化项，用于防止模型遗忘之前学习的知识，默认值为 None。
        # fisher_alpha：两个损失之间的权衡系数，默认值为 2000.0。
        # e_margin：超参数E0（公式 3）
        # d_margin：余弦相似度阈值的超参数epsilon（公式 5）
        super().__init__()
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被正确执行。
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # 将 episodic 保存为类的实例属性，用于后续判断是否采用情节式更新策略。

        self.num_samples_update_1 = 0
        # number of samples after First filtering, exclude unreliable samples
        # 记录第一次过滤后（排除不可靠样本）的样本数量。
        self.num_samples_update_2 = 0
        # number of samples after Second filtering, exclude both unreliable and redundant samples
        # 记录第二次过滤后（排除不可靠和冗余样本）的样本数量。
        self.e_margin = e_margin
        # hyper-parameter E_0 (Eqn. 3)保存超参数E_0
        self.d_margin = d_margin
        # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)
        # 保存余弦相似度阈值的超参数

        self.current_model_probs = None
        # the moving average of probability vector (Eqn. 4)
        # 保存概率向量的移动平均值

        self.fishers = fishers
        # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        # 保存 Fisher 正则化项，用于防止模型遗忘之前学习的知识，需要在模型自适应之前计算（公式 9）
        self.fisher_alpha = fisher_alpha
        # trade-off \beta for two losses (Eqn. 8)
        # 保存两个损失之间的权衡系数

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # 注意：如果模型从不进行重置，比如在持续自适应的情况下，
        # 那么跳过状态复制将节省内存
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        #     如果是阶段性自适应，需要将模型状态重置为初始状态，持续自适应则不需要
        if self.steps > 0:
            # self.steps 代表前向传播和自适应调整的步数。
            # 若 self.steps 大于 0，则进入循环执行多次自适应调整。
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
                #  forward_and_adapt_eata 函数会对输入数据 x 进行前向传播，并对模型参数进行自适应调整。它会返回四个值：
                # outputs：模型的输出结果
                # num_counts_2：第二次过滤后新增的样本数量。
                # num_counts_1：第一次过滤后新增的样本数量。
                # updated_probs：更新后的概率向量。
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                # 分别累加，记录总的样本数量。
                self.reset_model_probs(updated_probs)
                # 重置概率向量
        else:
            # 若 self.steps 等于 0，意味着不进行自适应调整。
            self.model.eval()
            # 将模型设置为评估模式self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        #         在无梯度计算的上下文环境下，对输入数据 x 进行前向传播，得到模型的输出结果得到模型的输出结果。
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    #             self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
    #             copy_model_and_optimizer(self.model, self.optimizer)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    # 定义了一个温度参数 temprature，并将其初始值设为 1。
    # 温度参数常用于在 softmax 操作中调整概率分布的平滑程度。
    # 这里 temprature 为 1，此操作实际上不改变 x 的值，但在后续修改温度参数时可用于调整分布。
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    # 输入数据 x、模型 model、优化器 optimizer、Fisher 信息矩阵 fishers、熵阈值 e_margin、当前模型概率的移动平均值 current_model_probs
    # fishers 指的是 Fisher 信息矩阵，它是一个用来衡量模型参数估计的不确定性的矩阵
    # fishers 存储着模型在之前训练阶段学习到的知识的重要性信息，能够帮助模型在测试阶段的自适应过程中避免遗忘之前学到的内容。
    # e_margin 是一个熵阈值，用于筛选可靠的样本。在 EATA 算法中，会计算模型输出的熵，熵值反映了模型预测的不确定性，熵值越小，说明模型对预测结果越有信心。
    # current_model_probs 表示模型概率的移动平均值，它记录了模型在之前所有样本上预测概率的平均情况。
    # 过滤冗余样本，如果 current_model_probs 不为 None，会计算它与可靠样本输出的 softmax 分布之间的余弦相似度
    # fisher_alpha=50.0
    # 含义：这是一个超参数，用于平衡熵损失和弹性权重巩固（ EWC）损失
    # EWC 损失用于防止模型在自适应过程中遗忘之前学习到的知识
    # 在计算总损失时，如果 fishers（Fisher 信息矩阵）不为 None，会计算 EWC 损失并将其加到熵损失上
    # fisher_alpha 越大，EWC 损失对总损失的影响就越大，模型就越倾向于保留之前学习到的知识
    # d_margin用于过滤冗余样本的余弦相似度阈值
    # 如果余弦相似度的绝对值小于 d_margin，则认为该样本是非冗余的，会被保留用于后续的损失计算和参数更新
    # scale_factor对某些数值进行缩放操作
    # num_samples_update 是一个计数器，用于记录经过第二次过滤（排除不可靠和冗余样本）后的样本数量。
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return:
    1. model outputs;
    2. the number of reliable and non-redundant samples;
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    # current_model_probs 表示模型概率的移动平均值，它记录了模型在之前所有样本上预测概率的平均情况。
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    # 使用 torch.where 函数找出熵值小于 e_margin 的样本的索引 filter_ids_1
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    # ds2 这里暂时是一个恒为真的筛选（ids1[0] > -0.1）
    entropys = entropys[filter_ids_1]
    # 根据 filter_ids_1 过滤出可靠样本的熵值 entropys。
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        # unsqueeze(dim=0) 方法用于在指定维度（这里是维度 0）上增加一个维度
        # outputs 是模型的输出结果，filter_ids_1 是经过第一次过滤（筛选出可靠样本）后得到的样本索引。
        # softmax(1) 是对选取的输出在维度 1上应用 softmax 函数，将输出转换为概率分布，使得每个样本的所有类别概率之和为 1。
        # dim=1：指定在哪个维度上计算余弦相似度。这里指定为维度 1
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        # 进一步过滤出非冗余样本的熵值 entropys，并更新 ids2
        ids2 = filter_ids_2
        # 更新 ids2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    #     更新模型概率的移动平均值
    else:
        # 如果 current_model_probs 为 None，则直接调用 update_model_probs 函数更新 updated_probs。
        # 初始情况
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    #     更新模型概率的移动平均值
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # entropys 是一个存储着模型输出的熵值的张量，每个元素代表一个样本的熵
    # clone() 方法会创建 entropys 的一个副本，这样后续对这个副本的操作不会影响原始的 entropys 张量。
    # detach() 方法会将这个副本从计算图中分离出来，意味着在后续的计算中不会对其进行梯度计算。这样做是因为权重系数 coeff 只是用于对损失进行加权。
    # e_margin 是预设的熵阈值
    # 这一步是将每个样本的熵值减去 e_margin，得到一个差值，该差值体现了样本的熵值相对于阈值的偏离程度。
    # torch.exp() 上述得到的差值进行指数运算，将差值映射到一个正数范围，方便后续计算权重系数。
    # 当样本的熵值小于 e_margin 时,那么 coeff 的值大于 1,在损失计算中更重视这些样本。
    # 计算权重系数：根据可靠且非冗余样本的熵值 entropys 计算权重系数 coeff，会给予更高的权重,用于对不同样本的熵损失进行加权
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    # mul() 是 PyTorch 中的逐元素乘法函数，它会将 entropys 张量中的每个元素与 coeff 张量中对应位置的元素相乘。
    # 通过这种逐元素相乘的操作，实现了对每个样本的熵损失进行重新加权。
    loss = entropys.mean(0)
    # 将熵值 entropys 乘以权重系数 coeff 进行重新加权
    # 然后计算加权后熵值的均值作为损失 loss。
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            # 对于每个参数，如果其名称存在于 fishers 字典中，说明该参数在之前的训练中有对应的 Fisher 信息和最优参数值。
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        #   fishers[name][0] 表示该参数对应的 Fisher 信息矩阵，
        # fishers[name][1] 表示该参数在之前训练中的最优值
        # (param - fishers[name][1])**2 计算当前参数值与最优值之间的差值的平方。
        #  fishers[name][0] * 对差值的平方进行加权，权重为 Fisher 信息矩阵。
        # fisher_alpha 是一个超参数，用于控制 EWC 损失在总损失中的权重。
        # 如果 fishers 不为 None，计算弹性权重巩固（EWC）损失 ewc_loss，并将其加到总损失 loss 中。
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        # x[ids1][ids2] 表示经过两次筛选后的输入样本
        # .size(0) 表示筛选后的样本数量
        # 如果样本数量不为 0，则进行反向传播和参数更新
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs
#     返回模型的输出 outputs、可靠且非冗余样本的数量 entropys.size(0)、可靠样本的数量 filter_ids_1[0].size(0) 以及更新后的模型概率的移动平均值 updated_probs。


def update_model_probs(current_model_probs, new_probs):
    # 当前的模型概率移动平均值，初始时可能为 None。
    # new_probs：新样本的预测概率
    if current_model_probs is None:
        # 初始状态,预测为0
        if new_probs.size(0) == 0:
            # new_probs 样本数量为 0
            return None
        # 如果 current_model_probs 为 None 且 new_probs 的样本数量为 0，则返回 None；
        # 否则，计算 new_probs 的均值作为新的模型概率的移动平均值。
        else:
            with torch.no_grad():
                # 如果 new_probs 有样本，使用 torch.no_grad() 上下文管理器
                return new_probs.mean(0)
    #         计算 new_probs 在维度 0 上的均值，将其作为新的模型概率移动平均值返回
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
#使用指数移动平均（, EMA）的方法更新模型概率移动平均值,这里的 0.9 和 0.1 是权重系数，控制了历史信息和新信息在更新过程中的比重


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
   # 收集参数


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state



def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
# 加载


def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            # 表明使用当前批次的统计信息（均值和方差），而不是使用之前累积的全局统计信息
            m.running_mean = None
            m.running_var = None
    return model
# 设置模型


def check_model(model):
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"
    # 检查模型