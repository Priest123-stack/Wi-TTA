from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import numpy as np

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # 实例属性赋值

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
    #     保存模型和优化器状态

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs
    # 向前传播

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        # copy_model_and_optimizer(self.model, self.optimizer)
        # 加载当前的模型和优化器并重新复制



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
# 熵计算

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    # print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

# @torch.no_grad()
# def efficient_perturb_parameters(parameters, random_seed: int, uniform: bool=False, use_beta: bool=False, scaling_factor=1):
#     torch.manual_seed(random_seed)
#     e = 0.001
#     for name, param in parameters:
#         if uniform:
#             # uniform distribution over unit sphere
#             z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#             z = z / torch.linalg.norm(z)
#         else:
#             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#         param.data = param.data + scaling_factor * z * e
#
# @torch.no_grad()
# def zo_forward(model, x):
#     model.eval()
#     outputs = model(x)
#     original_loss = softmax_entropy(outputs).mean(0)
#     return original_loss.detach()
#
# @torch.no_grad()
# def forward_and_adapt(model, x, num_perturbations=6):
#     print('开始zoTTA!')
#     parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
#     for _ in range(num_perturbations):
#         with torch.no_grad():
#             random_seed = np.random.randint(1000000000)
#             efficient_perturb_parameters(parameters, random_seed, uniform=False)
#             loss1 = zo_forward(model, x)
#             efficient_perturb_parameters(parameters, random_seed, scaling_factor=-1, uniform=False)
#             loss2 = zo_forward(model, x)
#         projected_grad = (loss1 - loss2) / (2 * 0.001)
#         # self.efficient_perturb_parameters(parameters, random_seed, uniform=False)
#             # 对每个参数立即应用梯度更新
#         torch.manual_seed(random_seed)
#         for name, param in parameters:
#                 z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
#                 param.data = param.data - 0.005 * projected_grad * z
#     outputs = model(x)
#     return outputs

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
# 收集归一化层的可训练参数

def collect_allparams(model):
    """Collect all parameters from the model.

    Walk through the model's modules and collect all parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for name, param in model.named_parameters():
        params.append(param)
        names.append(name)
    return params, names
# 收集所有参数
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    #         只更新归一化层的参数
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"