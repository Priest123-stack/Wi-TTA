# import torch.nn as nn
#
# from torch.nn import functional as F
# class AlphaBatchNorm(nn.Module):
#     """ Use the source statistics as a prior on the target statistics """
#
#     @staticmethod
#     def find_bns(parent, alpha):
#         print('1')
#         replace_mods = []
#         if parent is None:
#             return []
#         for name, child in parent.named_children():
#             if isinstance(child, nn.BatchNorm2d):
#                 module = AlphaBatchNorm(child, alpha)
#                 replace_mods.append((parent, name, module))
#             else:
#                 replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))
#
#         return replace_mods
#
#     @staticmethod
#     def adapt_model(model, alpha):
#         replace_mods = AlphaBatchNorm.find_bns(model, alpha)
#         print(f"| Found {len(replace_mods)} modules to be replaced.")
#         for (parent, name, child) in replace_mods:
#             setattr(parent, name, child)
#         return model
#
#     def __init__(self, layer, alpha=0.1):
#         assert alpha >= 0 and alpha <= 1
#
#         super().__init__()
#         self.layer = layer
#         print(self.layer)
#         self.layer.eval()
#         self.alpha = alpha
#
#         self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)
#
#     def forward(self, input):
#         self.norm(input)
#
#         running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
#         running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)
#
#         return F.batch_norm(
#             input,
#             running_mean,
#             running_var,
#             self.layer.weight,
#             self.layer.bias,
#             False,
#             0,
#             self.layer.eps,
#         )
import torch.nn as nn
# from methods.source import Source
# from methods.bn import AlphaBatchNorm, EMABatchNorm
# from utils.registry import ADAPTATION_REGISTRY


# @ADAPTATION_REGISTRY.register()
# class BNTest(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

def norm(model):
        model.eval()
        model.requires_grad_(False)
        # 将模型整体设置为评估模式并禁用梯度计算

        for m in model.modules():
            # Re-activate batchnorm layer
            if (isinstance(m, nn.BatchNorm1d)) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
        #         如果当前模块是批量归一化层，使用 m.train() 将其设置为训练模式。
        #         这样，批量归一化层会根据当前输入数据更新其统计信息（均值和方差），而不是使用预先计算好的全局统计信息。
        return model
