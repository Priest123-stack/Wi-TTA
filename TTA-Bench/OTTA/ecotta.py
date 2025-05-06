import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn.modules import Module
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),#512能保持长宽不变
        nn.BatchNorm2d(16, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)#长宽减半1000*15
        )

        self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(32, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)#500*7
        )

        self.layer3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) #250*3
        )

        self.layer4 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) #batch 128 125 1 四维
        )
        self.fc1 = nn.Linear(128*125*1,256)#输入 batch 128*125*1 二维
        self.Dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,7)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x =x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.Dropout(x)
        out = self.fc2(x)
        return out

class conv_block(Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        # return F.relu(self.conv(x))

class build_meta_block(Module):
    def __init__(self, BNchannel, inchannel, outchannel, kernel_size, stride, padding):
        super().__init__()
        self.meta_bn = nn.BatchNorm2d(BNchannel) #nn.Identity() #nn.Identity() #nn.BatchNorm2d(in_out_depth_s[1]) #nn.Identity()
        self.conv_block = conv_block(inchannel, outchannel, kernel_size, stride,padding)
        self.pool=nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        out = self.conv_block(x)
        out = self.pool(out)
        return out


class one_part_of_networks(Module):
    def __init__(self, original_part, meta_part):
        super().__init__()
        self.original_part = original_part
        self.meta_part = meta_part
        self.btsloss = None
        self.cal_mseloss = False

    def forward(self, x):
        # See Algorithm 1 in the paper (page13)
        if not self.cal_mseloss:
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
        else:
            x = x.detach()
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
            loss = nn.L1Loss(reduction='none')
            self.btsloss = loss(out, out1.detach()).mean()
        return out


class ecotta_networks(nn.Module):
    def __init__(self, model, optimizer=None):
        super(ecotta_networks, self).__init__()

        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 默认使用SGD优化器
        self.optimizer = optimizer
        self.conv1 = model.layer1
        self.encoders = nn.ModuleList([
            model.layer2,
            model.layer3,
            model.layer4
        ])
        self.classifier = nn.Sequential(model.fc1, model.fc2)


        self.meta_parts = []
        # meta_part1 = build_meta_block(32, 16, 32, 5, 1, 2)
        # self.meta_parts.append(meta_part1)
        # meta_part2 = build_meta_block(64, 32, 64, 5, 1, 2)
        # self.meta_parts.append(meta_part2)
        meta_part3 = build_meta_block(128, 64, 128, 5, 1, 2)
        self.meta_parts.append(meta_part3)

        for i in range(3):
            print(i)
            if (i==2):
                self.encoders[2] = one_part_of_networks(self.encoders[2], self.meta_parts[0])




    def forward(self, x ):

        out = self.conv1(x)
        for encoder in self.encoders:
            out = encoder(out)

        out = out.view(x.size(0), -1)
        out = self.classifier(out)

        loss = softmax_entropy(out).mean(0)
        # print(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return out

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """收集 model 中所有 meta_part 的参数"""
    params = []
    names = []

    # 遍历模型的所有模块
    for nm, m in model.named_modules():
        # 检查是否为 one_part_of_networks 中的 meta_part
        if isinstance(m, build_meta_block):  # 根据你提供的定义，meta_part 应该是 build_meta_block 类
            # 遍历 meta_part 内的参数（例如 meta_bn 和 conv_block）
            for sub_nm, sub_m in m.named_modules():
                if isinstance(sub_m, nn.BatchNorm2d) or isinstance(sub_m, nn.Conv2d) or isinstance(sub_m, nn.MaxPool2d):
                    for param_name, param in sub_m.named_parameters():
                        params.append(param)
                        names.append(f"{nm}.{sub_nm}.{param_name}")

    return params, names


def configure_model(model):
    # 设置模型为训练模式
    model.train()

    # 冻结除 meta_parts 外的所有网络参数
    for param in model.parameters():
        param.requires_grad = True

    # # 解冻 meta_parts 内的参数
    # if hasattr(model, 'meta_part'):  # 确保 meta_parts 存在
    #     for meta_part in model.meta_parts:
    #         for param in meta_part.parameters():
    #             param.requires_grad = True
    return model

