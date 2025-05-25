import torch
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import logging
import math
import OTTA.cotta as cotta
import OTTA.EATA as EATA
import OTTA.tent as tent
import OTTA.ecotta as ecotta
import OTTA.norm as norm
import TTDA.shot as shot

logger = logging.getLogger(__name__)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# torch.nn.functional 通常简称为 F
# 提供了大量的神经网络相关的函数，涵盖了激活函数、损失函数、卷积操作、池化操作等
# torch.nn.functional：提供了无状态的函数，这些函数不包含可学习的参数
# torch.nn：主要包含各种神经网络层和模块，它们包含可学习的参数
# torch.nn 更适合用于构建神经网络的结构
# torch.nn.functional 更适合用于定义网络的前向传播过程和一些临时的计算。

# id='5300-1'
# id='5300-2'
id='5300-3'
x_train = np.load('E:/wifi感知/'+id+'_npy/x_train.npy')
y_train = np.load('E:/wifi感知/'+id+'_npy/y_train.npy')
x_test = np.load('E:/wifi感知/'+id+'_npy/x_test.npy')
y_test = np.load('E:/wifi感知/'+id+'_npy/y_test.npy')
print(x_train.shape)
#为了使数据能让CNN处理转换为图格式数据
x_train = torch.tensor(x_train.reshape(len(x_train),1,2000,30)).float()
#  reshape 是 numpy 数组的一个方法，用于改变数组的形状。
#  (len(x_train), 1, 2000, 30) 是新的形状，分别为样本数量，通道数，高度or时间步长，宽度or每个时间步的特征数量
# 后面两个参数，如果为图像代表：高度，宽度；如果为时间序列数据代表：时间步长，每个时间步的特征数量
y_train = torch.tensor(y_train)
# torch.tensor 将 numpy 数组或其他可迭代对象转换为 PyTorch 的张量对象，作为输入传递给卷积层、全连接层等
x_test = torch.tensor(x_test.reshape(len(x_test),1,2000,30)).float()
y_test = torch.tensor(y_test)
# print(x_train.shape)
# torch.Size([3959, 1, 2000, 30])，样本数量为3959
# print(x_test.shape)
# torch.Size([990, 1, 2000, 30])
#批量化
train_dataset =TensorDataset(x_train,y_train)
train_loader = DataLoader(dataset = train_dataset,batch_size = 30,shuffle = True)
test_dataset = TensorDataset(x_test,y_test)
test_loader  = DataLoader(dataset = test_dataset,batch_size = 30,shuffle = True)
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
combined_loader = DataLoader(dataset=combined_dataset, batch_size=30, shuffle=True)


#定义CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),#512能保持长宽不变，输出通道数为16
            # [3959, 16, 2000, 30]
        nn.BatchNorm2d(16, momentum=0.9),
            #16：代表输入通道数（num_features；
            # momentum=0.9：这是用于计算移动平均统计量（均值和方差）的动量参数
            # momentum默认值是 0.1，这里设置为 0.9 意味着更看重之前批次的统计信息
            # 经过批量归一化层后，输出特征图的形状不变
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)#长宽减半1000*15
            #在进行最大池化操作时，会在输入数据的每个 2×2 区域内选取最大值作为输出。减少数据量，提取重要特征
            # [3959, 16, 1000, 15]
        )

        self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # [3959, 32, 1000, 15]
        nn.BatchNorm2d(32, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)#500*7
            # [3959, 16, 500, 7]
        )

        self.layer3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) #250*3
            # [3959, 64, 250, 3]
        )

        self.layer4 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128, momentum=0.9),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) #batch 128 125 1 四维
        )
            # [3959, 128, 125, 1]
        # ！！！！！原来的几行
        self.fc1 = nn.Linear(128*125*1,256)#输入 batch 128*125*1 二维
        # 256：代表输出特征的数量，即全连接层的输出维度
        self.Dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,7)
        # self.feature_layer = self.layer4
        #！！！！！


        # # Add device attribute to track which device the model is using
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x =x.view(x.size(0),-1)
        # x.size() 用于获取张量 x 各个维度的大小，即样本数量
        # -1表示展平相乘，比如经过view(2, 3 * 4 * 5) = (2, 60)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.Dropout(x)
        #随机地 “丢弃”（即置为 0）一部分神经元，每个神经元被 “丢弃”的概率为0.5
        out = self.fc2(x)
        return out

    #  gsdfa
    def extract_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def classify(self, feat):
        feat = F.relu(self.fc1(feat))
        return self.fc2(feat)

    # Geos
    # def forward(self, x):
    #     x = self.feature_extractor(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.aux_block(x)
    #     return x

class CNN_geos(nn.Module):
    def __init__(self):
        super(CNN_geos, self).__init__()
        # 特征提取器（卷积层）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 类似定义 layer2, layer3, layer4...
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 辅助模块（全连接层）
        self.aux_block = nn.Sequential(
            nn.Linear(128*125*1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.aux_block(x)
        return x

def test_time_adaptation(net,test_loader):
        net.train()
        correct = 0
        test_loss = 0
        total = 0
        for i,data in enumerate(test_loader):
            if (i % 50 == 0):
                print(i)
            x_test, y_test = data
            x_test = x_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            total += y_test.size(0)
            outputs = net(x_test)
            # outputs = net.update(x_test)
            _,predicted = torch.max(outputs,1)
            correct +=(predicted == y_test.long()).sum().item()
        test_acc = correct/total
        print("test_acc{}".format(test_acc))
# def evaluate_model(model, sample, device, adapt_steps=1):
#     """评估模型包含自适应逻辑的推理时间"""
#     # 计算可训练参数量占比
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     trainable_ratio = (trainable_params / total_params) * 100
#
#     # 确保样本维度正确 (4D: [batch, channel, height, width])
#     if sample.dim() == 3:
#         sample = sample.unsqueeze(0)  # 添加批次维度
#     sample = sample.to(device)
#
#     # 强制启用训练模式（触发自适应逻辑）
#     model.train()
#
#     # 预热（避免冷启动误差）
#     with torch.no_grad():  # 预热时不更新参数
#         for _ in range(10):
#             _ = model(sample)
#
#     # 正式测量（包含自适应逻辑）
#     num_runs = 100
#     start_time = time.time()
#     for _ in range(num_runs):
#         # 允许梯度计算但手动清除梯度，避免内存泄漏
#         model.zero_grad()
#         outputs = model(sample)
#         # 部分方法（如 CoTTA）会隐式执行 backward
#         # if hasattr(model, 'optimizer'):
#         #     model.optimizer.step()  # 显式执行参数更新
#     end_time = time.time()
#     avg_time = (end_time - start_time) * 1000 / num_runs
#
#     print(f"\n评估指标（含自适应逻辑）:")
#     print(f"可训练参数量占比: {trainable_ratio:.2f}%")
#     print(f"单样本平均推理时间: {avg_time:.2f}ms")
#     return trainable_ratio, avg_time

def evaluate_model(model, sample, device, adapt_steps=1):
    # 计算可训练参数量占比
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params) * 100

    # # 确保样本维度正确
    # if sample.dim() == 3:
    #     sample = sample.unsqueeze(0)
    # sample = sample.to(device)
    #
    # # 预热（禁用梯度）
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = model(sample)
    #
    # # 正式测量（禁用梯度）
    # num_runs = 100
    # start_time = time.time()
    # with torch.no_grad():  # 禁用梯度计算
    #     for _ in range(num_runs):
    #         _ = model(sample)
    # end_time = time.time()
    # avg_time = (end_time - start_time) * 1000 / num_runs

    print(f"\n评估指标（含自适应逻辑）:")
    print(f"可训练参数量占比: {trainable_ratio:.2f}%")
    # print(f"单样本平均推理时间: {avg_time:.2f}ms")
    return trainable_ratio
# , avg_time
def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    设置测试时的归一化自适应。
    通过使用测试批次的统计信息对特征进行归一化来实现自适应。
    统计信息是针对每个批次独立测量的；
    不使用滑动平均或其他跨批次的估计方法。
    """
    # norm_model = norm.AlphaBatchNorm(model,0.1)
    norm_model = norm.norm(model)
    # params, param_names = tent.collect_params(model)

    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    设置 Tent 自适应。
    将模型配置为可进行训练，并通过批次统计信息进行特征调制；
    收集用于通过梯度优化进行特征调制的参数；
    设置优化器，然后对模型进行 Tent 自适应调整。
    """
    model = tent.configure_model(model)
    #将模型配置为可进行训练
    params, param_names = tent.collect_params(model)
    optimizer = optim.Adam(params,
                           lr=1e-5,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    #params是需要优化的参数集合
    #betas 用于控制一阶矩估计和二阶矩估计的指数衰减率
    #一阶矩估计0.9决定了之前梯度信息对当前更新的影响程度,第二个元素 0.999用于控制梯度平方的加权平均
    #weight_decay 表示权重衰减系数，也称为 L2 正则化系数，在损失函数中添加一个正则化项，防止模型过拟合
    # weight_decay 不为0时，优化器会在更新参数时对参数进行一定程度的衰减。反之为 0，不使用权重衰减。
    tent_model = tent.Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    #steps 表示在处理每个测试批次数据时，优化器进行参数更新的步数。steps=1每批只更新一次
    # episodic=False 时，Tent 方法会在整个测试过程中持续更新模型的参数，不会进行参数重置。
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    #logger.info 是日志记录器的一个方法，用于记录信息级别的日志。
    #info 方法通常用于记录程序运行过程中的一些重要信息，方便后续的调试和监控。
    return tent_model

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    设置 TENT 自适应。
    将模型配置为可通过批量统计信息进行训练和特征调制，
    收集用于通过梯度优化进行特征调制的参数，
    设置优化器，然后对模型进行 TENT 自适应调整。
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = optim.Adam(params,
                           lr=1e-6,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=1,
                           episodic=False,
                           mt_alpha=0.999,
                            #mt_alpha 是均值教师的衰减系数，值越接近 1，历史参数的影响越大。
                           rst_m=0.01,
                            #随机自训练的重置概率，表示有 1% 的概率重置模型的预测结果
                           ap=0.92
                            #ap是自适应伪标签的阈值，只有预测置信度高于这个阈值的样本才会被用于更新模型参数
                              )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    # cotta.check_model(model)
    return cotta_model

def setup_ecotta(model):
    base_model = model
    model = ecotta.ecotta_networks(model)
    print(model)
    model = ecotta.configure_model(model)
    params, names = ecotta.collect_params(model)
    print(names)
    optimizer = optim.Adam(params,
                           lr=1e-6,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    ecotta_model = ecotta.ecotta_networks(base_model, optimizer)

    return ecotta_model


def setup_eata(model):

    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    optimizer = optim.Adam(params,
                           lr=1e-5,
                           betas=(0.9, 0.999),
                           weight_decay=0)

    eata_model = EATA.EATA(model, optimizer, e_margin=math.log(1000) / 2 - 1, d_margin=0.05)
    #e_margin 是熵阈值。在 EATA 方法中，熵常被用来衡量模型预测的不确定性。
    # 通过设定一个熵阈值，可以筛选出那些预测不确定性处于特定范围的样本，这些样本会被用于后续的参数更新
    #d_margin 是分布阈值。EATA 方法会考虑模型预测分布的变化情况，d_margin 用于控制模型预测分布的变化幅度。
    # 只有当预测分布的变化超过这个阈值时，才会对模型参数进行更新
    return eata_model

def setup_shot(model):
    shot_model = shot.SHOTWrapper(model)
    return shot_model


if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # base_model = CNN()
    # # base_model=CNN_geos()
    # model_path = 'D:/model/5300-3.pth'
    # # model_path = 'D:/model/CNN_geos_2.pth'
    # base_model.load_state_dict(torch.load(model_path, weights_only=True))
    # base_model.to(device)
    # sample = x_test[0].to(device)  # 直接取测试集第一个样本，形状应为[1, 2000, 30]

    # test_time_adaptation(base_model, test_loader)
    # test_time_adaptation(base_model, combined_loader)

    # eata_model = setup_eata(base_model)
    # test_time_adaptation(eata_model, combined_loader)
    # evaluate_model(eata_model, sample, device)
    # #
    # cotta_model = setup_cotta(base_model)
    # test_time_adaptation(cotta_model, combined_loader)
    # evaluate_model(cotta_model, sample, device)
    #
    # tent_model =setup_tent(base_model)
    # # tent_model.to(device)
    # test_time_adaptation(tent_model, combined_loader)
    # evaluate_model(tent_model, sample, device)

    # # 可训练参数量占比: 0.00%,不属于算法一类，排除了
    # norm_model = setup_norm(base_model)
    # test_time_adaptation(norm_model, combined_loader)
    # evaluate_model(norm_model,sample,device)

    # #梯度为0
    # shot_model = setup_shot(base_model)
    # test_time_adaptation(shot_model, combined_loader)
    # evaluate_model(shot_model,sample,device)

    # 梯度为0
    # ecotta_model =setup_ecotta(base_model)
    # ecotta_model.to(device)
    # test_time_adaptation(ecotta_model,combined_loader)
    # evaluate_model(ecotta_model, sample, device)

    # # 梯度为0
    # GEOS_model = GEOS(base_model)
    # test_time_adaptation(GEOS_model, combined_loader)
    # evaluate_model(GEOS_model, sample, device)


    model_path = "D:/model/cnn_source1.pth"
    mask_path = "D:/model/mask_source1.pt"
    base_model = CNN().to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    mask_tensor = torch.load(mask_path, map_location=device)
    # # 确保特征提取器参数被冻结
    # for param in base_model.parameters():
    #     param.requires_grad = False
    # classifier = nn.Sequential(
    #     base_model.fc1,
    #     base_model.fc2
    # ).to(device)

    # 创建分类器
    classifier = nn.Sequential(
        base_model.fc1,
        nn.ReLU(),
        base_model.Dropout,
        base_model.fc2
    ).to(device)
    for param in classifier.parameters():
        param.requires_grad = True
    # 初始化多层掩码
    mask_dict = {
        'fc1.weight': torch.ones(256, 1),  # 与fc1.weight形状匹配
        'fc1.bias': torch.ones(256),
        'fc2.weight': torch.ones(7, 1),
        'fc2.bias': torch.ones(7)
    }

    sdfa_adapt = test2.SDFAAdapt(
        model=base_model,
        classifier=classifier,
        mask_old=mask_dict,
        reg_lambda=0.001,
        max_iter=31
    )
    # sdfa_adapt = test2.SDFAAdapt(model=base_model, classifier=classifier,
    #                              mask_old=mask_tensor, reg_lambda=0.001,
    #                              max_iter=10)  # 增大迭代次数以便观察趋势

    trained_model, trained_classifier = sdfa_adapt.train(combined_loader)

    # 在main.py的可视化部分修改如下
    plt.figure(figsize=(10, 6))  # 增加画布高度

    # 生成密集刻度（从0.6到0.8，步长0.02）
    y_ticks = np.arange(0.68, 0.78, 0.01)
    plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks])  # 倾斜刻度标签

    # 设置纵坐标范围和网格
    plt.ylim(0.68, 0.78)
    plt.grid(True, linestyle='--', alpha=0.7)  # 虚线网格增强可读性

    # 优化后的完整绘图代码
    plt.plot(sdfa_adapt.get_gap_history(),
             'r--o',
             linewidth=1.5,
             markersize=8,
             markerfacecolor='white',
             markeredgewidth=1.5)

    plt.title('伪标签错误率随训练轮数的变化', fontsize=14, pad=20)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('错误率', fontsize=12)
    plt.xticks(range(len(sdfa_adapt.get_gap_history())))

    # 添加辅助元素
    plt.gca().spines['top'].set_visible(False)  # 移除顶部边框
    plt.gca().spines['right'].set_visible(False)  # 移除右侧边框
    plt.tight_layout()  # 自动调整边距

    plt.show()
    # sdfa_adapt = gsdfa.SDFAAdapt(model=base_model, classifier=classifier, mask_old=mask_tensor, reg_lambda=0.001,
    #                 max_iter=1)
    # trained_model, trained_classifier = sdfa_adapt.train(combined_loader)
    # test_time_adaptation(trained_model, combined_loader)
    # 训练完成后添加：
    # test2.plot_training_curve(sdfa_adapt.kl_history, sdfa_adapt.class_names)
    # evaluate_model(trained_model, sample, device)

    # # 在模型加载后添加参数冻结验证
    # base_model = CNN().to(device)
    # base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # base_model.eval()
    # evaluate_model(base_model, sample, device)

