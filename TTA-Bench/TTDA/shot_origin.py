import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SHOTBase(nn.Module):
    def __init__(self, src_model):
        super().__init__()
        # 特征提取器包含layer1~layer4和fc1
        self.feature_extractor = nn.Sequential(
            src_model.layer1,
            src_model.layer2,
            src_model.layer3,
            src_model.layer4,
            nn.Flatten(),  # 自动展平特征
            src_model.fc1,  # 包含fc1层
            nn.ReLU(),
            nn.Dropout(0.5)  # 与源模型结构一致
        )

        # 分类器为源模型的fc2（冻结）
        self.classifier = src_model.fc2
        for param in self.classifier.parameters():
            param.requires_grad = False

        # 优化器仅优化特征提取器
        self.optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=1e-5,
            weight_decay=1e-4
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def compute_loss(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        features = features.view(x.size(0), -1)

        # 计算logits
        logits = self.classifier(F.dropout(F.relu(features), training=True))
        prob = F.softmax(logits, dim=1)

        # === 信息最大化损失 ===
        entropy_loss = -torch.mean(torch.sum(prob * torch.log(prob + 1e-6), dim=1))
        mean_prob = torch.mean(prob, dim=0)
        diversity_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-6))

        # === 自监督伪标签损失 ===
        with torch.no_grad():
            pseudo_labels = torch.argmax(prob, dim=1)
            centroids = []
            for cls_idx in range(logits.size(1)):
                mask = (pseudo_labels == cls_idx)
                if torch.sum(mask) > 0:
                    cls_feat = features[mask]
                    centroids.append(cls_feat.mean(dim=0))
                else:
                    centroids.append(torch.zeros_like(features[0]))
            # 统一堆叠并归一化
            centroids = torch.stack(centroids)
            centroids = F.normalize(centroids, dim=1)

        # 计算余弦相似度
        norm_feat = F.normalize(features, dim=1)
        sim_matrix = torch.mm(norm_feat, centroids.t())

        # 伪标签交叉熵损失
        pl_loss = F.cross_entropy(sim_matrix, pseudo_labels)

        # 总损失
        total_loss = 0*entropy_loss + 0*diversity_loss + 1 * pl_loss
        return total_loss

    def adapt(self, data_loader, epochs=30):
        self.train()  # 初始设为训练模式

        best_acc = 0.0
        for epoch in range(epochs):
            # === 训练阶段 ===
            self.train()
            total_loss = 0.0
            for batch in data_loader:
                x, _ = batch  # 假设数据加载器包含标签（仅用于验证）
                x = x.to(next(self.parameters()).device)

                self.optimizer.zero_grad()
                loss = self.compute_loss(x)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # === 验证阶段 ===
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in data_loader:
                    x, y_true = batch  # 获取真实标签
                    x = x.to(next(self.parameters()).device)
                    y_true = y_true.to(x.device)

                    logits = self.forward(x)
                    y_pred = logits.argmax(dim=1)

                    correct += (y_pred == y_true).sum().item()
                    total += y_true.size(0)

            epoch_acc = correct / total * 100
            print(f"{epoch_acc:.2f}%")

            # 记录最佳准确率
            if epoch_acc > best_acc:
                best_acc = epoch_acc

        # 输出最终结果
        print(f"\n=== 训练完成 ===")
        print(f"最佳准确率: {best_acc:.2f}%")
        print(f"最终准确率: {epoch_acc:.2f}%")


# 使用示例
# 假设源模型结构如下
class SourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 512能保持长宽不变，输出通道数为16
            # [3959, 16, 2000, 30]
            nn.BatchNorm2d(16, momentum=0.9),
            # 16：代表输入通道数（num_features；
            # momentum=0.9：这是用于计算移动平均统计量（均值和方差）的动量参数
            # momentum默认值是 0.1，这里设置为 0.9 意味着更看重之前批次的统计信息
            # 经过批量归一化层后，输出特征图的形状不变
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 长宽减半1000*15
            # 在进行最大池化操作时，会在输入数据的每个 2×2 区域内选取最大值作为输出。减少数据量，提取重要特征
            # [3959, 16, 1000, 15]
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # [3959, 32, 1000, 15]
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 500*7
            # [3959, 16, 500, 7]
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 250*3
            # [3959, 64, 250, 3]
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # batch 128 125 1 四维
        )
        # [3959, 128, 125, 1]
        # ！！！！！原来的几行
        self.fc1 = nn.Linear(128 * 125 * 1, 256)  # 输入 batch 128*125*1 二维
        # 256：代表输出特征的数量，即全连接层的输出维度
        self.Dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)




# 准备目标域数据加载器
from torch.utils.data import DataLoader,TensorDataset
id='5300-1'
# id='5300-2'
# id='5300-3'
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

# def setup_shot(model):
#     shot_model = shot.SHOTWrapper(model)
#     return shot_model


if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # 初始化
    base_model= SourceModel()
    model_path = 'D:/model/5300-3.pth'
    base_model.load_state_dict(torch.load(model_path, weights_only=True))
    base_model.to(device)
    shot_model = SHOTBase(base_model)
    shot_model.adapt(combined_loader, epochs=30)
    # 最终测试（可选）
    shot_model.eval()
    with torch.no_grad():
        # 单独测试目标域数据（示例）
        correct_test = 0
        total =0
        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            logits = shot_model(x)
            correct_test += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        # print(f"\n目标域测试准确率: {correct_test / len(combined_dataset) * 100:.2f}%")
        test_acc = correct_test / total
        print("test_acc{}".format(test_acc))