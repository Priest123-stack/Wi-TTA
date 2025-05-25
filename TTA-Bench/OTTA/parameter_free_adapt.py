import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from T3A import T3A ,softmax_entropy
from LAME import LAMEWrapper
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        # 定义encoder部分（特征提取）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 定义classifier部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 125 * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def accuracy_ent(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.float()
            y = y.float()
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    network.train()
    accuracy = correct / total
    entropy = ent / total
    print("test_acc{}".format(accuracy))
    # print(f'Accuracy: {accuracy:.4f}, Entropy: {entropy:.4f}')
    return correct / total, ent / total


def tes_time_adaptation(net, test_loader):

        correct = 0
        test_loss = 0
        total = 0
        for i, data in enumerate(test_loader):
            if (i % 50 == 0):
                print(i)
            x_test, y_test = data
            x_test = x_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            total += y_test.size(0)
            outputs = net(x_test)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == y_test.long()).sum().item()
        test_acc = correct / total
        print("test_acc{}".format(test_acc))


if __name__=='__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
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
    print(x_test.shape)
    # 批量化
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(dataset=combined_dataset, batch_size=10, shuffle=True)

    # source_model = CNN1().to(device)
    # source_model = torch.load('D:/model/5300-2.pth')
    base_model = CNN1()
    model_path = 'D:/model/CNN1_3.pth'
    base_model.load_state_dict(torch.load(model_path, weights_only=True))
    base_model.to(device)


    #两个方法

    t3a = T3A(input_shape=(1*1*2000*30), num_classes=7, num_domains=1, filter_K =30 ,algorithm=base_model)
    # print("基模型在测试集上的准确度：")
    # accuracy_ent(network=base_model, loader=combined_loader, weights=None, device=device, adapt=None)
    # print("自适应模型在测试集上的准确度：")
    accuracy_ent(network = t3a, loader = combined_loader, weights= None, device = device, adapt=True)

    # lame = LAMEWrapper(base_model)
    # tes_time_adaptation(base_model,combined_loader)






