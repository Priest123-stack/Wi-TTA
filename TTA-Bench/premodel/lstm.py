import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from ptflops import get_model_complexity_info

# 修正标签文件路径（确保y_train和y_test来自正确的标签文件）
x_train = np.load('E:/wifi感知/5300-2_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-2_npy/y_train.npy')  # 修改为y_train.npy
x_test = np.load('E:/wifi感知/5300-2_npy/x_test.npy')  # 修改为x_test.npy
y_test = np.load('E:/wifi感知/5300-2_npy/y_test.npy')  # 修改为y_test.npy

# 确保标签的维度正确（一维）
y_train = y_train.squeeze()  # 去除多余的维度（如果存在）
y_test = y_test.squeeze()

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train).long()  # 直接转换为long类型
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test).long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.fc = nn.Linear(hidden_size, 7)

    def forward(self, x):
        x, (hn, cn) = self.LSTM(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, batch_first):  # 参数名修正为num_layers
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,  # 确保参数顺序正确
#             batch_first=batch_first  # 使用关键字参数明确指定
#         )
#         self.fc = nn.Linear(hidden_size, 7)
#     def forward(self, x):
#         x, hidden = self.rnn(x)
#         x = x[:, -1, :]
#         x = self.fc(x)
#         return x


def training(module, train_loader):
    module.train()  # 显式设置模型为训练模式
    module = module.float()
    cost = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(module.parameters(), lr=0.0001)
    epoches = 100

    for epoch in range(epoches):
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader):
            # if i % 50 == 0:
            #     print(i)
            x_train, y_train = data
            x_train = x_train.float().to(device)
            y_train = y_train.long().to(device)

            total += y_train.size(0)
            optimizer.zero_grad()
            output = module(x_train)
            loss = cost(output, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, dim=1)
            correct += (predicted == y_train).sum().item()
            train_loss += loss.item()
        train_acc = correct / total
        average_loss = train_loss / total
        print(f"epoch {epoch + 1}, train_loss {average_loss:.4f}, acc {train_acc:.4f}")

# ...（其余代码保持不变）

def testing(module,test_loader):
    module.eval()
    with torch.no_grad():
        correct = 0
        test_loss = 0
        total = 0
        for i,data in enumerate(test_loader):
            x_test, y_test = data
            x_test = x_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            total += y_test.size(0)
            outputs = module(x_test)
            _,predicted = torch.max(outputs,1)
            correct +=(predicted == y_test.long()).sum().item()
        test_acc = correct/total
        print("test_acc{}".format(test_acc))





if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    LSTM = LSTM(30, 64, 3, True).to(device)

    # 计算FLOPs和Params
    macs, params = get_model_complexity_info(LSTM, (x_train.shape[1], x_train.shape[2]), as_strings=False)
    print(f"FLOPs (M): {macs * 2 / 1e6:.2f}")
    print(f"Params (M): {params / 1e6:.2f}")

    training(LSTM, train_loader)
    testing(LSTM, test_loader)
    # # 测试RNN
    # print("\n==== Testing RNN ====")
    # rnn_model = RNN(30, 64, 3, True).to(device)  # 参数需与LSTM对齐
    # macs, params = get_model_complexity_info(rnn_model, (x_train.shape[1], x_train.shape[2]), as_strings=False)
    # print(f"RNN FLOPs (M): {macs * 2 / 1e6:.2f}")
    # print(f"RNN Params (M): {params / 1e6:.2f}")
    # training(rnn_model, train_loader)
    # testing(rnn_model, test_loader)