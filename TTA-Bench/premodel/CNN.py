import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from ptflops import get_model_complexity_info

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
# 修正加载标签数据
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
# 修正加载标签数据
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')


# x_train2 = np.load('E:/WiFi感知入门-内部资料（论文+数据集+代码）/5300-3_npy/x_train.npy')
# y_train2 = np.load('E:/WiFi感知入门-内部资料（论文+数据集+代码）/5300-3_npy/y_train.npy')
# x_test2 = np.load('E:/WiFi感知入门-内部资料（论文+数据集+代码）/5300-3_npy/x_test.npy')
# y_test2 = np.load('E:/WiFi感知入门-内部资料（论文+数据集+代码）/5300-3_npy/y_test.npy')

# x_train = np.concatenate([x_train1, x_train2], axis=0)  # 沿着第一个轴（样本数）连接
# y_train = np.concatenate([y_train1, y_train2], axis=0)
#
# # 合并测试数据
# x_test = np.concatenate([x_test1, x_test2], axis=0)
# y_test = np.concatenate([y_test1, y_test2], axis=0)

#为了使数据能让CNN处理转换为图格式数据
x_train = torch.tensor(x_train.reshape(len(x_train),1,2000,30))
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test.reshape(len(x_test),1,2000,30))
y_test = torch.tensor(y_test)
print(x_train.shape)
print(x_test.shape)

#批量化
train_dataset =TensorDataset(x_train,y_train)
train_loader = DataLoader(dataset = train_dataset,batch_size = 10,shuffle = True)
test_dataset = TensorDataset(x_test,y_test)
test_loader  = DataLoader(dataset = test_dataset,batch_size = 10,shuffle = True)
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
combined_loader = DataLoader(dataset=combined_dataset, batch_size=10, shuffle=True)


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

def training(net,train_loader):

    net.train()
    net = net.float()
    cost = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(),lr = 0.0001)
    epoches =30

    for epoch in range (epoches):
        train_loss = 0.0
        correct = 0.0 #正确的图量
        total = 0.0  #总的图数量
        for i ,data in enumerate(train_loader):  #返回当前批次的数据
            # if(i%50==0):
            #     print(i)

            x_train, y_train = data
            x_train = x_train.float()
            y_train = y_train.long()
            x_train, y_train = x_train.to(device), y_train.to(device)

            total +=y_train.size(0)
            optimizer.zero_grad()
            output = net(x_train)
            loss = cost(output,y_train)
            #loss.backward(retain qraph = True) #多次反向传播是否有必要？
            loss.backward()
            optimizer.step()
            _,predicted = torch.max(output.data,dim= 1) # dim=1表示按列查找，即找每一行的最大值，返回最大值及对应的索引，这里只要索引
            correct += (predicted ==y_train).sum().item()
            train_loss +=loss.item()
        train_acc =correct/total
        average_loss = train_loss/total
        print("epoch{},  train_loss{}  acc{}".format(epoch+1,average_loss,train_acc))


def tes_ing(net,test_loader):

    net.eval()  # 切换到评估模式
    # 进行测试，不会更新参数
    with torch.no_grad():

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
            _,predicted = torch.max(outputs,1)
            correct +=(predicted == y_test.long()).sum().item()
        test_acc = correct/total
        print("test_acc{}".format(test_acc))





if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    base_model = CNN()
    base_model.to(device)
    # 计算FLOPs和Params
    macs, params = get_model_complexity_info(base_model, (1, 2000, 30), as_strings=False)
    print(f"FLOPs (M): {macs * 2 / 1e6:.2f}")
    print(f"Params (M): {params / 1e6:.2f}")
    training(base_model,train_loader)
    torch.save(base_model.state_dict(), 'E:/model/5300_1_test.pth')
    tes_ing(base_model, test_loader)


