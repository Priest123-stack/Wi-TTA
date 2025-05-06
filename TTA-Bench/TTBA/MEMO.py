from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from CSI_Data_Augmentation import CSIDataAugmentation
augment = CSIDataAugmentation()

#定义CNN
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



x_train = np.load('E:/wifi感知/5300-3_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-3_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-3_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-3_npy/y_test.npy')


#为了使数据能让CNN处理转换为图格式数据
x_train = torch.tensor(x_train.reshape(len(x_train), 1, 2000, 30))
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test.reshape(len(x_test), 1, 2000, 30))
y_test = torch.tensor(y_test)
print(x_train.shape)
#批量化
train_dataset =TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size =1, shuffle = True)
test_dataset = TensorDataset(x_test, y_test)
test_loader  = DataLoader(dataset=test_dataset, batch_size =1, shuffle = True)
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
combined_loader = DataLoader(dataset=combined_dataset, batch_size=1,shuffle=True)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits



def adapt_single(image):
    net.eval()
    for iteration in range(args.niter):
        inputs = [augment.augmix_csi(image) for _ in range(args.batch_size)]
        inputs = torch.stack(inputs).cuda()
        inputs = inputs.squeeze(1)
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss, logits = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()

def tes_single(model, image, label):
    model.eval()
    inputs = image
    inputs = inputs.float()
    with torch.no_grad():
        outputs = model(inputs.cuda())
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    return correctness, confidence



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--group_norm', default=8, type=int)
    parser.add_argument('--lr', default=0.0000001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    print('Running...')
    correct = []
    net = CNN()
    model_path1 = 'D:/model/5300-1.pth'
    net.load_state_dict(torch.load(model_path1))
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    for image, label in tqdm(combined_loader):
        net.load_state_dict(torch.load(model_path1))
        adapt_single(image)
        correct.append(tes_single(net, image, label)[0])

    print(f'MEMO adapt test acc {(np.mean(correct))*100:.2f}')