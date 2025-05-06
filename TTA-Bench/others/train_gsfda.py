import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')

x_train = torch.tensor(x_train.reshape(len(x_train),1,2000,30))
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test.reshape(len(x_test),1,2000,30))
y_test = torch.tensor(y_test)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

class MaskedCNN(nn.Module):
    def __init__(self):
        super(MaskedCNN, self).__init__()
        self.layer1 = self._make_layer(1, 16)
        self.layer2 = self._make_layer(16, 32)
        self.layer3 = self._make_layer(32, 64)
        self.layer4 = self._make_layer(64, 128)
        self.fc1 = nn.Linear(128 * 125 * 1, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)
        self.mask = nn.Parameter(torch.ones(256))

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x * self.mask)
        x = self.dropout(x)
        return self.fc2(x)

def train_source(model, train_loader, device):
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()
    reg_lambda = 0.001

    for epoch in range(40):
        total_loss, correct, total = 0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            # x, y = x.float().to(device), y.to(device)
            x, y = x.float().to(device), y.long().to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y) + reg_lambda * torch.norm(model.mask, 1)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            # x, y = x.float().to(device), y.to(device)
            x, y = x.float().to(device), y.long().to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Test Acc: {correct/total:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Running on the GPU" if torch.cuda.is_available() else "Running on the CPU")
    net = MaskedCNN().to(device)
    train_source(net, train_loader, device)
    torch.save(net.state_dict(), 'D:/model/cnn_source1.pth')
    torch.save(net.mask.detach().cpu(), 'D:/model/mask_source1.pt')
    test_model(net, test_loader, device)



