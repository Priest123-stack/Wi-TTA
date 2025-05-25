import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from thop import profile
import math

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test).float()
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)


d_model = 30   # 字 Embedding 的维度
d_ff = 128     # 前向传播隐藏层维度
d_k = d_v = 30 # K(=Q), V的维度
n_layers =2    # 有多少个encoder和decoder
n_heads = 6
src_vocab_size = 2000

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()               # pos_table: [max_len, d_model]

    def forward(self, enc_inputs):    # enc_inputs: [batch_size, seq_len, d_model]
        # print("enc_inputs",enc_inputs.shape)
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # print("enc_inputs",enc_inputs.shape)
        return self.dropout(enc_inputs.cuda())

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):  # Q: [batch_size, n_heads, len_q, d_k]
        # print("Q",Q.shape)
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        # print("attn",attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        # print("self.W_Q",self.W_Q)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        # print("input_Q",input_Q.shape)
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V)  # context: [batch_size, n_heads, len_q, d_v]  # attn: [batch_size, n_heads, len_q, len_k]
        # print("context",context.shape)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # print("output",output.shape)
        # print("context", context.shape)
        # print("residual",residual.shape)
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # print("inputs",inputs.shape)
        residual = inputs
        output = self.fc(inputs)
        return output
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs):                                # enc_inputs: [batch_size, src_len, d_model]
        # print("enc",enc_inputs.shape)
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        res=enc_inputs
        enc_inputs=nn.LayerNorm(d_model).cuda()(enc_inputs)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = nn.LayerNorm(d_model).cuda()(res+enc_outputs)
        b=enc_outputs
        enc_outputs0=nn.LayerNorm(d_model).cuda()(enc_inputs)
        enc_outputs0 = self.pos_ffn(enc_outputs0)
        d=b+enc_outputs0  # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs1=nn.LayerNorm(d_model).cuda()(d)
        return enc_outputs1, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.out_layer = nn.Linear(d_model, 7)

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)  # enc_outputs :   [batch_size, src_len, d_model]
            enc_self_attns.append(enc_self_attn)             # enc_self_attn : [batch_size, n_heads, src_len, src_len]
        #print("enc_outputs", enc_outputs.shape)
        # print(enc_self_attns)
        enc_outputs = enc_outputs[:, 0]
        #print("enc_outputs", enc_outputs.shape)
        output = self.out_layer(enc_outputs)
        #print("output", output.shape)
        return output, enc_self_attns

def training():#定义训练方式
    transformer = Encoder().cuda()  # type: object
    cost = nn.CrossEntropyLoss()#定义损失函数
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)#定义优化函数

    epochs = 2#学习周期，1个epoch表示过了1遍训练集中的所有样本。
    lo = [] # lo用于记录每个batch的训练损失值
    los = [] # los用于记录每轮训练的训练集损失和
    for ecpoch in range(epochs):
        train_loss = 0.0#初始化为0
        correct = 0#定义预测正确的图片数，初始化为0
        total = 0 # 总共参与测试的图片数，也初始化为0

        for i ,data in enumerate(train_loader): # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            x_train, y_train = data
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            optimizer.zero_grad()  # 每次反向传播前需要梯度清零
            output,_ = transformer(x_train)
            loss = cost(output, y_train.long())#计算损失值
            loss.backward()#即反向传播求梯度
            optimizer.step()#即更新所有参数
            a, predicted = torch.max(output, 1)
            total += y_train.size(0)# 总共参与测试的数量
            correct += ((predicted == y_train.long()).sum().item())# 预测正确的数量
            train_loss += loss.item()#损失和
            lo.append(loss.item())#每个batch的损失值（loss）添加到一个列表（lo）中，用于记录每个batch的训练损失值。
        train_acc = correct / total
        los.append(train_loss)#训练集损失值（train_loss）添加到一个列表（los）中，用于记录训练过程中每轮训练的损失值

        print("ecpoch:{}  train_loss:{}  train_acc:{} ".format(ecpoch, train_loss, train_acc))
    return transformer
       # if train_acc>=0.5:
         #   torch.save(net, '/home/wxy/WXY/tf_model.pt')
        # with torch.no_grad():  # 进行评测的时候网络不更新梯度，即不进行反向传播和参数更新
        #     correct = 0
        #     total = 0
        #     for i, data in enumerate(test_loader):
        #         x_test, y_test = data
        #         x_test = Variable(x_test).cuda()
        #         y_test = Variable(y_test).cuda()
        #         outputs, _ = transformer(x_test)
        #         _, predicted = torch.max(outputs.data, 1)#torch.max()函数获取每个输出中最大的值及对应的索引，并将索引作为预测结果
        #         total += y_test.size(0)  # labels的长度，参与测试的图片总数
        #         correct += (predicted == y_test.long()).sum().item()  # 预测正确的数目
        #
        #     test_acc = correct / total
        #     print("test_acc:{}".format(test_acc))



if __name__ == '__main__':
    transformer = training()  # 接收返回的模型
    # 计算FLOPs和Params
    input_sample = x_train[:1].cuda()  # 提取第一个样本作为输入示例
    flops, params = profile(transformer, inputs=(input_sample,))
    print(f"\nFLOPs: {flops / 1e6:.2f} M")
    print(f"Params: {params / 1e6:.2f} M\n")
    training()
    model = torch.load('/home/wxy/WXY/tf_model.pt')
    with torch.no_grad():  # 进行评测的时候网络不更新梯度，即不进行反向传播和参数更新
        correct = 0
        total = 0
        for i, data in enumerate(test_loader):
            x_test, y_test = data
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            outputs, _ = transformer(x_test)
            _, predicted = torch.max(outputs.data, 1)  # torch.max()函数获取每个输出中最大的值及对应的索引，并将索引作为预测结果
            total += y_test.size(0)  # labels的长度，参与测试的图片总数
            correct += (predicted == y_test.long()).sum().item()  # 预测正确的数目

        test_acc = correct / total
        print("test_acc:{}".format(test_acc))
