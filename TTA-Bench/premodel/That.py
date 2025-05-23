# 在文件顶部添加以下代码以抑制警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import time
from copy import deepcopy

import torch
import numpy as np
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
#

x_train = np.load('E:/wifi感知/5300-1_npy/x_train.npy')
y_train = np.load('E:/wifi感知/5300-1_npy/y_train.npy')
x_test = np.load('E:/wifi感知/5300-1_npy/x_test.npy')
y_test = np.load('E:/wifi感知/5300-1_npy/y_test.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
##
## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Gaussian Encoding ------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Gaussian_Position(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_dim_time,
                 var_num_gaussian=10):
        #
        ##
        super(Gaussian_Position, self).__init__()
        #
        ## var_embedding: shape (var_dim_k, var_dim_feature)
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype=torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        #
        ## var_position: shape (var_dim_time, var_dim_k)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad=False)
        #
        ## var_mu: shape (1, var_dim_k)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time / var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad=True)
        #
        ## var_sigma: shape (1, var_dim_k)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad=True)

    #
    ##
    def calculate_pdf(self,
                      var_position,
                      var_mu,
                      var_sigma):
        #
        ##
        var_pdf = var_position - var_mu  # (position-mu)
        #
        var_pdf = - var_pdf * var_pdf  # -(position-mu)^2
        #
        var_pdf = var_pdf / var_sigma / var_sigma / 2  # -(position-mu)^2 / (2*sigma^2)
        #
        var_pdf = var_pdf - torch.log(var_sigma)  # -(position-mu)^2 / (2*sigma^2) - log(sigma)
        #
        return var_pdf

    #
    ##
    def forward(self,
                var_input):
        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)

        var_pdf = torch.softmax(var_pdf, dim=-1)
        #
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        #
        # print(var_input.shape, var_position_encoding.shape)
        var_output = var_input + var_position_encoding.unsqueeze(0)
        #
        return var_output


#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Encoder ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Encoder(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_num_head=10,
                 var_size_cnn=[1, 3, 5]):
        #
        ##
        super(Encoder, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature,
                                                           var_num_head,
                                                           batch_first=True)
        #
        self.layer_dropout_0 = torch.nn.Dropout(0.1)
        #
        ##
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, 1e-6)
        #
        layer_cnn = []
        #
        for var_size in var_size_cnn:
            #
            layer = torch.nn.Sequential(torch.nn.Conv1d(var_dim_feature,
                                                        var_dim_feature,
                                                        var_size,
                                                        padding="same"),
                                        torch.nn.BatchNorm1d(var_dim_feature),
                                        torch.nn.Dropout(0.1),
                                        torch.nn.LeakyReLU())
            layer_cnn.append(layer)
        #
        self.layer_cnn = torch.nn.ModuleList(layer_cnn)
        #
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input


        var_t = self.layer_norm_0(var_t)
        #

        var_t, _ = self.layer_attention(var_t, var_t, var_t)

        var_t = self.layer_dropout_0(var_t)
        #
        var_t = var_t + var_input
        #
        ##
        var_s = self.layer_norm_1(var_t)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim=0)
        #
        var_s = torch.sum(var_c, dim=0) / len(self.layer_cnn)
        #
        var_s = self.layer_dropout_1(var_s)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_output = var_s + var_t
        #
        return var_output


#
##
## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- THAT -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class THAT(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(THAT, self).__init__()
        #
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        self.layer_left_pooling = torch.nn.AvgPool1d(kernel_size=20, stride=20)
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, var_dim_time // 20)
        #
        var_num_left = 4
        var_dim_left = var_dim_feature
        self.layer_left_encoder = torch.nn.ModuleList([Encoder(var_dim_feature=var_dim_left,
                                                               var_num_head=10,
                                                               var_size_cnn=[1, 3, 5])
                                                       for _ in range(var_num_left)])
        #
        self.layer_left_norm = torch.nn.LayerNorm(var_dim_left, eps=1e-6)
        #
        self.layer_left_cnn_0 = torch.nn.Conv1d(in_channels=var_dim_left,
                                                out_channels=128,
                                                kernel_size=8)

        self.layer_left_cnn_1 = torch.nn.Conv1d(in_channels=var_dim_left,
                                                out_channels=128,
                                                kernel_size=16)
        #
        self.layer_left_dropout = torch.nn.Dropout(0.5)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        self.layer_right_pooling = torch.nn.AvgPool1d(kernel_size=20, stride=20)
        #
        var_num_right = 1
        var_dim_right = var_dim_time // 20
        self.layer_right_encoder = torch.nn.ModuleList([Encoder(var_dim_feature=var_dim_right,
                                                                var_num_head=10,
                                                                var_size_cnn=[1, 2, 3])
                                                        for _ in range(var_num_right)])
        #
        self.layer_right_norm = torch.nn.LayerNorm(var_dim_right, eps=1e-6)
        #
        self.layer_right_cnn_0 = torch.nn.Conv1d(in_channels=var_dim_right,
                                                 out_channels=16,
                                                 kernel_size=2)

        self.layer_right_cnn_1 = torch.nn.Conv1d(in_channels=var_dim_right,
                                                 out_channels=16,
                                                 kernel_size=4)
        #
        self.layer_right_dropout = torch.nn.Dropout(0.5)
        #
        ##
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input  # shape (batch_size, time_steps, features)
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        var_left = torch.permute(var_t, (0, 2, 1))
        var_left = self.layer_left_pooling(var_left)
        var_left = torch.permute(var_left, (0, 2, 1))
        #
        var_left = self.layer_left_gaussian(var_left)
        #
        for layer in self.layer_left_encoder: var_left = layer(var_left)
        #
        var_left = self.layer_left_norm(var_left)
        #
        var_left = torch.permute(var_left, (0, 2, 1))
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        #
        var_left_0 = torch.sum(var_left_0, dim=-1)
        var_left_1 = torch.sum(var_left_1, dim=-1)
        #
        var_left = torch.concat([var_left_0, var_left_1], dim=-1)
        var_left = self.layer_left_dropout(var_left)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        var_right = torch.permute(var_t, (0, 2, 1))  # shape (batch_size, features, time_steps)
        var_right = self.layer_right_pooling(var_right)
        #
        for layer in self.layer_right_encoder: var_right = layer(var_right)
        #
        var_right = self.layer_right_norm(var_right)
        #
        var_right = torch.permute(var_right, (0, 2, 1))
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        #
        var_right_0 = torch.sum(var_right_0, dim=-1)
        var_right_1 = torch.sum(var_right_1, dim=-1)
        #
        var_right = torch.concat([var_right_0, var_right_1], dim=-1)
        var_right = self.layer_right_dropout(var_right)
        #
        ## concatenate
        var_t = torch.concat([var_left, var_right], dim=-1)
        #
        var_output = self.layer_output(var_t)
        #
        ##
        return var_output


def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device):
    """
    [description]
    : generic training function for WiFi-based models
    [parameter]
    : model: Pytorch model to train
    : optimizer: optimizer to train model (e.g., Adam)
    : loss: loss function to train model (e.g., BCEWithLogitsLoss)
    : data_train_set: training set
    : data_test_set: test set
    : var_threshold: threshold to binarize sigmoid outputs
    : var_batch_size: batch size of each training step
    : var_epochs: number of epochs to train model
    : device: device (cuda or cpu) to train model
    [return]
    : var_best_weight: weights of trained model
    """
    #
    ##
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle = True, pin_memory = True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))
    #
    ##
    var_best_accuracy = 0
    var_best_weight = None
    #
    ##
    for var_epoch in range(var_epochs):
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_time_e0 = time.time()
        #
        model.train()
        model = model.to(torch.bfloat16)
        #
        for data_batch in data_train_loader:
            #
            ##
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)
            data_batch_x = data_batch_x.to(device).to(torch.bfloat16)  # 将数据转换为 BFloat16
            data_batch_y = data_batch_y.to(device).to(torch.bfloat16)

            #
            predict_train_y = model(data_batch_x)
            #
            var_loss_train = loss(predict_train_y,
                                  data_batch_y.reshape(data_batch_y.shape[0], -1).float())
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            optimizer.step()
        #
        ##
        predict_train_y = (torch.sigmoid(predict_train_y) > var_threshold).float()

        # 转换 data_batch_y 为 NumPy 数组，使用 .detach() 确保它从计算图中分离出来
        data_batch_y_numpy = data_batch_y.detach().cpu().to(torch.float32).numpy()

        # 转换 predict_train_y 为 NumPy 数组
        predict_train_y = predict_train_y.detach().cpu().numpy()

        # 重塑 predict_train_y 的形状
        predict_train_y = predict_train_y.reshape(-1, data_batch_y.shape[-1])

        # # 将 data_batch_y 转换为整数类型并转换为 NumPy 数组
        # data_batch_y_int = data_batch_y.to(torch.int).cpu().numpy()
        #
        # # 计算训练集的准确率
        # var_accuracy_train = accuracy_score(data_batch_y_int, predict_train_y.argmax(axis=1))
        if data_batch_y.dim() == 1:  # 如果标签是类别索引
            data_batch_y_int = data_batch_y.cpu().numpy()
        else:  # 如果标签是one-hot编码
            data_batch_y_int = data_batch_y.argmax(dim=1).cpu().numpy()

        var_accuracy_train = accuracy_score(data_batch_y_int,
                                            predict_train_y.argmax(axis=1))

        # 输出准确率
        print(f"Training accuracy: {var_accuracy_train}")


        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        model.eval()
        #
        with torch.no_grad():
            #
            ##
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            #
            predict_test_y = model(data_test_x)
            #
            var_loss_test = loss(predict_test_y,
                                 data_test_y.reshape(data_test_y.shape[0], -1).float())
            #
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
            #
            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()
            #
            predict_test_y = predict_test_y.reshape(-1, data_test_y.shape[-1])
            data_test_y = data_test_y.reshape(-1, data_test_y.shape[-1])
            #
            var_accuracy_test = accuracy_score(data_test_y.astype(int),
                                               predict_test_y.astype(int))
        #
        ## ---------------------------------------- Print -----------------------------------------
        #
        # print(f"Epoch {var_epoch}/{var_epochs}",
        #       "- %.6fs" % (time.time() - var_time_e0),
        #       "- Loss %.6f" % var_loss_train.cpu(),
        #       "- Accuracy %.6f" % var_accuracy_train,
        #       "- Test Loss %.6f" % var_loss_test.cpu(),
        #       "- Test Accuracy %.6f" % var_accuracy_test)
        #
        ##
        if var_accuracy_test > var_best_accuracy:
            #
            var_best_accuracy = var_accuracy_test
            var_best_weight = deepcopy(model.state_dict())
    #
    ##
    return var_loss_train.item()
#
##
preset = {
    #
    ## define model
    "model": "THAT",                                    # "ST-RF", "MLP", "LSTM", "CNN-1D", "CNN-2D", "CLSTM", "ABLSTM", "THAT"
    #
    ## define task
    "task": "activity",                                 # "identity", "activity", "location"
    #
    ## number of repeated experiments
    "repeat": 10,
    #
    ## path of data
    "path": {
        "data_x": "dataset/wifi_csi/amp",               # directory of CSI amplitude files
        "data_y": "dataset/annotation.csv",             # path of annotation file
        "save": "result.json"                           # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0", "1", "2", "3", "4", "5"],    # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["2.4"],                           # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["classroom"],                   # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,                                 # default length of CSI
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 1e-3,                                     # learning rate
        "epoch": 200,                                   # number of epochs
        "batch_size": 128,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
    },
    #
    ## encoding of activities and locations
    "encoding": {
        "activity": {                                   # encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "location": {                                   # encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
}
def run_that(data_train_x,
             data_train_y,
             data_test_x,
             data_test_y,
             var_repeat=10):
    """
    [description]
    : run WiFi-based model THAT
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape, data_train_y[0].reshape(-1).shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(THAT(var_x_shape, var_y_shape),
                                                     var_x_shape, as_strings=False)
    #
    # print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_that = THAT(var_x_shape, var_y_shape).to(device)
        #
        optimizer = torch.optim.Adam(model_that.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model=model_that,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_test_set=data_test_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device)
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_that.load_state_dict(var_best_weight)
        # 计算FLOPs和Params
        macs, params = get_model_complexity_info(model_that, var_x_shape, as_strings=False)
        print(f"FLOPs (M): {macs * 2 / 1e6:.2f}")
        print(f"Params (M): {params / 1e6:.2f}")


        #
        with torch.no_grad():
            predict_test_y = model_that(torch.from_numpy(data_test_x).to(device))
        #
        predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##
        # data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
        if data_test_y.ndim == 2:  # 如果是one-hot编码
            data_test_y_c = data_test_y.argmax(axis=1)
        else:
            data_test_y_c = data_test_y

        predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
        #
        ## Accuracy
        # result_acc = accuracy_score(data_test_y_c.astype(int),
        #                             predict_test_y_c.astype(int))
        result_acc = accuracy_score(data_test_y_c,
                                    predict_test_y_c.argmax(axis=1))
        print(f"test_acc {result_acc:.4f}")

        #
        ## Report
        result_dict = classification_report(data_test_y_c,
                                            predict_test_y_c,
                                            digits=6,
                                            zero_division=0,
                                            output_dict=True)
        #
        result["repeat_" + str(var_r)] = result_dict
        #
        result_accuracy.append(result_acc)
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        #
        print("repeat_" + str(var_r), result_accuracy)
        print(result)
    #
    ##
    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    result["complexity"] = {"parameter": var_params, "flops": var_macs * 2}
    #
    return result


# 修改主程序部分
if __name__ == '__main__':
    # 数据预处理
    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], -1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], -1)

    # 获取输入输出维度
    var_x_shape = x_train_reshaped[0].shape
    var_y_shape = y_train[0].reshape(-1).shape

    # 计算复杂度
    macs, params = get_model_complexity_info(THAT(var_x_shape, var_y_shape),
                                             var_x_shape,
                                             as_strings=False)
    print(f"FLOPs (M): {macs * 2 / 1e6:.2f}")
    print(f"Params (M): {params / 1e6:.2f}")

    # 运行训练测试流程
    result = run_that(x_train, y_train, x_test, y_test)

    # 提取最终结果
    print(f"\nFinal Result:")
    print(f"train_loss {result['accuracy']['avg']:.4f}")  # 此处需根据实际数据结构调整
    print(f"test_acc {result['accuracy']['avg']:.4f}")