# Wi-TTA
TTA - Bench 项目使用说明

一、项目概述

TTA - Bench 是一个基于 PyTorch 的深度学习项目，主要用于无线感知领域的模型训练、测试和自适应调整。项目包含多种模型，如 CNN、RNN、GeOS 等，以及数据增强、自适应训练和评估等功能。

二、环境要求

Python 版本：Python 3.x

主要依赖库：

torch：深度学习框架

numpy：用于数值计算

pandas：用于数据处理

matplotlib：用于数据可视化

seaborn：用于数据可视化

ptflops：用于计算模型的 FLOPs 和参数数量

sklearn：用于机器学习相关操作

可以使用以下命令安装依赖库：

bash

pip install torch numpy pandas matplotlib seaborn ptflops scikit - learn

三、数据集准备

项目使用的数据集为无线感知相关数据，数据以 .npy 格式存储。数据集包含训练集和测试集，分别存储在 x_train.npy、y_train.npy、x_test.npy 和 y_test.npy 文件中。请将数据集文件放置在指定路径下，如 E:/wifi感知/5300 - 2_npy/ 或 E:/wifi感知/5300 - 3_npy/。

四、代码结构

项目主要包含以下几个模块：

premodel：预训练模型模块，包含 CNN、RNN、THAT 等模型的定义和训练代码。

OTTA：在线测试时自适应（OTTA）模块，包含参数无关自适应训练和评估代码。

TTBA：测试时批量自适应（TTBA）模块，包含数据增强和自适应训练代码。

compare：模型比较模块，包含不同算法的准确率对比可视化代码。

TTDA：测试时域自适应（TTDA）模块，包含多次运行模型并统计准确率的代码。

3C - GAN.py：3C - GAN 模型代码。

GeOS.py：GeOS 自适应模型代码。

五、使用方法

1. 预训练模型训练

以 CNN 模型为例，在 premodel/CNN.py 文件中，取消注释以下代码进行模型训练：


python

运行

training(base_model, train_loader)

torch.save(base_model.state_dict(), 'E:/model/5300_1.pth')

运行命令：

bash

python premodel/CNN.py

2. 模型测试

在 premodel/CNN.py 文件中，取消注释以下代码进行模型测试：


python

运行

tes_ing(base_model, test_loader)

运行命令：

bash

python premodel/CNN.py

3. 自适应训练和评估

以 GeOS 自适应模型为例，运行 GeOS.py 文件：

bash

python GeOS.py

该脚本将加载预训练模型，进行无监督自适应训练，并在测试集上进行评估，同时生成数据可视化结果。

4. 模型比较

运行 compare/epoch_acc.py 文件，生成不同算法的训练轮数与准确率关系的折线图：

bash

python compare/epoch_acc.py

5. 多次运行模型并统计准确率

运行 TTDA/run_ttda.py 文件，多次运行 shot_origin.py 脚本并统计准确率：

bash

python TTDA/run_ttda.py

六、注意事项

请确保数据集文件路径正确，否则会导致数据加载失败。

在运行代码前，请检查是否有 GPU 可用，若有 GPU，代码将自动使用 GPU 进行计算。

部分代码中的模型路径和数据集路径可能需要根据实际情况进行修改。

七、贡献

如果你对本项目有任何建议或改进，欢迎提交 Pull Request 或 Issue。
