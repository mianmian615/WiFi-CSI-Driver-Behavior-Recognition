# networks.py
# 主要功能：定义各种主干网络结构（MLP、ContextNet、Featurizer、Classifier等），为算法模块提供统一的特征提取和分类接口。
 
# 主要内容：
# 1. MLP：多层感知机，适合结构化输入。
# 2. ContextNet1D/2D：一维/二维卷积上下文网络。
# 3. Featurizer：根据hparams动态选择backbone。
# 4. Classifier：根据hparams选择线性或非线性分类头。
# 5. WholeFish：特定算法的整体网络封装。
# 6. cg_classfier：自定义1D卷积分类器。
#

from numpy import True_
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from widgbed.models import *

class MLP(nn.Module):
    """
    多层感知机（MLP），适合结构化输入。作为域分类器
    """
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
            for _ in range(hparams["mlp_depth"] - 2)
        ])
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs
    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ContextNet1D(nn.Module):
    """
    一维卷积上下文网络。
    """
    def __init__(self, input_shape):
        super(ContextNet1D, self).__init__()
        self.context_net = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.context_net(x)

class ContextNet2D(nn.Module):
    """
    二维卷积上下文网络。
    """
    def __init__(self, input_shape):
        super(ContextNet2D, self).__init__()
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )
    def forward(self, x):
        return self.context_net(x)

# 动态获取算法类
# 未用到的函数，保留以备后续扩展
# def get_algorithm_class(algorithm_name):
#     if algorithm_name not in globals():
#         raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
#     return globals()[algorithm_name]

def Featurizer(input_shape, hparams):
    """
    根据hparams动态选择backbone。
    """
    backbones_class = get_backbones_class(hparams['NIC'], hparams['backbone'])
    return backbones_class(input_shape, hparams)

def Classifier(in_features, out_features, hparams, nonlinear=False):
    """
    分类头，根据hparams选择线性或非线性结构。
    """
    if hparams['backbone'] == "CNN_GRU":
        return torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, out_features),
            torch.nn.Softmax(dim=1)
        )
    if nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

class WholeFish(nn.Module):
    """
    特定算法（如Fish）整体网络封装。
    """
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams)
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))
    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))
    def forward(self, x):
        return self.net(x)

class cg_classfier(nn.Module):
    """
    自定义1D卷积分类器。
    """
    def __init__(self, n_outputs, num_classes):
        super(cg_classfier, self).__init__()
        self.l1 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.l2 = nn.BatchNorm1d(512)
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_outputs, num_classes)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x