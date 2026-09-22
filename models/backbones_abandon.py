#放置一些废弃不用/效果不好的模型代码

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# 'RNN', # 'RNNold', # CSIDA旧的好一点点， NTUFI新的好很多
# 'GRU', # 'GRUold', # CSIDA、NTUFI新的好
# 'LSTM',# 'LSTMold',  # CSIDA、NTUFI新的好
# 'BiLSTM', # 'BiLSTMold',  # CSIDA新的好  IGA_BiLSTM'很耗GPU，'IGA_BiLSTM2'没问题
# 新'RNN2', 'GRU2', 'LSTM2', 'BiLSTM2'的精度比旧的好很多。
# 新的hidden_size=512，有self.layer_norm(x)和self.layer_pooling(x)，旧的hidden_size=128，无self.layer_norm(x)和self.layer_pooling(x)

# =========================
# RNN/GRU/LSTM/BiLSTM/CNN_GRU工厂
# =========================

def RNN_Factory(input_shape, hparams, rnn_type='RNN'):
    """
    RNN/GRU/LSTM/BiLSTM工厂函数，根据hparams动态构建网络。
    """
    class _RNN(nn.Module):
        def __init__(self, input_shape, hparams,rnn_type):
            super().__init__()
            self.input_shape = input_shape
            self.hparams = hparams
            self.input_size = self.input_shape[1]*self.input_shape[2] if self.hparams['NIC'] == "BVP" else self.input_shape[0]*self.input_shape[1]
            self.seq_len = input_shape[-1]
            self.n_outputs = 128
            if "BiLSTM" in hparams['backbone']:
                ifbidirectional = True
                self.n_outputs = 256
            else:
                ifbidirectional = False
            rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM,'BiLSTM': nn.LSTM}[rnn_type]
            self.rnn = rnn_cls(
                input_size=self.input_size,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=ifbidirectional
            )
        def forward(self, x):
            if self.hparams['NIC'] == "Atheros":
                x = x.view(-1,self.input_shape[0]*self.input_shape[1],self.input_shape[2])
                x = x.permute(0,2,1)
            elif self.hparams['NIC'] == "Intel":
                x = x.view(-1,self.input_shape[2],self.input_shape[0]*self.input_shape[1])
                x = x.permute(1,0,2)
            elif self.hparams['NIC'] == "BVP":
                x = x.view(-1,self.input_shape[0],self.input_shape[1]*self.input_shape[2])
                x = x.permute(1,0,2)
            else:
                raise ValueError('wrong type')
            out, _ = self.rnn(x)
            out = out[:, -1, :]  # 取最后一个时间步
            return out
    return _RNN(input_shape, hparams,rnn_type)

# =========================
# 参数表驱动配置
# =========================
LENET_CONFIG = {
    'Atheros': [
        (32, (15, 23), 9),
        (64, 3, (1, 3)),
        (96, (7, 3), (1, 3)),
    ],
    'Atheros_ARIL': [
        (32, (15, 23), 9),
        (64, 3, (1, 3)),
        (96, (3, 3), (1, 3)),
    ],
    'Intel': [
        (32, 7, (3, 1)),
        (64, (5, 4), (2, 2), (1, 0)),
        (96, (3, 3), 1),
    ],
    'BVP': [
        (32, 6, 2),
        (64, 3, 1),
        (96, 3, 1),
    ],
}

# =========================
# 通用Flatten+FC模块
# =========================
class FlattenFC(nn.Module):
    """
    通用flatten+全连接层模块。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# LeNet工厂
# =========================
def LeNet(input_shape, hparams):
    """
    LeNet工厂函数，根据hparams动态构建网络。
    """
    class _LeNet(nn.Module):
        def __init__(self, input_shape, hparams):
            super().__init__()
            self.inchannels = input_shape[0]
            self.n_outputs = 128
            # 选择配置
            if hparams['NIC'] == 'Atheros' and hparams.get('dataset') == 'ARIL':
                config = LENET_CONFIG['Atheros_ARIL']
            else:
                config = LENET_CONFIG[hparams['NIC']]
                layers = []
                in_c = self.inchannels
            for i, params in enumerate(config):
                out_c, kernel, stride = params[:3]
                padding = params[3] if len(params) > 3 else 0
                layers.append(nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding))
                layers.append(nn.ReLU(True))
                in_c = out_c
            self.encoder = nn.Sequential(*layers)
        def forward(self, x):
            x = self.encoder(x)
            a = x.shape[1] * x.shape[2] * x.shape[3]
            x = x.view(-1,a) #flatten
            fc = nn.Sequential(
                nn.Linear(a,128),
                nn.ReLU()).to(x.device)
            return fc(x)
    return _LeNet(input_shape, hparams)