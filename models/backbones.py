# Sensefi_model.py
# 主要功能：定义SenseFi相关的主干网络（ResNet、RNN、GRU、LSTM、ViT等）。
# 支持多种输入形状和参数配置，适配不同的CSI/无线信号数据。

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def MLP_Feature(input_shape, hparams):
    """
    MLP_Feature工厂函数，根据hparams动态构建网络。
    """
    class _MLP_Feature(nn.Module):
        def __init__(self, input_shape, hparams):
            super().__init__()

            self.inchannels = input_shape[0] * input_shape[1] * input_shape[2]
            self.n_outputs = 128
            
            self.layer_norm = nn.BatchNorm1d(self.inchannels)
            
            self.layer_0 = torch.nn.Linear(self.inchannels, 256)
            self.layer_1 = torch.nn.Linear(256, 128)
            #
            self.layer_relu = torch.nn.ReLU()
            self.layer_dropout = torch.nn.Dropout(0.1)
            #
            torch.nn.init.xavier_uniform_(self.layer_0.weight)
            torch.nn.init.xavier_uniform_(self.layer_1.weight)

        # self.classifier = nn.Linear(128,num_classes)
        def forward(self,x):
            
            x = x.view(-1,self.inchannels) #(n,3*114*500)
            x = self.layer_norm(x)
            #
            x = self.layer_0(x)
            x = self.layer_relu(x)
            x = self.layer_dropout(x)
            #
            x = self.layer_1(x)
            x = self.layer_relu(x)
            out = self.layer_dropout(x)

            return out
    return _MLP_Feature(input_shape, hparams)


def CNN1D(input_shape, hparams):
    """
    CNN1D工厂函数，根据hparams动态构建网络。
    """
    class _CNN1D(nn.Module):
        def __init__(self, input_shape, hparams):
            super().__init__()
            self.inchannels = input_shape[0]*input_shape[1]
            self.n_outputs = 512
            # 选择配置
            self.layer_norm = nn.BatchNorm1d(self.inchannels)
            self.layer_relu = nn.ReLU()
            self.layer_dropout = nn.Dropout(0.2)
            self.layer_cnn_1d_0 = nn.Conv1d(self.inchannels, 128, kernel_size=29, stride=13)
            self.layer_cnn_1d_1 = nn.Conv1d(128, 256, kernel_size=15, stride=7)
            self.layer_cnn_1d_2 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
            
            nn.init.xavier_uniform_(self.layer_cnn_1d_0.weight)
            nn.init.xavier_uniform_(self.layer_cnn_1d_1.weight)
            nn.init.xavier_uniform_(self.layer_cnn_1d_2.weight)
        
        def forward(self, x):
            x=x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3])
            x = self.layer_norm(x)
            #
            x = self.layer_cnn_1d_0(x)
            x = self.layer_relu(x)
            x = self.layer_dropout(x)

            x = self.layer_cnn_1d_1(x)
            x = self.layer_relu(x)
            x = self.layer_dropout(x)

            x = self.layer_cnn_1d_2(x)
            x = self.layer_relu(x)
            x = self.layer_dropout(x)

            x = torch.mean(x, dim = -1)
            
            out = self.layer_dropout(x)

            return out
    return _CNN1D(input_shape, hparams)

def CNN2D(input_shape, hparams):
    """
    CNN2D工厂函数，根据hparams动态构建网络。
    """
    class _CNN2D(nn.Module):
        def __init__(self, input_shape, hparams):
            super().__init__()
            self.inchannels = input_shape[0]
            self.n_outputs = 128
            # 选择配置
            self.layer_norm_0 = nn.BatchNorm2d(self.inchannels)
            self.layer_norm_1 = nn.BatchNorm2d(32)
            self.layer_norm_2 = nn.BatchNorm2d(64)
            self.layer_norm_3 = nn.BatchNorm2d(128)
            
            self.layer_leakyrelu = nn.LeakyReLU()
            self.layer_dropout = nn.Dropout(0.2)
            
            self.layer_cnn_2d_0 = nn.Conv2d(self.inchannels, 32, kernel_size=(27, 27), stride=(5, 5))
            self.layer_cnn_2d_1 = nn.Conv2d(32, 64, kernel_size=(15, 15), stride=(3, 3))
            self.layer_cnn_2d_2 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(1, 1))
            
            nn.init.xavier_uniform_(self.layer_cnn_2d_0.weight)
            nn.init.xavier_uniform_(self.layer_cnn_2d_1.weight)
            nn.init.xavier_uniform_(self.layer_cnn_2d_2.weight)
        
        def forward(self, x):
            x = self.layer_norm_0(x)
            #
            x = self.layer_cnn_2d_0(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_dropout(x)

            x = self.layer_norm_1(x)
            x = self.layer_cnn_2d_1(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_dropout(x)

            x = self.layer_norm_2(x)
            x = self.layer_cnn_2d_2(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_dropout(x)

            x = self.layer_norm_3(x)

            out = torch.mean(x, dim = (-2, -1))

            return out
    return _CNN2D(input_shape, hparams)

def CNNLSTM(input_shape, hparams):
    """
    CNN_LSTM工厂函数，根据hparams动态构建网络。
    """
    class _CNNLSTM(nn.Module):
        def __init__(self, input_shape, hparams):
            super().__init__()
            self.input_shape=input_shape
            self.inchannels = input_shape[0]*input_shape[1]
            self.n_outputs = 512
            # 选择配置
            self.layer_norm = nn.BatchNorm1d(self.inchannels)
            self.layer_norm_0 = nn.BatchNorm1d(64)
            self.layer_norm_1 = nn.BatchNorm1d(128)
            self.layer_norm_2 = nn.BatchNorm1d(256)
            
            self.layer_leakyrelu = nn.LeakyReLU()
            self.layer_dropout = nn.Dropout(0.5)
            
            self.layer_cnn_1d_0 = nn.Conv1d(self.inchannels, 64, kernel_size=128, stride=8)
            self.layer_cnn_1d_1 = nn.Conv1d(64, 128, kernel_size=64, stride=4)
            self.layer_cnn_1d_2 = nn.Conv1d(128, 256, kernel_size=32, stride=2)
            self.layer_lstm = nn.LSTM(input_size = 256,hidden_size = 512, batch_first = True)
            
            nn.init.xavier_uniform_(self.layer_cnn_1d_0.weight)
            nn.init.xavier_uniform_(self.layer_cnn_1d_1.weight)
            nn.init.xavier_uniform_(self.layer_cnn_1d_2.weight)
        
        def forward(self, x):
            x = x.view(-1,self.input_shape[0]*self.input_shape[1],self.input_shape[2])
            x = self.layer_norm(x)
            #
            x = self.layer_cnn_1d_0(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_norm_0(x)

            x = self.layer_cnn_1d_1(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_norm_1(x)

            x = self.layer_cnn_1d_2(x)
            x = self.layer_leakyrelu(x)
            x = self.layer_norm_2(x)

            x = torch.permute(x, (0, 2, 1)) 

            out, _ = self.layer_lstm(x)
            out = out[:, -1, :]
            out = self.layer_dropout(out)

            return out
    return _CNNLSTM(input_shape, hparams)


def RNN_Factory2(input_shape, hparams, rnn_type='RNN'):
    """
    RNN/GRU/LSTM/BiLSTM工厂函数，根据hparams动态构建网络。
    """
    class _RNN2(nn.Module):
        def __init__(self, input_shape, hparams,rnn_type):
            super().__init__()
            self.input_shape = input_shape
            self.hparams = hparams
            self.input_size = self.input_shape[0]*self.input_shape[1]
            self.seq_len = input_shape[-1]
            self.n_outputs = 512
            
            if hparams['backbone'] in ["BiLSTM2", "ABLSTM"]:
                ifbidirectional = True
                self.n_outputs = self.n_outputs*2
            else:
                ifbidirectional = False
            if hparams['backbone'] == "ABLSTM":
                self.layer_linear = nn.Linear(2*512, 2*512)
                self.layer_activation = nn.LeakyReLU()
                self.layer_softmax = nn.Softmax(dim = -2)
                self.layer_pooling = nn.AvgPool1d(8, 8)
                self.layer_norm = nn.BatchNorm1d(self.input_size)
                self.layer_dropout = nn.Dropout(0.6)
                nn.init.xavier_uniform_(self.layer_linear.weight)
            else:
                self.layer_norm = nn.BatchNorm1d(self.input_size)
                self.layer_pooling = nn.AvgPool1d(10, 10)
            rnn_cls = {'RNN2': nn.RNN, 'GRU2': nn.GRU, 'LSTM2': nn.LSTM,'BiLSTM2': nn.LSTM,'ABLSTM': nn.LSTM }[rnn_type]
            self.rnn = rnn_cls(
                input_size=self.input_size,
                hidden_size=512,
                #num_layers=2,
                batch_first=True,
                bidirectional=ifbidirectional
            )
        def forward(self, x):
            x = x.view(-1,self.input_shape[0]*self.input_shape[1],self.input_shape[2])
            x = self.layer_norm(x)
            x = self.layer_pooling(x)
            x = x.permute(0,2,1)
            out, _ = self.rnn(x)
            if hparams['backbone'] == "ABLSTM":
                s = self.layer_linear(out)
                s = self.layer_activation(s)
                a = self.layer_softmax(s)
                t = out * a
                t = torch.sum(t, dim = -2)
                out = self.layer_dropout(t)
            else:
                out = out[:, -1, :]  # 取最后一个时间步
            return out
    return _RNN2(input_shape, hparams,rnn_type)  

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 9, patch_size_h = 25, emb_size = 9*25, img_size = 342*500, inputshape=None, hparams=None):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        self.inputshape = inputshape
        self.hparams=hparams
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
    
    def forward(self, x):
        if self.hparams['NIC'] in ["Atheros", "BVP"]:
            x = x.view(-1,1,self.inputshape[0]*self.inputshape[1],self.inputshape[2])
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 225, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 225,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
        

        

class NTU_Fi_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 9,
                patch_size_h = 25,
                emb_size = 225,
                img_size = 342*500,
                depth = 1,
                *,
                num_classes,
                inputshape,
                hparams,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size,inputshape, hparams),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )
        self.n_outputs = num_classes
        
def ViT(input_shape,hparams):
        
        n_outputs = 128
        
        in_channels = 1
        if hparams['NIC'] == "Atheros":
            patch_size_w = 9
            patch_size_h = 25
            emb_size = 225
        elif hparams['NIC'] == "Intel":
            patch_size_w = 50
            patch_size_h = 18
            emb_size = 900
        elif hparams['NIC'] == "BVP":
            patch_size_w = 2
            patch_size_h = 40
            emb_size = 80
        else:
            raise ValueError('wrong type')
        img_size = input_shape[0] * input_shape[1] * input_shape[2]  # Assuming input_shape is (3, 114, 500)
        depth = 1 
        return NTU_Fi_ViT(
            in_channels=in_channels,
            patch_size_w=patch_size_w,
            patch_size_h=patch_size_h,
            emb_size=emb_size,
            img_size=img_size,
            depth=depth,
            num_classes=n_outputs,
            inputshape=input_shape,
            hparams=hparams
        )
        
