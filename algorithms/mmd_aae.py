# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from widgbed.models import networks
from widgbed.algorithms.algorithms import Algorithm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def MMD_Loss_func(num_source, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]
    def loss(e_pred,d_ture):
        cost = 0.0
        for i in range(num_source):
            domain_i = e_pred[d_ture == i]
            for j in range(i+1,num_source):
                domain_j = e_pred[d_ture == j]
                single_res = mmd_two_distribution(domain_i,domain_j,sigmas=sigmas)
                cost += single_res
        return cost
    return loss

def mmd_two_distribution(source, target, sigmas):
    sigmas = torch.tensor(sigmas).cuda()
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    sigmas = sigmas.reshape(sigmas.shape + (1,))
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    dist = torch.zeros(x.size(0),y.size(0)).cuda()
    for i in range(x.size(0)):
        dist[i,:] = torch.sum(torch.square(x[i].expand(y.shape) - y),dim=1)
    return dist

class Encoder(nn.Module):
    def __init__(self,input_shape,hidden_layer = 2000):
        super(Encoder,self).__init__()
        self.fc = nn.Linear(input_shape,hidden_layer)
        # self.fc=nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_shape[0]*input_shape[1],128),
        #     nn.Linear(128,hidden_layer),
        # )

        return

    def forward(self,x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape, hidden_layer=2000):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_layer,input_shape)
        
        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Taskout(nn.Module):
    def __init__(self, n_class, hidden_layer=2000):
        super(Taskout, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, n_class)
        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x

class Adversarial(nn.Module):
    def __init__(self, hidden_layer=200):
        super(Adversarial, self).__init__()
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MA_model(nn.Module):
    def __init__(self, input_shape, nClass, hidden_layer):
        super(MA_model,self).__init__()
        self.input_shape = input_shape
        self.E = Encoder(input_shape = input_shape, hidden_layer=hidden_layer)
        self.D = Decoder(input_shape = input_shape, hidden_layer=hidden_layer)
        self.T = Taskout(n_class = nClass, hidden_layer=hidden_layer)
        return

    def forward(self,x):
        e = self.E(x)
        d = self.D(e)
        t = self.T(e)

        return e,d,t



class MMD_AAE(Algorithm):
    """https://github.com/mousecpn/MMD_AAE_PyTorch"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        hidden_layer=32#2000/1000的精度在CSIDA上奇高
        self.adv = Adversarial(hidden_layer).cuda()
        self.model = MA_model(self.featurizer.n_outputs, num_classes,hidden_layer).cuda()
        
        
        self.optimizer = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_adv = self.new_optimizer(self.adv.parameters())
        self.optimizer_model = self.new_optimizer(self.model.parameters())

        self.encoderLoss = MMD_Loss_func(3)
        self.taskLoss = nn.CrossEntropyLoss()
        self.decoderLoss = nn.MSELoss()
        # advLoss = nn.BCELoss()
        self.advLoss = nn.MSELoss()
        
                
    def update(self, minibatches, unlabeled=None):
        w_mmd = self.hparams["w_mmd"]#2
        w_adv = self.hparams["w_adv"]#0.1
        w_ae = self.hparams["w_ae"]#0.1
        w_cls = self.hparams["w_cls"]#1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )
        
        all_f=self.featurizer(all_x)
        
        e, d, t = self.model(all_f) #torch.Size([480, 2000]) #torch.Size([480, 512]) #torch.Size([480, 6])
        real = self.adv(e)
        
        t_loss = self.taskLoss(t,all_y)
        d_loss = self.decoderLoss(d,all_f)
        
        real_labels = torch.ones(all_x.shape[0],1).cuda()#torch.Size([480, 1])
        real = torch.square(real)#torch.Size([480, 1])
        adv_loss = self.advLoss(real, real_labels)
        mmd_loss = self.encoderLoss(e,all_d)
        
        total_loss = w_adv * adv_loss + w_cls * t_loss + w_ae * d_loss  + w_mmd * mmd_loss
        self.optimizer.zero_grad()
        self.optimizer_model.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.optimizer_model.step()
        
        loss_summary = {"t_loss": t_loss.item(),"d_loss":d_loss.item(),"adv_loss":adv_loss.item(),"mmd_loss":mmd_loss.item()}

        fake_e = np.random.laplace(0, 1, size=e.shape).astype('float32')
        fake_e  = torch.tensor(fake_e).cuda()
        real_labels = torch.ones(all_x.shape[0],1).cuda()
        fake_labels = torch.zeros(all_x.shape[0],1).cuda()
        all_data = torch.cat((e,fake_e),dim=0)
        all_labels = torch.cat((fake_labels,real_labels),dim=0)

        preds = self.adv(all_data.detach())
        adv_loss = self.advLoss(torch.square(preds), all_labels)
        
        self.optimizer.zero_grad()
        self.optimizer_adv.zero_grad()
        adv_loss.backward()
        self.optimizer.step()
        self.optimizer_adv.step()

        return loss_summary

    def predict(self, x):
        f=self.featurizer(x)      
        e, d, t = self.model(f) 
        return t