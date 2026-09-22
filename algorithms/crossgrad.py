# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from widgbed.models import networks
from widgbed.algorithms.algorithms import Algorithm

#单域跟CrossGrad一样
class CrossGrad_mul(Algorithm):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        device = "cuda" 
        self.hparams=hparams
        self.eps_f = hparams["eps_f"]
        self.eps_d = hparams["eps_d"]
        self.alpha_f = hparams["alpha_f"]
        self.alpha_d = hparams["alpha_d"]

        print("Building F")
        self.featurizer_f = networks.CSIResNet(input_shape,if_classfier=False)
        self.classifier_f = networks.cg_classfier(self.featurizer_f.n_outputs,num_classes).cuda()
        self.network_f = nn.Sequential(self.featurizer_f, self.classifier_f)

        self.optimizer_f = self.new_optimizer(
            self.network_f.parameters(),
        )
        # self.scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_f , T_max=self.hparams["max_epoch"]) # Initialize scheduler


        print("Building D")
        self.featurizer_d = networks.CSIResNet(input_shape,if_classfier=False)
        self.optimizer_d = self.new_optimizer(
                self.featurizer_d.parameters(),
            )
        # self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d , T_max=self.hparams["max_epoch"]) # Initialize scheduler
        
        self.classifier_dc=[]
        self.optimizer_dc=[]
        # self.scheduler_dc=[]
        for n_subdomain in hparams['n_subdomains'].values(): 
            classifier_d = networks.cg_classfier(self.featurizer_f.n_outputs, n_subdomain).cuda()
            self.classifier_dc.append(classifier_d)

            optimizer_c = self.new_optimizer(
                classifier_d.parameters(),
            )
            self.optimizer_dc.append(optimizer_c)

            # scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c , T_max=self.hparams["max_epoch"]) # Initialize scheduler
            # self.scheduler_dc.append(scheduler_c)
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, d in minibatches])
        all_y = torch.cat([y for x, y, d in minibatches])
        all_domain = torch.cat([d for x, y, d in minibatches])

        all_d=[]
        for i in range(all_domain.shape[1]):
            d_label=copy.deepcopy(all_domain[:,i])
            u_label=torch.unique(d_label,sorted=True)
            for j in range(len(u_label)):
                d_label[d_label == u_label[j]]=j
            all_d.append(d_label)


        all_n_domains=1
        for n_subdomain in self.hparams['n_subdomains'].values(): 
            all_n_domains*=n_subdomain
        if all_n_domains != len(minibatches):
            raise ValueError('wrong')
        
        
        all_x.requires_grad = True

        # Compute domain perturbation
        input_ds=[]
        for i in range(len(all_d)):
            loss_d = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(all_x)), all_d[i])
            loss_d.backward()
            grad_d = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
            input_d = all_x.data + self.eps_f * grad_d
            input_ds.append(input_d)

        # Compute label perturbation
        all_x.grad.data.zero_()
        loss_f = F.cross_entropy(self.network_f(all_x), all_y)
        loss_f.backward()
        grad_f = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
        input_f = all_x.data + self.eps_d * grad_f

        all_x = all_x.detach()

        # Update label net
        loss_f1 = F.cross_entropy(self.network_f(all_x), all_y)
        loss_f = (1 - self.alpha_f) * loss_f1
        for i in range(len(all_d)):
            loss_f2 = F.cross_entropy(self.network_f(input_ds[i]), all_y)
            loss_f+= self.alpha_f * loss_f2
        self.optimizer_f.zero_grad()
        loss_f.backward()
        self.optimizer_f.step()

        loss_summary = {"loss_f": loss_f.item()}

        # Update domain net
        for i in range(len(all_d)):
            loss_d1 = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(all_x)), all_d[i])
            loss_d2 = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(input_f)), all_d[i])
            loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
            self.optimizer_d.zero_grad()
            self.optimizer_dc[i].zero_grad()
            loss_d.backward()
            self.optimizer_d.step()
            self.optimizer_dc[i].step()
            loss_summary["loss_d"+str(i)]=loss_d.item()

        return loss_summary

    def predict(self, x):
        return self.network_f(x)

#单域跟CrossGrad不一样
class CrossGrad_widg(Algorithm):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        device = "cuda" 
        self.hparams=hparams
        self.eps_f = hparams["eps_f"]
        self.eps_d = hparams["eps_d"]
        self.alpha_f = hparams["alpha_f"]
        self.alpha_d = hparams["alpha_d"]

        print("Building F")
        self.featurizer_f = networks.CSIResNet(input_shape,if_classfier=False)
        self.classifier_f = networks.cg_classfier(self.featurizer_f.n_outputs,num_classes).cuda()
        self.network_f = nn.Sequential(self.featurizer_f, self.classifier_f)

        self.optimizer_f = self.new_optimizer(
            self.network_f.parameters(),
        )
        # self.scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_f , T_max=self.hparams["max_epoch"]) # Initialize scheduler


        print("Building D")
        self.featurizer_d = networks.CSIResNet(input_shape,if_classfier=False)
        self.optimizer_d =self.new_optimizer(
                self.featurizer_d.parameters(),
            )
        # self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d , T_max=self.hparams["max_epoch"]) # Initialize scheduler
        
        self.classifier_dc=[]
        self.optimizer_dc=[]
        # self.scheduler_dc=[]
        for n_subdomain in hparams['n_subdomains'].values(): 
            classifier_d = networks.cg_classfier(self.featurizer_f.n_outputs, n_subdomain).cuda() 
            self.classifier_dc.append(classifier_d)

            optimizer_c = self.new_optimizer(
                classifier_d.parameters(),
            )
            self.optimizer_dc.append(optimizer_c)

            # scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c , T_max=self.hparams["max_epoch"]) # Initialize scheduler
            # self.scheduler_dc.append(scheduler_c)
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, d in minibatches])
        all_y = torch.cat([y for x, y, d in minibatches])
        all_domain = torch.cat([d for x, y, d in minibatches])

        all_d=[]
        for i in range(all_domain.shape[1]):
            d_label=copy.deepcopy(all_domain[:,i])
            u_label=torch.unique(d_label,sorted=True)
            for j in range(len(u_label)):
                d_label[d_label == u_label[j]]=j
            all_d.append(d_label)


        all_n_domains=1
        for n_subdomain in self.hparams['n_subdomains'].values(): 
            all_n_domains*=n_subdomain
        # if all_n_domains != len(minibatches):
        #     raise ValueError('wrong')
        
        
        all_x.requires_grad = True

        # Compute domain perturbation
        input_d = all_x.data
        for i in range(len(all_d)):
            loss_d = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(all_x)), all_d[i])
            loss_d.backward()
            grad_d = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
            input_d += self.eps_f * grad_d

        # Compute label perturbation
        all_x.grad.data.zero_()
        loss_f = F.cross_entropy(self.network_f(all_x), all_y)
        loss_f.backward()
        grad_f = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
        input_f = all_x.data + self.eps_d * grad_f

        all_x = all_x.detach()

        # Update label net
        loss_f1 = F.cross_entropy(self.network_f(all_x), all_y)
        loss_f2 = F.cross_entropy(self.network_f(input_d), all_y)
        loss_f=(1 - self.alpha_f) * loss_f1 + self.alpha_f * loss_f2
        self.optimizer_f.zero_grad()
        loss_f.backward()
        self.optimizer_f.step()

        loss_summary = {"loss_f": loss_f.item()}

        # Update domain net
        for i in range(len(all_d)):
            loss_d1 = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(all_x)), all_d[i])
            loss_d2 = F.cross_entropy(self.classifier_dc[i](self.featurizer_d(input_f)), all_d[i])
            loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
            self.optimizer_d.zero_grad()
            self.optimizer_dc[i].zero_grad()
            loss_d.backward()
            self.optimizer_d.step()
            self.optimizer_dc[i].step()
            loss_summary["loss_d"+str(i)]=loss_d.item()

        return loss_summary

    def predict(self, x):
        return self.network_f(x)
