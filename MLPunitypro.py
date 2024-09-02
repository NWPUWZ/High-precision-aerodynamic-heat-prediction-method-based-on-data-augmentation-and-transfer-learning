# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:11:21 2021

@author: suowei
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import visdom
import numpy as np
import time
import sys


class ActFunc(nn.Module):
    def __init__(self, beta):
        super(ActFunc, self).__init__() 
        self.beta = beta
    
    def forward(self, x):
#        x = x * torch.sigmoid(self.beta * x)    
#        x = F.leaky_relu(x,self.beta)
#       x = torch.sin(x)
#        x = 0.5 * x * (1 + torch.tanh(0.79788*(x+0.044715*x**3)))
#        x = F.gelu(x)
#        x = torch.tanh(x)
#        x = x * (torch.tanh(F.softplus(x)))    
        x = F.elu(x, self.beta)
#        x = F.relu(x)
        return x


class Net(nn.Module):
    """
   The MLP network structure is generated according to the input parameters

   Initialization parameter
    -------------
    layer : List or tuple
            The length represents the number of layers, and the number represents the number of neurons in each layer
    bn :
        Specifies whether to add the Bn layer. By default, the BN layer is not added

    Examples
    -----------
    >>> model = Net([8, 10, 1], bn=False).net
    >>> Sequential(
            (fc1): Linear(in_features=8, out_features=10, bias=True)
            (relu1): LeakyReLU(negative_slope=0.01)
            (fc2): Linear(in_features=10, out_features=1, bias=True)
            )
    """
    def __init__(self, layer, bn=None, drop=False):
        super(Net, self).__init__()
        self.layer = layer
        self.bn = bn
        self.net = nn.Sequential()
        for i in range(len(self.layer)-1):
            self.net.add_module('fc{}'.format(i+1),
                                nn.Linear(self.layer[i], self.layer[i+1]))
            if i != (len(self.layer) - 2):
                if (bn):
                    self.net.add_module('bn{}'.format(i+1),
                                        nn.BatchNorm1d(self.layer[i+1]))
                if drop:
                    self.net.add_module('drop{}'.format(i+1),
                                        nn.Dropout(drop))
                self.net.add_module('actfunc{}'.format(i+1), 
                                    ActFunc(beta=0.02))
        p = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.register_parameter('p', p)
    def forward(self, x):
        x = self.net(x)
        return x
    

class DeepReZero(nn.Module):
    def __init__(self, input_range, width, depth, output_range):
        super(DeepReZero, self).__init__()
        self.linear_input = nn.Linear(input_range, width)
        self.linear_layers = nn.ModuleList([nn.Linear(width, width) 
                                            for i in range(depth)])
        self.linear_output = nn.Linear(width, output_range)
        self.actfunc = ActFunc(0.2)
        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        
    def forward(self, x_in):
        x = self.actfunc(self.linear_input(x_in))
        for i, j in enumerate(self.linear_layers):
            x = x + self.resweight[i] * self.actfunc(self.linear_layers[i](x))
        x = self.linear_output(x)
#        x = torch.sigmoid(x)
        return x


class TransNet(nn.Module):
    def __init__(self, layer1, layer2):
        super(TransNet, self).__init__()
        self.net1 = Net(layer1)
        self.net2 = Net(layer2)
        
    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x
        

        
class Nets(nn.Module):
    '''
    Two DeepReZero models are connected in series to realize simultaneous training of the series models
    '''
    def __init__(self, input_range1, width1, depth1, output_range1, 
                 input_range2, width2, depth2, output_range2):
        super(Nets, self).__init__()
        self.net1 = DeepReZero(input_range1, width1, depth1, output_range1)
        self.net2 = DeepReZero(input_range2, width2, depth2, output_range2)
        self.input_range1=input_range1
        self.input_range2=input_range2
        self.output_range1 = output_range1
    def forward(self, x):
        x1=x[:,0:self.input_range1]
        x2=x[:,self.input_range1:self.input_range1+self.input_range2-self.output_range1]
        mid = self.net1(x1)
        x_mid = torch.cat((x2, mid), dim = 1)
        x = self.net2(x_mid)
        return x, mid    
    
def weight_init(net, bias_constant=0):

    if isinstance(net, nn.Linear):
#        nn.init.kaiming_uniform_(net.weight, nonlinearity='leaky_relu')
#        nn.init.kaiming_normal_(net.weight, nonlinearity='leaky_relu')
        
#        nn.init.xavier_uniform_(net.weight, gain=nn.init.calculate_gain('leaky_relu', 0.02))
        nn.init.xavier_normal_(net.weight, gain=nn.init.calculate_gain('leaky_relu', 0.02))
        nn.init.constant_(net.bias, bias_constant)
    if isinstance(net, nn.BatchNorm1d):
        nn.init.constant_(net.weight, 1)
        nn.init.constant_(net.bias, 0)


class Trainer(object):


    def __init__(self, model=None, loss_func=None, optimizer=None,
                 trainset=None, valset=None, sch=None, vis=None,):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.trainset = trainset
        self.valset = valset
        self.sch = sch
        self.vis = vis
        self.weight = {}
        self.bias = {}
        for i in range(5):
            self.weight[i] = []
            self.bias[i] = []

    def run(self, input_range, output_range, epoch=1):

        self.input_range = input_range
        self.output_range = output_range

        if self.sch == 'Reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, verbose=2,
                    factor=0.5,
                    threshold=3e-6,
                    threshold_mode='rel',
                    mode='min',
                    cooldown=3,
                    min_lr=1e-5)    
        if self.sch == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=20,
                    T_mult=10
                    )

        self.train_losses = []
        self.val_losses = []
        t1 = time.time()
        self.max_val_loss = 100
        self.cont = 0    
        for i in range(epoch):
            self.train_loss = []
            self.val_loss = []
            loss1 = self.train(self.input_range, self.output_range)
            self.train_losses.append(loss1)
            loss2 = self.eval()
            self.val_losses.append(loss2)
            if self.sch:
                self.scheduler.step(self.val_losses[-1])
            if self.val_losses[-1] < self.max_val_loss:
                self.max_val_loss = self.val_losses[-1]
                torch.save(self.model, './model/best_model.pth')
                print('Save best model!')
            if(i>2):
                if(abs(self.val_losses[-1]-self.val_losses[-2])<1e-7):
                    self.cont = self.cont + 1
                else:
                    self.cont = 0
            if(self.cont==15):
                print('The verification error did not decrease for 5 consecutive times, and the training was terminated！')
                break
            if i % 1 == 0:
                print(f'epoch:{i}|{epoch} \t train_loss:{self.train_losses[-1]} \
              val_loss:{self.val_losses[-1]}')
        t2 = time.time()
        print(f'time : {t2-t1}')

    def train(self, input_range, output_range,):
        self.model.train()
        self.input_range = input_range
        self.output_range = output_range
        for data in self.trainset:
            batch_x = data[:, self.input_range]
            batch_y = data[:, self.output_range:]
            factor = data[:,-1]
            batch_pred = self.model(batch_x)
            loss = self.loss_func(inputx=batch_pred, target=batch_y, 
                                  model=self.model)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
           
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                           max_norm=100.0)
            self.optimizer.step()
            self.train_loss.append(loss.item())
        return np.array(self.train_loss).mean()

    def eval(self):
        self.model.eval()
        for data in self.valset:
            batch_x = data[:, self.input_range]
            batch_y = data[:, self.output_range:]
            batch_pred = self.model(batch_x)
            loss = self.loss_func(inputx=batch_pred, target=batch_y, 
                                  model=self.model,)
            self.val_loss.append(loss.item())
        return np.array(self.val_loss).mean()

