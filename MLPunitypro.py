# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:11:21 2021

@author: Wangze
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
#        x = x * torch.sigmoid(self.beta * x)    # swish激活函数
#        x = F.leaky_relu(x,self.beta)
#       x = torch.sin(x)
#        x = 0.5 * x * (1 + torch.tanh(0.79788*(x+0.044715*x**3)))
#        x = F.gelu(x)
#        x = torch.tanh(x)
#        x = x * (torch.tanh(F.softplus(x)))    # Mish激活函数
        x = F.elu(x, self.beta)
#        x = F.relu(x)
        return x


class Net(nn.Module):
    """
    根据输入的参数生成对应的MLP网络结构

    初始化参数
    -------------
    layer : 列表或者元组
            MLP的结构参数，长度表示层数，数字表示每一层的神经元个数
    bn :
        是否添加Bn层，默认不添加

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
        
#class Nets(nn.Module):
#    '''
#    将两个DeepReZero模型串联起来, 实现串联模型同时训练
#    增加BatchNorm1d层
#    '''
#    def __init__(self, input_range1, width1, depth1, output_range1, 
#                 input_range2, width2, depth2, output_range2):
#        super(Nets, self).__init__()
#        self.net1 = DeepReZero(input_range1, width1, depth1, output_range1)
#        self.net2 = DeepReZero(input_range2, width2, depth2, output_range2)
#        self.bn = nn.BatchNorm1d(output_range1)
#        
#    def forward(self, x):
#        mid = self.net1(x)
#        midBn = self.bn(mid)
#        x_midBn = torch.cat((x, midBn), dim =1)
#        x = self.net2(x_midBn)
#        return x, mid
        
class Nets(nn.Module):
    '''
    将两个DeepReZero模型串联起来, 实现串联模型同时训练
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
    """
    模型参数初始化，采用kaiming初始化

    Parameters
    -------------
    bias_constant : float
            每一层bias初始化时的常数，默认为0
    """
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
    """
    模型训练类

    Parameters
    ---------
    - model : 模型
    - loss_func : 损失函数
    - optimizer : 优化器
    - trainset :训练数据
    - valet :  验证数据
    - sch : 是否学习率调整
    - vis : 是否可视化
    """

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
        """
        模型训练与验证函数

        参数
        -------
        epoch : int
                训练次数
        input_range : slice
                数据集中输入特征的起止范围
        output_range : int
                数据集中标签的开始列数

        返回
        -------
        None

        Examples
        ---------
        >>> input_range = slice(0, 5)
        >>> 模型的输入特征在trainset中的范围为trainset[:, 0:5]

        >>> output_range = slice(-2, -1)
        >>> 标签的范围为trainset[:, -2：-1]
        """
        # 每一个epoch,就是一次train的过程
        # 训练可视化
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
                    min_lr=1e-5)    #最小化的学习率
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
        self.cont = 0    #记录误差不再下降的次数
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
                print('验证误差连续5次不再下降，终止训练！')
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
            #加入梯度裁剪(gradient clipping)防止梯度爆炸问题
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

