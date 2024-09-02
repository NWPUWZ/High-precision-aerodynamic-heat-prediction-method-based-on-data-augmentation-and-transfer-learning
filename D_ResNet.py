# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:40:35 2022

@author: wangze
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pandas as pd
from MLPunitypro import Nets, weight_init
from sklearn.model_selection import train_test_split
import os
from ranger import Ranger
#%%
class LossFunc(nn.Module):
    """
    自定义损失函数
    """
    def __init__(self):
        """ 初始化父类（nn.Module）属性 """
        super(LossFunc, self).__init__()
        """ 关联父类和子类 """
        self.lossfunc = nn.MSELoss()
    
    def forward(self, y, mid, pred, predmid):
        # 预测值相对误差(输出层)
        error_pre = torch.abs(y - pred)
        Lx1 = torch.where(error_pre < 0.001 * y, 0 * error_pre, error_pre/10)
#        Lx1 = (Lx1**2).mean()  
        Lx1 = Lx1.mean()
        # 数据之间的误差(输出层)
        Lx2 = self.lossfunc(pred, y)
       
        # 预测值相对误差(中间层)
        error_mid = torch.abs(mid - predmid)
        Lx3 = torch.where(error_mid < 0.001 * mid, 0 * error_mid, error_mid/10)
#        Lx3 = (Lx3**2).mean() 
        Lx3 = Lx3.mean()
        # 数据之间的误差(中间层)
        Lx4 = self.lossfunc(predmid, mid)
         # 防止输出数据小于0(输出层)
        index = (pred <= 0)
        y0 = pred[index]
        Lx5 = torch.abs(y0).mean()
        Lx5 = torch.where(Lx5 == Lx5 , Lx5, torch.tensor(0).float())
        loss = Lx2+ Lx4
        loss_item = torch.stack([Lx1,Lx2, Lx4, Lx3,Lx5])
#        print('Lx0: ', Lx0.item(), 'Lx1: ', Lx1.item(), 'Lx2: ', Lx2.item(),
#              'Lx3: ', Lx3.item(), 'Lx4: ', Lx4.item())
        return loss,loss_item
 #%%   
class Trainer():
    """
    训练模型的类
    """       
    def __init__(self, length, model, trainset, valset, batch, lr, L2):
        """
        初始化，读取数据，构建模型
        """
#        max_trainset = np.max(trainset[:, 6 : ], axis = 1)
#        trainset = trainset[np.argsort(max_trainset)]
        trainset = torch.tensor(trainset).float()
        valset = torch.tensor(valset).float()
        self.length = length
        self.model = model
#        self.sampler = Batch_Samper_Neigh(data_size = trainset.shape[0], 
#                                          batch_size = batch)
        self.trainloader = Data.DataLoader(dataset = trainset,
                                           batch_size = batch,
                                           shuffle = True)#将numpy数组变成张量，有这个作用
#        self.trainloader = Data.DataLoader(dataset = trainset,
#                                           batch_sampler = self.sampler)        
        self.valloader = Data.DataLoader(dataset = valset,
                                         batch_size = batch,
                                         shuffle = True)
#        self.optimizer = torch.optim.AdamW(self.model.parameters(),
#                                           lr = lr, weight_decay = L2)        
        self.optimizer = Ranger(self.model.parameters(),
                                lr = lr, weight_decay = L2)
        
        self.lossfunc = LossFunc()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, verbose = 2,
                    factor = 0.5,
                    threshold = 3e-6,
                    threshold_mode = 'rel',
                    mode = 'min',
                    cooldown = 3,
                    min_lr = 1e-5)    #最小化的学习率
    
    def run(self, epoch, inputrange):
        #epoch训练次数
        self.inputrange = inputrange
        self.train_losses = []
        self.val_losses = []
        t1 = time.time()
        max_val_loss = 100
        cont = 0    #记录误差不再下降的次数
        for i in range(epoch):
            loss1 = self.train(i)
            self.train_losses.append(loss1)
            loss2,val_loss_item = self.val()
            self.val_losses.append(loss2)
            self.scheduler.step(self.val_losses[-1])
            if self.val_losses[-1] < max_val_loss:
                max_val_loss = self.val_losses[-1]
                torch.save(self.model, './model_UTmulti/best_model.pth')
                print('save best model!')
            if (i > 2):
                if(abs(self.val_losses[-1] - self.val_losses[-2]) < 1e-7):
                    cont = cont + 1
                else:
                    cont = 0
            if(cont == 5):
                print('验证误差连续5次不再下降，终止训练！')
                break
            print(f'epoch:{i + 1}|{epoch} \t train_loss:{self.train_losses[-1]} \
                   val_loss:{self.val_losses[-1]}')
            print(val_loss_item)
        t2 = time.time()
        print(f'time : {t2-t1}')
        
    def train(self, i):
        self.model.train()
        train_loss = []
        for j, data in enumerate(self.trainloader):
            batch_x = data[:, 0 : self.inputrange+self.inputrange]           
            batch_y = data[:, self.inputrange+self.inputrange : ]#全部的输出列，包括中间层和最后一层
            batch_mid = batch_y[:, self.length :]#把全部的输出中的中间层输出部分找出
            batch_y = batch_y[:, 0 : self.length]#把全部的输出中的最后层输出部分找出
#            np.savetxt(f'./batchdata/modelice_TrainPro/train_{i}_{j}.dat', batch_y, 
#                       fmt = '%.08f')            
            batch_pred, batch_predmid = self.model(batch_x)
            #这一步是调用了model，根据主函数得知Trainer中的model是类Nets中的，类Nets返回的是两个值
            loss,loss_item = self.lossfunc(batch_y, batch_mid, batch_pred, batch_predmid)
            self.optimizer.zero_grad()#使前面的计算的梯度变为0
            loss.backward()#梯度反传
            
#            grad_weight = self.model.net2.linear_output.weight.grad.abs().mean()
##            print(self.model.net.fc5.weight.grad)
##            print(grad_weight)
#            for para in self.model.parameters(): 
#                para.grad = para.grad / (grad_weight + 1e-8)
                
            grad_weight = self.model.net2.linear_output.weight.grad.abs().max()#求最后输出层权重梯度绝对值的最大值
#            print(self.model.net.fc5.weight.grad)
#            print(grad_weight)
            for para in self.model.parameters(): 
                para.grad = para.grad / (grad_weight + 1e-8)#类似的一个参数归一化

            self.optimizer.step()
            train_loss.append(loss.item())#loss.item()中的.item将张量变为numpy数组
        return np.array(train_loss).mean()
   
    def val(self):
        self.model.eval()
        val_loss = []
        val_loss_item = []
        for data in self.valloader:
            batch_x = data[:, 0:self.inputrange+self.inputrange]
            batch_y = data[:, self.inputrange+self.inputrange:]
            batch_mid = batch_y[:, self.length:]
            batch_y = batch_y[:, 0 : self.length]
            batch_pred, batch_predmid = self.model(batch_x)
            loss,loss_item = self.lossfunc(batch_y, batch_mid, batch_pred, batch_predmid)
            val_loss.append(loss.item())#把tensor变成标量
            val_loss_item.append(loss_item.detach().numpy())
        return np.array(val_loss).mean(), np.array(val_loss_item).mean(0)

    def save(self, **params):
        """
        保存模型训练参数
        params : 字典文件，包含模型运行的必要参数        
        """
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        dir_name = '/{}_val={}/'.format(now, min(self.val_losses))
        dir_path = './model_UTmulti/' + dir_name
        os.makedirs(dir_path)
        torch.save(self.model, dir_path + 'model.pth')    # 保存模型本体
        # 保存模型超参数信息和训练信息
        with open(dir_path + 'model_info.dat', 'w') as f:
            f.write('LR = {} \n'.format(params['lr']))
            f.write('Batch = {} \n'.format(params['batch']))
            f.write('L2 = {} \n'.format(params['L2']))
            f.write('Init = {} \n'.format(params['init']))
            f.write('Inputrange = {} \n'.format(params['inputrange']))
            f.write('min train loss = {} \n'.format(min(self.train_losses)))
            f.write('min val loss = {} \n'.format(min(self.val_losses)))
            f.write('sca = {} \n'.format(params['sca']))
            f.write('actfunc = {} \n'.format(params['actfunc']))
            f.write(str(self.model))
        if(params['sca'] == 'mean-std'):
            np.savetxt(dir_path + 'sca_mean.dat', params['sca_mean'], fmt = '%.08f')
            np.savetxt(dir_path + 'sca_std.dat', params['sca_std'], fmt = '%.08f')
        else:
            np.savetxt(dir_path + 'sca_max.dat', params['sca_max'], fmt = '%.08f')
            np.savetxt(dir_path + 'sca_min.dat', params['sca_min'], fmt = '%.08f')
        loss = np.ones((len(self.train_losses), 3))
        loss[:, 0] = range(len(self.train_losses))
        loss[:, 1] = self.train_losses
        loss[:, 2] = self.val_losses
        np.savetxt(dir_path + 'loss.dat', loss,                    
                   header = 'Variables = epoch trainLoss valLoss',
                   comments = '', fmt = '%.08f')
        print('model has been saved!')            
    

""" 数据输入"""
#data_all=np.loadtxt('./data_file/data_all.dat')
data_train=np.loadtxt('./data_file/data_train.dat')
data_train, data_val = train_test_split(data_train, test_size=0.1)
#data_val=np.loadtxt('./data_file/data_val.dat')
data_pre=np.loadtxt('./data_file/data_pre2.dat')
""" 归一化处理"""
sca_max=np.max(data_train,0)
sca_min=np.min(data_train,0)
#sca_max[5:9]=1
#sca_min[5:9]=0
#sca_max[21]=1
#sca_min[21]=0
#sca_mean = np.mean(data_all, axis = 0)
#sca_std = np.std(data_all, axis = 0)
#sca_con = (trainset - sca_mean) / sca_std
trainset=(data_train-sca_min)/(sca_max-sca_min)
valset=(data_val-sca_min)/(sca_max-sca_min)
preset=(data_pre-sca_min)/(sca_max-sca_min)
"""保存归一化参数"""
np.savetxt('./data_file/sca_min.dat', sca_min, fmt = '%.18f')
np.savetxt('./data_file/sca_max.dat', sca_max, fmt = '%.18f')
np.savetxt
"""训练模型"""
length = 1        #最后输出特征数
width = 1           #中间输出层的的特征数
batch = 64#训练时递入的一批数据是32个
lr = 1e-3
L2 = 1e-3
epoch = 400
inputrange = 11
model = Nets(inputrange, 64, 6, width, width + inputrange, 64, 6, length)
model.apply(weight_init) 
trainer = Trainer(length = length, model = model, trainset = trainset, valset = valset,
                  batch = batch, lr = lr , L2 = L2)
trainer.run(epoch, inputrange)
"""保存模型"""
params = {'lr': lr, 'batch': batch, 'L2': L2, 'epoch': epoch,
          'inputrange': inputrange, 'init': 'xavier_uniform',
          'actfunc': 'sin', 'sca': 'min-max', 'sca_min': sca_min,
          'sca_max': sca_max}
trainer.save(**params)
"""画图"""
trainloss, valloss = trainer.train_losses, trainer.val_losses
plt.figure(1)
plt.plot(trainloss, label='train loss')
plt.plot(valloss, label='val loss')
plt.legend()
plt.yscale('log')
plt.show()    

