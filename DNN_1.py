# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:07:42 2021

@author: sun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as Data
import time
import os
import matplotlib.pyplot as plt
import xlwt
k_RANS_all=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
num=np.arange(0,15,1)

#num=np.arange(0,1,1)
r_mse_all=[]
for num in num :
# %%
    class ActFunc(nn.Module):
        def __init__(self, beta):
            """
            Custom activation functions
            
            """
            super(ActFunc, self).__init__()
            self.beta = beta
    
        def forward(self, x):
    #        x = x * torch.sigmoid(self.beta * x)    
    #        x = F.leaky_relu(x, self.beta)
    #        x = torch.sin(x)
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
                    if bn:
                        self.net.add_module('bn{}'.format(i+1),
                                            nn.BatchNorm1d(self.layer[i+1]))
                    if drop:
                        self.net.add_module('drop{}'.format(i+1),
                                            nn.Dropout(drop))
                    self.net.add_module('actfunc{}'.format(i+1),
                                        ActFunc(beta=0.02))
    
        def forward(self, x):
            x = self.net(x)
            return x
    
    
    class DeepReZero(nn.Module):
        def __init__(self, input_num, width, depth, output_num):
            """
            This class is a residual neural network and is not needed when using a fully connected neural network
            """
            super(DeepReZero, self).__init__()
            self.linear_input = nn.Linear(input_num, width)
            self.linear_layers = nn.ModuleList([nn.Linear(width, width)
                                                for i in range(depth)])
            self.linear_output = nn.Linear(width, output_num)
            self.actfunc = ActFunc(0.02)
            self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
    
        def forward(self, x_in):
            x = self.actfunc(self.linear_input(x_in))
            for i, j in enumerate(self.linear_layers):
                x = x + self.resweight[i] * self.actfunc(self.linear_layers[i](x))
            x = self.linear_output(x)
            return x
    
    
    def weight_init(net, bias_constant=0):

        if isinstance(net, nn.Linear):
    #        nn.init.kaiming_uniform_(net.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(net.weight, nonlinearity='leaky_relu')
    
    #        nn.init.xavier_uniform_(net.weight, gain=nn.init.calculate_gain('leaky_relu', 0.02))
    #        nn.init.xavier_normal_(net.weight, gain=nn.init.calculate_gain('leaky_relu', 0.02))
            nn.init.constant_(net.bias, bias_constant)
        if isinstance(net, nn.BatchNorm1d):
            nn.init.constant_(net.weight, 1)
            nn.init.constant_(net.bias, 0)
    
    
    
    class LossFunc(nn.Module):
        """
        Custom loss function
        """
        def __init__(self):
            super(LossFunc, self).__init__()
            self.lossfunc = nn.MSELoss()
    
        def forward(self, y, pred):
            
            Lx0 = (torch.abs(y) - y) / 2.0
            Lx0 = (Lx0**2).mean()
            
            error = torch.abs(y - pred)
            Lx1 = torch.where(error < 0.01 * y, 0 * error, error)
            Lx1 = (Lx1**2).mean()
            
            Lx2 = self.lossfunc(torch.abs(pred), y)
            # loss = Lx0 + Lx1 + Lx2
            loss = Lx2
            return loss
    
    
    class Trainer():
        """
        Classes for training models
        """
        def __init__(self, model, batch,lr, L2, trainset,valset):
            """
            Initialize, read the data, build the model
            """
            trainset = torch.tensor(trainset).float()
            valset = torch.tensor(valset).float()
            
            self.model = model
            self.trainloader = Data.DataLoader(dataset=trainset,
                                               batch_size=batch,
                                               shuffle=True)
            self.valloader = Data.DataLoader(dataset=valset,
                                             batch_size=batch,
                                             shuffle=True)

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=lr, weight_decay=L2)

            self.lossfunc = LossFunc()
    
        def run(self, epoch, inputrange):
            self.inputrange = inputrange
            self.train_losses = []
            self.val_losses = []
            t1 = time.time()
            max_val_loss = 100
            cont = 0    

            for i in range(epoch):
                loss1 = self.train()
                self.train_losses.append(loss1)
                loss2 = self.val()
                self.val_losses.append(loss2)
                if self.val_losses[-1] < max_val_loss:
                    max_val_loss = self.val_losses[-1]
                    torch.save(self.model, './model/best_model.pth')
                    print('Save best model!')
                if(i > 2):
                    if(abs(self.val_losses[-1] - self.val_losses[-2]) < 1e-7):
                        self.cont = self.cont + 1
                    else:
                        self.cont = 0
                if(cont == 5):
                    print('The verification error did not decrease for 5 consecutive times, and the training was terminated！')
                    break
                print(f'epoch:{i}|{epoch} \t train_loss:{self.train_losses[-1]} \
                       val_loss:{self.val_losses[-1]}')
            t2 = time.time()
            print(f'time : {t2-t1}')
    
        def train(self):

            self.model.train()
            train_loss = []
            for data in self.trainloader:
                batch_x = data[:, 0:self.inputrange]
                batch_y = data[:, self.inputrange:]
                batch_pred = self.model(batch_x)
                index_RANS=(batch_y[:,1]!=0)
                index_EXP=(batch_y[:,1]==0)
                batch_RANS=batch_y[:,[0]][index_RANS]
                batch_EXP=batch_y[:,[0]][index_EXP]
                batch_pred_RANS=batch_pred[index_RANS]
                batch_pred_EXP=batch_pred[index_EXP]
                #if (index_RANS.sum() == 0):
                    #batch_RANS=torch.tensor([0])
                    #batch_pred_RANS=torch.tensor([0])
                #if (index_EXP.sum() == 0):
                    #batch_EXP=torch.tensor([0])
                    #batch_pred_EXP=torch.tensor([0])
                loss_RANS = self.lossfunc(batch_RANS, batch_pred_RANS)
                loss_EXP = self.lossfunc(batch_EXP, batch_pred_EXP)
                if (loss_RANS!=loss_RANS):
                    loss_RANS=torch.tensor(0)
                if (loss_EXP!=loss_EXP):
                    loss_EXP=torch.tensor(0)
                loss=k_RANS_all[num]*loss_RANS+1*loss_EXP
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            return np.array(train_loss).mean()
    
        def val(self):
            """
            How the model behaves on a validation set
            """
            self.model.eval()
            val_loss = []
            loss_RANS_out = []
            loss_EXP_out = [] 
            for data in self.valloader:
                batch_x = data[:, 0:self.inputrange]
                batch_y = data[:, self.inputrange:]
                batch_pred = self.model(batch_x)
                index_RANS=(batch_y[:,1]!=0)
                index_EXP=(batch_y[:,1]==0)
                batch_RANS=batch_y[:,[0]][index_RANS]
                batch_EXP=batch_y[:,[0]][index_EXP]
                batch_pred_RANS=batch_pred[index_RANS]
                batch_pred_EXP=batch_pred[index_EXP]
                loss_RANS = self.lossfunc(batch_RANS, batch_pred_RANS)
                loss_EXP = self.lossfunc(batch_EXP, batch_pred_EXP)
                if (loss_RANS!=loss_RANS):
                    loss_RANS=torch.tensor(0)
                if (loss_EXP!=loss_EXP):
                    loss_EXP=torch.tensor(0)
                loss=k_RANS_all[num]*loss_RANS+1*loss_EXP
                #loss = self.lossfunc(batch_y, batch_pred)
                val_loss.append(loss.item())
                loss_RANS_out.append(loss_RANS.item())
                loss_EXP_out.append(loss_EXP.item())
            print(np.array(loss_RANS_out).mean())
            print(np.array(loss_EXP_out).mean())
            return np.array(val_loss).mean()
    
        def save(self, **params):
            """
            Save the model training parameters 
        params: dictionary file that contains the necessary parameters for the model to run 
            """
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            dir_path = r"./model/{}_val={}/".format(now, min(self.val_losses))
            os.makedirs(dir_path)
            self.model = torch.load('./model/best_model.pth')
            torch.save(self.model, dir_path + '/model.pth')
            torch.save(self.model,  './model/models/model'+str(num+1)+'.pth')
           """ Save model hyperparameter information and training information"""
            with open(dir_path+'model_info.dat', 'w') as f:
                f.write('LR = {} \n'.format(params['lr']))
                f.write('Batch = {} \n'.format(params['batch']))
                f.write('L2 = {} \n'.format(params['L2']))
                f.write('Inputrange = {} \n'.format(params['inputrange']))
                f.write('min train loss = {} \n'.format(min(self.train_losses)))
                f.write('min val loss = {} \n'.format(min(self.val_losses)))
                f.write('sca = {} \n'.format(params['sca']))
                f.write('actfunc = {} \n'.format(params['actfunc']))
                f.write(str(self.model))
            if(params['sca'] == 'mean-std'):
                np.savetxt(dir_path + 'sca_mean.dat', params['sca_mean'])
                np.savetxt(dir_path + 'sca_std.dat', params['sca_std'])
            else:
                np.savetxt(dir_path + 'sca_max.dat', params['sca_max'])
                np.savetxt(dir_path + 'sca_min.dat', params['sca_min'])
            loss = np.ones((len(self.train_losses), 3))
            loss[:, 0] = range(len(self.train_losses))
            loss[:, 1] = self.train_losses
            loss[:, 2] = self.val_losses
            
            np.savetxt(dir_path + 'loss.dat', loss)
            print('model has been saved!')
    
    if __name__ == '__main__':
        # %%
        """  Prepare to read training data. Assume that the data is saved as train.dat. Prepare the data in advance"""
        train_Data = np.loadtxt('./train.dat')
        test_Data = np.loadtxt('./test.dat')
        #train_Data[:,3:6]=np.log10(train_Data[:,3:6])
       # train_Data[:,10]=np.log10(train_Data[:,10])
       """ Normalized processing"""
    
        #sca_max = np.loadtxt('./sca_max.dat')
        #sca_min = np.loadtxt('./sca_min.dat')
        sca_max = np.max(train_Data,0)
        sca_min = np.min(train_Data,0)
        np.savetxt('./sca_min.dat', sca_min, fmt = '%.18f')
        np.savetxt('./sca_max.dat', sca_max, fmt = '%.18f')
        trainset=(train_Data-sca_min)/(sca_max-sca_min)
        valset=(test_Data-sca_min)/(sca_max-sca_min)
        
        
        #trainData, TestData = train_test_split(data, test_size=0.01)
        
        # %%
        batch = 64  # batch size，an index of 2 is recommended
        lr = 0.00001    # Learning rate
        L2 = 0.00001    # L2 regularization coefficient
        epoch = 1600   #Training times
        bn = False    # Whether to add the bn layer
        drop = False    # Add dropout or not
        
        layer = (5,64,64,64,64,1)    # Network structure
        inputrange = layer[0]    # Enter the number of features
        model = Net(layer, bn, drop)
        #model=torch.load('./model/best_model.pth')
        trainer = Trainer(model, batch, lr, L2, trainset,valset)#trainData)
        params = {'lr': lr, 'batch': batch, 'L2': L2, 'epoch': epoch,
                  'inputrange': inputrange, 'actfunc': 'ReLU',
                  'sca': 'min-max', 'sca_max': sca_max, 'sca_min': sca_min}
        """Model training"""
        trainer.run(epoch, inputrange)
        """Save model"""
        trainer.save(**params)
    
