# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:05:17 2022

@author: sss
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DNN_1 import Net, ActFunc
# class Net(nn.Module):
#     def __init__(self, layer, bn=None, drop=False):
#         super(Net, self).__init__()
#         self.layer = layer
#         self.bn = bn
#         self.net = nn.Sequential()
#         for i in range(len(self.layer)-1):
#             self.net.add_module('fc{}'.format(i+1),
#                                 nn.Linear(self.layer[i], self.layer[i+1]))
#             if i != (len(self.layer) - 2):
#                 if bn:
#                     self.net.add_module('bn{}'.format(i+1),
#                                         nn.BatchNorm1d(self.layer[i+1]))
#                 if drop:
#                     self.net.add_module('drop{}'.format(i+1),
#                                         nn.Dropout(drop))
#                 self.net.add_module('actfunc{}'.format(i+1),
#                                     ActFunc(beta=0.02))

#     def forward(self, x):
#         x = self.net(x)
#         return x
nrmse_all=[]
for num in np.arange(0,12,1):
    inputrange=5
    test_Data = np.loadtxt('./data_file/test0.dat')
    sca_max = np.loadtxt('./sca_max.dat')
    sca_min = np.loadtxt('./sca_min.dat')
    testData=(test_Data -sca_min)/(sca_max-sca_min)
    test_x=testData[:,0:inputrange]
    #test_y=testData[:,-1]
    test_x1 = torch.FloatTensor(test_x)
    #model = torch.load('./model/best_model.pth')
    model = torch.load('./model/models/model'+str(num+1)+'.pth')
    model.eval()
    test_pred1 = model(test_x1)
    test_pred=test_pred1.data.numpy() 
    #test_loss = (test_pred1-test_y1).data.numpy() 
    sca_max_Q=sca_max[inputrange]
    sca_min_Q=sca_min[inputrange]
    test_pred=test_pred*(sca_max_Q-sca_min_Q)+sca_min_Q
    test_pred=test_pred.reshape(-1,1)
    Q_true=test_Data[:,-2].reshape(-1,1)
    sign=test_Data[:,-1].reshape(-1,1)
    xyz=test_Data[:,0:3].reshape(-1,3)
    data_result=np.hstack([xyz,np.abs(test_pred),Q_true])
    
    # index_RANS=(sign!=0)
    # index_EXP=(sign==0)
    # data_result_RANS=data_result[index_RANS.flatten(),:]
    # index_RANS_z=(data_result_RANS[:,2]==0)
    # index_RANS_x=(data_result_RANS[:,2]!=0)
    # result_RANS_z=data_result_RANS[index_RANS_z,:]
    # #result_RANS_z=data_result_RANS
    # result_RANS_zup=result_RANS_z[0:130,:]
    # result_RANS_zdown=result_RANS_z[130:,:]
    # result_RANS_zdown[:,0]=-result_RANS_zdown[:,0]
    # result_RANS_x=data_result_RANS[index_RANS_x,:]
    # index_RANS_x1=(result_RANS_x[:,0]==0.078)
    # index_RANS_x2=(result_RANS_x[:,0]==0.12)
    # result_RANS_x1=result_RANS_x[index_RANS_x1,:]
    # result_RANS_x2=result_RANS_x[index_RANS_x2,:]
    
    # data_result_EXP=data_result[index_EXP.flatten(),:]
    # index_EXP_z=(data_result_EXP[:,2]==0)
    # index_EXP_x=(data_result_EXP[:,2]!=0)
    # result_EXP_z=data_result_EXP[index_EXP_z,:]
    # #result_EXP_z=data_result_EXP
    # result_EXP_zup= result_EXP_z[0:16,:]
    # result_EXP_zdown= result_EXP_z[16:,:]
    # result_EXP_zdown[:,0]=-result_EXP_zdown[:,0]
    # result_EXP_x=data_result_EXP[index_EXP_x,:]
    # index_EXP_x1=(result_EXP_x[:,0]==-0.0798)
    # index_EXP_x2=(result_EXP_x[:,0]==-0.0378)
    # result_EXP_x1=result_EXP_x[index_EXP_x1,:]
    # result_EXP_x2=result_EXP_x[index_EXP_x2,:]
    
    error=abs(data_result[:,-1]-data_result[:,-2])
    r_error=error.mean()/data_result[:,-1].mean()
    #print(max(r_error))
    #print(r_error.mean())
    rmse=((error*error).mean())**0.5
    #print(rmse)
    nrmse=rmse/((data_result[:,-1]*data_result[:,-1]).mean())**0.5
    print(nrmse)
    nrmse_all.append(nrmse)
