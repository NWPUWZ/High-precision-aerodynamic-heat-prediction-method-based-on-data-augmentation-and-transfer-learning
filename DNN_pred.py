# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:05:17 2023

@author: sss
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DNN_1 import Net, ActFunc

r_mse_all=[]
for num in np.arange(0,15,1):
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
    
    #求z平面试验数据与预测数据的mse
    error=abs(data_result[:,-1]-data_result[:,-2])
    r_error=error.mean()/data_result[:,-1].mean()
    #print(max(r_error))
    #print(r_error.mean())
    rmse=((error*error).mean())**0.5
    #print(rmse)
    r_mse=rmse/((data_result[:,-1]*data_result[:,-1]).mean())**0.5
    print(r_mse)
    r_mse_all.append(r_mse)
