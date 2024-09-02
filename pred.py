# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:02:00 2022

@author: wangze
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
""" 输入数据"""
sca_max = np.loadtxt('./data_file/sca_max.dat')
sca_min = np.loadtxt('./data_file/sca_min.dat')
data_pre=np.loadtxt('./data_file/data_pre6_EXP.dat')
xyz=np.loadtxt('./data_file/xyz6.dat')
xyz[:,0]=xyz[:,0]+0.1578
"""数据处理 """
inputrange=22
preset=(data_pre-sca_min)/(sca_max-sca_min)
preset_in=preset[:,0:inputrange]
preset_out=preset[:,inputrange:]
"""调用训练好的模型"""
model = torch.load('./model_UTmulti/best_model.pth')
"""结果预测"""
preset_in = torch.tensor(preset_in).float()
result_pred, result_mid = model(preset_in)
result_pred=result_pred.data.numpy() 
result_mid=result_mid.data.numpy() 
sca_max_Q=sca_max[inputrange]
sca_min_Q=sca_min[inputrange]
sca_max_mid=sca_max[inputrange+1:,]
sca_min_mid=sca_min[inputrange+1:,]
result_pred=result_pred*(sca_max_Q-sca_min_Q)+sca_min_Q
result_mid=result_mid*(sca_max_mid-sca_min_mid)+sca_min_mid
Q_ture=data_pre[:,inputrange].reshape(-1,1)
mid_true=data_pre[:,inputrange+1:].reshape(-1,1)
Ls=data_pre[:,0].reshape(-1,1)
xyz=xyz.reshape(-1,3)
data_result=np.hstack([xyz,np.abs(result_pred),Q_ture])
np.savetxt('result.dat',data_result)
#计算mse
error=abs(data_result[:,3]-data_result[:,4])
r_error=error/data_result[:,4]
rmse=((error*error).mean())**0.5
r_mse=rmse/data_result[:,4].mean()
print(r_mse)
#r_mse_all.append(r_mse)


"""结果输出 """
zone0=data_result[0:1891,:]
np.savetxt('./data_file/zone0.dat',zone0,header = 'Variables = x, y, z, Q_Pre, Q_true \n zone T = "Zone0" \n I = 1 \n J = 31 \n K = 61 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone1=data_result[1891:1891+21*31,:]
np.savetxt('./data_file/zone1.dat',zone1,header = ' zone T = "Zone1" \n I = 21 \n J = 1 \n K = 31 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone3=data_result[2542:4433,:]
np.savetxt('./data_file/zone3.dat',zone3,header = ' zone T = "Zone3" \n I = 61 \n J = 31 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone5=data_result[4433:6019,:]
np.savetxt('./data_file/zone5.dat',zone5,header = ' zone T = "Zone5" \n I = 61 \n J = 26 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone6=data_result[6019:6825,:]
np.savetxt('./data_file/zone6.dat',zone6,header = ' zone T = "Zone6" \n I = 26 \n J = 31 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone7=data_result[6825:7321,:]
np.savetxt('./data_file/zone7.dat',zone7,header = ' zone T = "Zone7" \n I = 16 \n J = 1 \n K = 31 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone8=data_result[7321:10432,:]
np.savetxt('./data_file/zone8.dat',zone8,header = ' zone T = "Zone8" \n I = 61 \n J = 51 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone9=data_result[10432:11238,:]
np.savetxt('./data_file/zone9.dat',zone9,header = ' zone T = "Zone9" \n I = 26 \n J = 31 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone10=data_result[11238:11868,:]
np.savetxt('./data_file/zone10.dat',zone10,header = ' zone T = "Zone10" \n I = 30 \n J = 1 \n K = 21 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone16=data_result[11868:13194,:]
np.savetxt('./data_file/zone16.dat',zone16,header = ' zone T = "Zone16" \n I = 51 \n J = 26 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone18=data_result[13194:16305,:]
np.savetxt('./data_file/zone18.dat',zone18,header = ' zone T = "Zone18" \n I = 51 \n J = 61 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone19=data_result[16305:16956,:]
np.savetxt('./data_file/zone19.dat',zone19,header = ' zone T = "Zone19" \n I = 21 \n J = 1 \n K = 31 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone20=data_result[16956:17628,:]
np.savetxt('./data_file/zone20.dat',zone20,header = ' zone T = "Zone20" \n I = 32 \n J = 1 \n K = 21 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone21=data_result[17628:18954,:]
np.savetxt('./data_file/zone21.dat',zone21,header = ' zone T = "Zone21" \n I = 51 \n J = 26 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone22=data_result[18954:19930,:]
np.savetxt('./data_file/zone22.dat',zone22,header = ' zone T = "Zone22" \n I = 16 \n J = 61 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone23=data_result[19930:20426,:]
np.savetxt('./data_file/zone23.dat',zone23,header = ' zone T = "Zone23" \n I = 16 \n J = 1 \n K = 31 \n f = POINT',
                    comments = '', fmt = '%.08f')
file_names=['zone0.dat','zone1.dat','zone3.dat','zone5.dat','zone6.dat','zone7.dat','zone8.dat','zone9.dat','zone10.dat'\
            ,'zone16.dat','zone18.dat','zone19.dat','zone20.dat','zone21.dat','zone22.dat','zone23.dat']
for file_name in file_names:
    with open('./data_file/'+file_name)as file_zone:
        data=file_zone.read()
    with open('./data_file/result.dat','a')as file_object:
          file_object.write(data)

    
