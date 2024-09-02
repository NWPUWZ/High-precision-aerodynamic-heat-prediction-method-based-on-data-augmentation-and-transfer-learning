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
""" Input data"""
sca_max = np.loadtxt('./data_file/sca_max.dat')
sca_min = np.loadtxt('./data_file/sca_min.dat')
data_pre=np.loadtxt('./data_file/data_pre20.dat')
xyz=np.loadtxt('./data_file/xyz.dat')
xyz[:,0]=xyz[:,0]-np.max(xyz[:,0])
"""Data processing """
inputrange=22
preset=(data_pre-sca_min)/(sca_max-sca_min)
preset_in=preset[:,0:inputrange]
preset_out=preset[:,inputrange:]
"""Call the trained model"""
model = torch.load('./model_UTmulti/best_model.pth')
"""Outcome prediction"""
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
# data_result[:,3]=data_result[:,3]/np.max(data_result[:,3])
# data_result[:,4]=data_result[:,4]/np.max(data_result[:,4])
#np.savetxt('result.dat',data_result)
"""Calculating nrmse """
error=abs(data_result[:,3]-data_result[:,4])
r_error=error/data_result[:,4]
rmse=((error*error).mean())**0.5
nrmse=rmse/((data_result[:,4]*data_result[:,4]).mean())**0.5
print(nrmse)
#r_mse_all.append(r_mse)


"""Result output """
zone0=data_result[0:4500,:]
np.savetxt('./data_file/zone0.dat',zone0,header = 'Variables = x, y, z, q_pred, q_recon \n zone T = "Zone0" \n I = 50 \n J = 1 \n K = 90 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone2=data_result[4500:9000,:]
np.savetxt('./data_file/zone2.dat',zone2,header = ' zone T = "Zone2" \n I = 90 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone4=data_result[9000:13500,:]
np.savetxt('./data_file/zone4.dat',zone4,header = ' zone T = "Zone4" \n I = 90 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone5=data_result[13500:18000,:]
np.savetxt('./data_file/zone5.dat',zone5,header = ' zone T = "Zone5" \n I = 90 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone8=data_result[18000:22500,:]
np.savetxt('./data_file/zone8.dat',zone8,header = ' zone T = "Zone8" \n I = 90 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone9=data_result[22500:27000,:]
np.savetxt('./data_file/zone9.dat',zone9,header = ' zone T = "Zone9" \n I = 50 \n J = 1 \n K = 90 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone10=data_result[27000:28700,:]
np.savetxt('./data_file/zone10.dat',zone10,header = ' zone T = "Zone10" \n I = 25 \n J = 68 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone11=data_result[28700:30400,:]
np.savetxt('./data_file/zone11.dat',zone11,header = ' zone T = "Zone11" \n I = 68 \n J = 1 \n K = 25 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone13=data_result[30400:33730,:]
np.savetxt('./data_file/zone13.dat',zone13,header = ' zone T = "Zone13" \n I = 37 \n J = 90 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone17=data_result[33730:36330,:]
np.savetxt('./data_file/zone17.dat',zone17,header = ' zone T = "Zone17" \n I = 52 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone19=data_result[36330:38930,:]
np.savetxt('./data_file/zone19.dat',zone19,header = ' zone T = "Zone19" \n I = 52 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone22=data_result[38930:40630,:]
np.savetxt('./data_file/zone22.dat',zone22,header = ' zone T = "Zone22" \n I = 25 \n J = 68 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone23=data_result[40630:42330,:]
np.savetxt('./data_file/zone23.dat',zone23,header = ' zone T = "Zone23" \n I = 68 \n J = 1 \n K = 25 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone25=data_result[42330: 45678,:]
np.savetxt('./data_file/zone25.dat',zone25,header = ' zone T = "Zone25" \n I = 54 \n J = 62 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone27=data_result[45678:46803,:]
np.savetxt('./data_file/zone27.dat',zone27,header = ' zone T = "Zone27" \n I = 25 \n J = 45 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone29=data_result[46803:47928,:]
np.savetxt('./data_file/zone29.dat',zone29,header = ' zone T = "Zone29" \n I = 45 \n J = 1 \n K = 25 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone30=data_result[47928:49878,:]
np.savetxt('./data_file/zone30.dat',zone30,header = ' zone T = "Zone30" \n I = 39 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone33=data_result[49878:50777,:]
np.savetxt('./data_file/zone33.dat',zone33,header = ' zone T = "Zone33" \n I = 31 \n J = 29 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone38=data_result[50777:52727,:]
np.savetxt('./data_file/zone38.dat',zone38,header = ' zone T = "Zone38" \n I = 39 \n J = 50 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')
zone42=data_result[52727:53423,:]
np.savetxt('./data_file/zone42.dat',zone42,header = ' zone T = "Zone42" \n I = 24 \n J = 29 \n K = 1 \n f = POINT',
                    comments = '', fmt = '%.08f')




file_names=['zone0.dat','zone2.dat','zone4.dat','zone5.dat','zone8.dat','zone9.dat','zone10.dat','zone11.dat','zone13.dat'\
            ,'zone17.dat','zone19.dat','zone22.dat','zone23.dat','zone25.dat','zone27.dat','zone29.dat'
            ,'zone30.dat','zone33.dat','zone38.dat','zone42.dat']
for file_name in file_names:
    with open('./data_file/'+file_name)as file_zone:
        data=file_zone.read()
    with open('./data_file/result.dat','a')as file_object:
          file_object.write(data)
    
