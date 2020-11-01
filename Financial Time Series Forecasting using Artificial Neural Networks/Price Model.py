# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:21:00 2020

@author: USER
"""


record = []

############ Hyperparameters #############

SAE = True

latent_factor = 23                              ###         latent_dim = data.shape[1]-autoencoder_depth*latent_factor
encoder_activation = 'tanh'
encoder_recurrent_activation = 'sigmoid'
decoder_activation = 'tanh'
decoder_recurrent_activation = 'sigmoid'
autoencoder_optimizer = 'adam'
autoencoder_loss = 'mse'
autoencoder_iterations = 3000                   ###       
autoencoder_batch_size = 32
autoencoder_depth = 3
finetuning_iterations = 1000                    ###

window = 35                                     ###
# LSTM_layers = 1                           
# LSTM_layers_sizes = [100,100,100,100]       
LSTM_activation = 'tanh'
LSTM_recurrent_activation = 'sigmoid'
# Dropout_rate = 0.2
LSTM_optimizer = 'adam'
LSTM_loss = 'mse'
LSTM_iterations = 15000                         ###
LSTM_batch_size = 32
LSTM_layer_size = 120                           ###

smoothing = 2
P = 20

############ Packages #############

import math
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
import matplotlib as plt

import datetime

import pandas as pd

from keras.models import Sequential
from keras.models import load_model
# from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Input
# from keras.layers import BatchNormalization
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed


# from tensorflow.keras.layers.experimental import preprocessing



############ Data #############

SP500=np.array(pd.read_csv('S&P500.csv'))
DAX=np.array(pd.read_csv('DAX.csv'))
CAC=np.array(pd.read_csv('CAC40.csv'))
N225=np.array(pd.read_csv('NIKKEI225.csv'))
HSI=np.array(pd.read_csv('HANG SENG.csv'))
DJIA=np.array(pd.read_csv('DJIA.csv'))
NASDAQ=np.array(pd.read_csv('NASDAQ.csv'))
FTSE=np.array(pd.read_csv('UK 100 Historical Data.csv'))
DAAA=np.array(pd.read_csv('DAAA.csv'))
DCOILWTICO=np.array(pd.read_csv('DCOILWTICO.csv'))
DFF=np.array(pd.read_csv('DFF.csv'))
T10YIE=np.array(pd.read_csv('T10YIE.csv'))
TRY=np.array(pd.read_csv('TREASURY YIELD.csv'))
US30FUT=np.array(pd.read_csv('US 30 Futures Historical Data.csv'))
DOLLARINDEX=np.array(pd.read_csv('US Dollar Index Futures Historical Data.csv'))


variables=[SP500,DAX,CAC,N225,HSI,DJIA,NASDAQ,FTSE,DAAA,DCOILWTICO,DFF,T10YIE,TRY,US30FUT,DOLLARINDEX]        

for j in range(0,len(variables)):
    for i in range(0,variables[j].shape[0]):        
        if pd.isnull(variables[j][i,1]) == True or variables[j][i,1] == '.':
            variables[j][i,1:variables[j].shape[1]] = 0


SP500=np.array(variables[0])
DAX=np.array(variables[1])
CAC=np.array(variables[2])
N225=np.array(variables[3])
HSI=np.array(variables[4])
DJIA=np.array(variables[5])
NASDAQ=np.array(variables[6])
FTSE=np.array(variables[7])
DAAA=np.array(variables[8])
DCOILWTICO=np.array(variables[9])
DFF=np.array(variables[10])
T10YIE=np.array(variables[11])
TRY=np.array(variables[12])
US30FUT=np.array(variables[13])
DOLLARINDEX=np.array(variables[14])

TRY = np.delete(TRY,6,1)
FTSE = np.delete(FTSE,6,1)
FTSE = np.delete(FTSE,5,1)

for j in range(5,7):
    for i in range(0,DOLLARINDEX.shape[0]):
        DOLLARINDEX[i,j] = float(DOLLARINDEX[i,j][:-1])

for j in range(5,7):
    for i in range(0,US30FUT.shape[0]):
        US30FUT[i,j] = float(US30FUT[i,j][:-1])

variables=[SP500,DAX,CAC,N225,HSI,DJIA,NASDAQ,FTSE,DAAA,DCOILWTICO,DFF,T10YIE,TRY,US30FUT,DOLLARINDEX]  

for i in range(0,len(variables)):
    variables[i][:,1:variables[i].shape[1]]=np.array(variables[i][:,1:variables[i].shape[1]], dtype=float)

for i in range(1,len(variables)):
    for date in SP500[:,0]:
        if date in variables[i][:,0]:
            continue
        else:
            a = np.array([date])
            for j in range(0,variables[i].shape[1]-1):
                a = np.append(a,0)
            a = np.reshape(a,(1,variables[i].shape[1]))
            variables[i] = np.append(variables[i],a,0)


date = '2005-01-01'
date = datetime.datetime.strptime(date, '%Y-%m-%d')
end_date = '2020-01-01'
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')


data = []

while date < end_date:
    
    if str(date.date()) in SP500[:,0]:
        data.append([])
        data[len(data)-1]=SP500[np.where(SP500==str(date.date()))[0][0],1:]
        for j in range(1,len(variables)):
            data[len(data)-1]=np.concatenate((data[len(data)-1],variables[j][np.where(variables[j]==str(date.date()))[0][0],1:]))
            
    date = date + datetime.timedelta(days=1)


data = np.array(data, dtype=float)


for j in range(0,data.shape[1]):
    for i in [0,data.shape[0]-1]:
        if data[i,j]==0:
            if i==0:
                k=1
                while data[i+k,j]==0:
                    k+=1
                for l in range(0,k):
                    data[i+l,j]=data[i+k,j]
            else:
                k=1
                while data[i-k,j]==0:
                    k+=1
                for l in range(0,k):
                    data[i-l,j]=data[i-k,j]
                

for j in range(0,data.shape[1]):
    for i in range(0,data.shape[0]):
        if data[i,j]==0:
            k=1
            while data[i+k,j]==0:
                k+=1
            for l in range(0,k):
                data[i+l,j]=data[i-1,j]+(l+1)*((data[i+k,j]-data[i-1,j])/(k+1))


EMA12 = []          
k = 0
for j in [4,10,16,22,28,34,40,42]:                
    EMA12.append([])
    for i in range(0,data.shape[0]):
        if i < 11:
            EMA12[k].append(0)
        elif i == 11:
            EMA12[k].append(np.mean(data[0:i+1,j]))
        else:
            EMA12[k].append(data[i,j]*smoothing/(1+12)+EMA12[k][i-1]*(1-smoothing/(1+12)))
    k+=1
        
EMA12 = np.array(EMA12, dtype=float)
EMA12 = np.moveaxis(EMA12,1,0)

EMA26 = []          
k = 0
for j in [4,10,16,22,28,34,40,42]:                
    EMA26.append([])
    for i in range(0,data.shape[0]):
        if i < 25:
            EMA26[k].append(0)
        elif i == 25:
            EMA26[k].append(np.mean(data[0:i+1,j]))
        else:
            EMA26[k].append(data[i,j]*smoothing/(1+26)+EMA26[k][i-1]*(1-smoothing/(1+26)))
    k+=1
        
EMA26 = np.array(EMA26, dtype=float)
EMA26 = np.moveaxis(EMA26,1,0)

MACD = EMA12-EMA26

MEMA = []
k = 0
for j in range(0,MACD.shape[1]):                 
    MEMA.append([])
    for i in range(0,MACD.shape[0]):
        if i < 33:
            MEMA[k].append(0)
        elif i == 33:
            MEMA[k].append(np.mean(MACD[25:i+1,j]))
        else:
            MEMA[k].append(MACD[i,j]*smoothing/(1+9)+MEMA[k][i-1]*(1-smoothing/(1+9)))
    k+=1

MEMA = np.array(MEMA, dtype=float)
MEMA = np.moveaxis(MEMA,1,0)


typical = []
k = 0
for j in [1,7,13,19,25,31,37,42]:
    typical.append([])
    for i in range(0,data.shape[0]):
        if i < P-1:
            typical[k].append(0)
        else:
            if j == 42:
                typical[k].append((data[i,j]+data[i,j+2]+data[i,j+3])/3)
            else:
                typical[k].append((data[i,j]+data[i,j+1]+data[i,j+2])/3)
    k+=1


typical = np.array(typical, dtype=float)
typical = np.moveaxis(typical,1,0)


MA = []
k = 0
for j in range(0,typical.shape[1]):
    MA.append([])
    for i in range(0,typical.shape[0]):
        if i < 2*(P-1):  
            MA[k].append(0)
        else:
            MA[k].append(np.mean(typical[i+1-P:i+1,j]))
    k+=1

MA = np.array(MA, dtype=float)
MA = np.moveaxis(MA,1,0)


deviation = np.abs(typical-MA)

mean_deviation = []
k = 0
for j in range(0,deviation.shape[1]):
    mean_deviation.append([])
    for i in range(0,deviation.shape[0]):
        if i < 3*(P-1):  
            mean_deviation[k].append(0)
        else:
            mean_deviation[k].append(np.mean(deviation[i+1-P:i+1,j]))
    k+=1

mean_deviation = np.array(mean_deviation, dtype=float)
mean_deviation = np.moveaxis(mean_deviation,1,0)


CCI = []
k = 0
for j in range(0,typical.shape[1]):
    CCI.append([])
    for i in range(0,typical.shape[0]):
        if i < 3*(P-1): 
            CCI[k].append(0)
        else:
            CCI[k].append((typical[i,j]-MA[i,j])/(0.015*mean_deviation[i,j]))
    k+=1


CCI = np.array(CCI, dtype=float)
CCI = np.moveaxis(CCI,1,0)


true_range = []
k = 0
for j in [1,7,13,19,25,31,37,42]:
    true_range.append([])
    for i in range(0,data.shape[0]):
        if i < 1:
            true_range[k].append(0)
        else:
            if j == 42:
                true_range[k].append(max(data[i,j+2]-data[i,j+3],abs(data[i,j+2]-data[i-1,j]),abs(data[i,j+3]-data[i-1,j])))
            else:
                true_range[k].append(max(data[i,j]-data[i,j+1],abs(data[i,j]-data[i-1,j+2]),abs(data[i,j+1]-data[i-1,j+2])))
    k+=1

true_range = np.array(true_range, dtype=float)
true_range = np.moveaxis(true_range,1,0)


ATR = []
k = 0
for j in range(0,true_range.shape[1]):
    ATR.append([])
    for i in range(0,true_range.shape[0]):
        if i < 14:  
            ATR[k].append(0)
        else:
            ATR[k].append(np.mean(true_range[i+1-14:i+1,j]))
    k+=1

ATR = np.array(ATR, dtype=float)
ATR = np.moveaxis(ATR,1,0)


square_deviation=np.square(typical-MA)

stdeviation = []
k = 0
for j in range(0,square_deviation.shape[1]):
    stdeviation.append([])
    for i in range(0,square_deviation.shape[0]):
        if i < 3*(P-1):  
            stdeviation[k].append(0)
        else:
            stdeviation[k].append(np.sqrt(np.mean(square_deviation[i+1-P:i+1,j])))
    k+=1

stdeviation = np.array(stdeviation, dtype=float)
stdeviation = np.moveaxis(stdeviation,1,0)


BOLU = MA + 2*stdeviation
BOLD = MA - 2*stdeviation


EMA20 = []          
k = 0
for j in [4,10,16,22,28,34,40,42]:                 
    EMA20.append([])
    for i in range(0,data.shape[0]):
        if i < 19:
            EMA20[k].append(0)
        elif i == 19:
            EMA20[k].append(np.mean(data[0:i+1,j]))
        else:
            EMA20[k].append(data[i,j]*smoothing/(1+20)+EMA20[k][i-1]*(1-smoothing/(1+20)))
    k+=1
        
EMA20 = np.array(EMA20, dtype=float)
EMA20 = np.moveaxis(EMA20,1,0)


momentum = []
k = 0
for j in [4,10,16,22,28,34,40,42]:                 
    momentum.append([])
    for i in range(0,data.shape[0]):
        if i < 10:
            momentum[k].append(0)
        else:
            momentum[k].append(data[i,j]-data[i-10,j])
    k+=1
  
momentum = np.array(momentum, dtype=float)
momentum = np.moveaxis(momentum,1,0)


ROC = []
k = 0
for j in [4,10,16,22,28,34,40,42]:                 
    ROC.append([])
    for i in range(0,data.shape[0]):
        if i < 20:
            ROC[k].append(0)
        else:
            ROC[k].append((data[i,j]-data[i-20,j])/data[i-20,j])
    k+=1
  
ROC = np.array(ROC, dtype=float)
ROC = np.moveaxis(ROC,1,0)



SMI = []
k = 0
for j in [1,7,13,19,25,31,37,42]:
    SMI.append([])
    for i in range(0,data.shape[0]):
        if i < 14:
            SMI[k].append(0)
        else:
            if j == 42:
                SMI[k].append((data[i,j]-(max(data[i-14:i,j+2])-min(data[i-14:i,j+3]))/2)/(max(data[i-14:i,j+2])-min(data[i-14:i,j+3])))
            else:
                SMI[k].append((data[i,j+2]-(max(data[i-14:i,j])-min(data[i-14:i,j+1]))/2)/(max(data[i-14:i,j])-min(data[i-14:i,j+1])))
    k+=1

SMI = np.array(SMI, dtype=float)
SMI = np.moveaxis(SMI,1,0)



AD = []
k = 0
for j in [4,10,16,22,28,34,40]:                 
    AD.append([])
    for i in range(0,data.shape[0]):
        if i == 0:
            AD[k].append(((data[i,j-1]-data[i,j-2])-(data[i,j-3]-data[i,j-1]))/(data[i,j-3]-data[i,j-2])*data[i,j+1])
        else:
            if data[i,j-3] == data[i,j-2]:
                AD[k].append(AD[k][i-1])
            else:
                AD[k].append(AD[k][i-1]+((data[i,j-1]-data[i,j-2])-(data[i,j-3]-data[i,j-1]))/(data[i,j-3]-data[i,j-2])*data[i,j+1])
    k+=1
  
AD = np.array(AD, dtype=float)
AD = np.moveaxis(AD,1,0)

data = np.concatenate((data,BOLU,BOLD,EMA20,MACD,MEMA,CCI,ATR,momentum,ROC,SMI,AD),1) 
data = data[57:]


############ Preprocessing #############

split = math.floor(0.8*data.shape[0])
split2 = math.ceil(0.9*data.shape[0])

features_set = []

for i in range(0,data.shape[1]):
    features_set.append([])

for j in range (0,data.shape[1]):
    for i in range(window,split):
        if j in [0,1,2,3,4,67,75,83]:
            minimum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).min()
            maximum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [6,7,8,9,10,68,76,84]:
            minimum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).min()
            maximum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [12,13,14,15,16,69,77,85]:
            minimum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).min()
            maximum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [18,19,20,21,22,70,78,86]:
            minimum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).min()
            maximum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [24,25,26,27,28,71,79,87]:
            minimum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).min()
            maximum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [30,31,32,33,34,72,80,88]:
            minimum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).min()
            maximum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [36,37,38,39,40,73,81,89]:
            minimum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).min()
            maximum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [42,43,44,45,74,82,90]:
            minimum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).min()
            maximum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [91,92,93,94,95,96,97,98]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [99,100,101,102,103,104,105,106]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [50,51,52,53,54]:
            minimum = data[i-window:i,50:55].min()
            maximum = data[i-window:i,50:55].max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [55,56,57,58,59,60]:
            minimum = data[i-window:i,55:61].min()
            maximum = data[i-window:i,55:61].max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [61,62,63,64,65,66]:
            minimum = data[i-window:i,61:67].min()
            maximum = data[i-window:i,61:67].max()
            
            features_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
            
        else:
            if min(data[i-window:i,j]) == max(data[i-window:i,j]):
                features_set[j].append(data[i-window:i, j]-min(data[i-window:i,j]))
            else:
                features_set[j].append((data[i-window:i, j]-min(data[i-window:i,j]))/(max(data[i-window:i,j])-min(data[i-window:i,j])))


labels = []
training_reality = []

for i in range(window, split):
    minimum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).min()
    maximum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).max()
    labels.append((data[i, 4]-minimum)/(maximum-minimum))
    training_reality.append(data[i, 4])
    
features_set, labels = np.array(features_set, dtype=float), np.array(labels, dtype=float)
features_set = np.moveaxis(features_set, 0, -1)


forecasting_set = []

for i in range(0,data.shape[1]):
    forecasting_set.append([])

for j in range (0,data.shape[1]):
    for i in range(split+window,split2):
        if j in [0,1,2,3,4,67,75,83]:
            minimum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).min()
            maximum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [6,7,8,9,10,68,76,84]:
            minimum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).min()
            maximum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [12,13,14,15,16,69,77,85]:
            minimum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).min()
            maximum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [18,19,20,21,22,70,78,86]:
            minimum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).min()
            maximum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [24,25,26,27,28,71,79,87]:
            minimum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).min()
            maximum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [30,31,32,33,34,72,80,88]:
            minimum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).min()
            maximum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [36,37,38,39,40,73,81,89]:
            minimum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).min()
            maximum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [42,43,44,45,74,82,90]:
            minimum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).min()
            maximum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [91,92,93,94,95,96,97,98]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [99,100,101,102,103,104,105,106]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [50,51,52,53,54]:
            minimum = data[i-window:i,50:55].min()
            maximum = data[i-window:i,50:55].max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [55,56,57,58,59,60]:
            minimum = data[i-window:i,55:61].min()
            maximum = data[i-window:i,55:61].max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [61,62,63,64,65,66]:
            minimum = data[i-window:i,61:67].min()
            maximum = data[i-window:i,61:67].max()
            
            forecasting_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
                        
        else:
            if min(data[i-window:i,j]) == max(data[i-window:i,j]):
                forecasting_set[j].append(data[i-window:i, j]-min(data[i-window:i,j]))
            else:
                forecasting_set[j].append((data[i-window:i, j]-min(data[i-window:i,j]))/(max(data[i-window:i,j])-min(data[i-window:i,j])))
    


reality = []
naive_predictions = []

for i in range(split+window, split2):
    reality.append(data[i, 4])
    naive_predictions.append(data[i-1, 4])

forecasting_set, reality = np.array(forecasting_set, dtype=float), np.array(reality, dtype=float)
forecasting_set = np.moveaxis(forecasting_set, 0, -1)


test_set = []

for i in range(0,data.shape[1]):
    test_set.append([])

for j in range (0,data.shape[1]):
    for i in range(split2+window,data.shape[0]):
        if j in [0,1,2,3,4,67,75,83]:
            minimum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).min()
            maximum = np.concatenate((data[i-window:i,0:5],data[i-window:i,67:68],data[i-window:i,75:76],data[i-window:i,83:84]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [6,7,8,9,10,68,76,84]:
            minimum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).min()
            maximum = np.concatenate((data[i-window:i,6:11],data[i-window:i,68:69],data[i-window:i,76:77],data[i-window:i,84:85]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [12,13,14,15,16,69,77,85]:
            minimum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).min()
            maximum = np.concatenate((data[i-window:i,12:17],data[i-window:i,69:70],data[i-window:i,77:78],data[i-window:i,85:86]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [18,19,20,21,22,70,78,86]:
            minimum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).min()
            maximum = np.concatenate((data[i-window:i,18:23],data[i-window:i,70:71],data[i-window:i,78:79],data[i-window:i,86:87]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [24,25,26,27,28,71,79,87]:
            minimum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).min()
            maximum = np.concatenate((data[i-window:i,24:29],data[i-window:i,71:72],data[i-window:i,79:80],data[i-window:i,87:88]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [30,31,32,33,34,72,80,88]:
            minimum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).min()
            maximum = np.concatenate((data[i-window:i,30:35],data[i-window:i,72:73],data[i-window:i,80:81],data[i-window:i,88:89]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [36,37,38,39,40,73,81,89]:
            minimum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).min()
            maximum = np.concatenate((data[i-window:i,36:41],data[i-window:i,73:74],data[i-window:i,81:82],data[i-window:i,89:90]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [42,43,44,45,74,82,90]:
            minimum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).min()
            maximum = np.concatenate((data[i-window:i,42:46],data[i-window:i,74:75],data[i-window:i,82:83],data[i-window:i,90:91]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [91,92,93,94,95,96,97,98]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j+8:j+9]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [99,100,101,102,103,104,105,106]:
            minimum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).min()
            maximum = np.concatenate((data[i-window:i,j:j+1],data[i-window:i,j-8:j-7]),1).max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [50,51,52,53,54]:
            minimum = data[i-window:i,50:55].min()
            maximum = data[i-window:i,50:55].max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [55,56,57,58,59,60]:
            minimum = data[i-window:i,55:61].min()
            maximum = data[i-window:i,55:61].max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
        elif j in [61,62,63,64,65,66]:
            minimum = data[i-window:i,61:67].min()
            maximum = data[i-window:i,61:67].max()
            
            test_set[j].append((data[i-window:i, j]-minimum)/(maximum-minimum))
                        
        else:
            if min(data[i-window:i,j]) == max(data[i-window:i,j]):
                test_set[j].append(data[i-window:i, j]-min(data[i-window:i,j]))
            else:
                test_set[j].append((data[i-window:i, j]-min(data[i-window:i,j]))/(max(data[i-window:i,j])-min(data[i-window:i,j])))
        

test_labels = []

for i in range(split2+window, data.shape[0]):
    test_labels.append(data[i, 4])

test_set, test_labels = np.array(test_set, dtype=float), np.array(test_labels, dtype=float)
test_set = np.moveaxis(test_set, 0, -1)



############ SAE #############

if SAE == True:                                 
    
    latent_dim = data.shape[1]-autoencoder_depth*latent_factor   
    epic_dictionary = {}
    x = features_set
     
    
    for i in range(0,autoencoder_depth):
    
        epic_dictionary[f'encoded{i}'] = LSTM(int(latent_dim+math.floor((data.shape[1]-latent_dim)/autoencoder_depth)*(autoencoder_depth-1-i)), activation=encoder_activation, recurrent_activation=encoder_recurrent_activation, return_sequences=True)
        epic_dictionary[f'decoded{i}'] = LSTM(int(latent_dim+math.floor((data.shape[1]-latent_dim)/autoencoder_depth)*(autoencoder_depth-i)), activation=decoder_activation, recurrent_activation=decoder_recurrent_activation, return_sequences=True)
        
        sequence_autoencoder = Sequential()
        
        sequence_autoencoder.add(epic_dictionary[f'encoded{i}'])
        sequence_autoencoder.add(epic_dictionary[f'decoded{i}'])
        
        encoder = Sequential()
        for j in range(0,i+1):
            encoder.add( epic_dictionary[f'encoded{j}'])
        
        sequence_autoencoder.compile(optimizer=autoencoder_optimizer, loss=autoencoder_loss)
        sequence_autoencoder.fit(x, x, epochs=math.ceil(autoencoder_iterations*autoencoder_batch_size/features_set.shape[0]), batch_size=autoencoder_batch_size)
                       
        x = encoder.predict(features_set)
    
    
    SAE = Sequential()
    for i in range(0,autoencoder_depth):
        SAE.add(epic_dictionary[f'encoded{i}'])
    for i in range(0,autoencoder_depth):
        SAE.add(epic_dictionary[f'decoded{autoencoder_depth-1-i}'])
    
    
    SAE.compile(optimizer=autoencoder_optimizer, loss=autoencoder_loss)
    SAE.fit(features_set, features_set, epochs=math.ceil(finetuning_iterations*autoencoder_batch_size/features_set.shape[0]), batch_size=autoencoder_batch_size)
     
    features_set = encoder.predict(features_set)
    forecasting_set = encoder.predict(forecasting_set)
    test_set = encoder.predict(test_set)



############ Forecasting Model #############

model = Sequential()

model.add(LSTM(LSTM_layer_size, activation=LSTM_activation, recurrent_activation=LSTM_recurrent_activation, return_sequences=True))    
# model.add(Dropout(Dropout_rate))

model.add(LSTM(LSTM_layer_size, activation=LSTM_activation, recurrent_activation=LSTM_recurrent_activation, return_sequences=True))    
# model.add(Dropout(Dropout_rate))

model.add(LSTM(LSTM_layer_size, activation=LSTM_activation, recurrent_activation=LSTM_recurrent_activation))
# model.add(Dropout(Dropout_rate))

model.add(Dense(1))


model.compile(optimizer=LSTM_optimizer, loss=LSTM_loss)
model.fit(features_set, labels, epochs=math.ceil(LSTM_iterations*LSTM_batch_size/features_set.shape[0]), batch_size=LSTM_batch_size)

predictions = model.predict(forecasting_set)

for j in range(split+window,split2):
    predictions[j-split-window] = predictions[j-split-window]*(max(data[j-window:j,4])-min(data[j-window:j,4]))+min(data[j-window:j,4])
        
predictions = np.reshape(predictions,(predictions.shape[0],))
error = np.subtract(predictions,reality)

naive_predictions = np.array(naive_predictions, dtype=float)
naive_error = np.subtract(naive_predictions,reality)

abs_error = np.absolute(error)
abs_naive_error = np.absolute(naive_error)


print(f'\nWindow = {window}')
print(f'Latent dimension = {latent_dim}')
print(f'Autoencoder iterations = {autoencoder_iterations}')
print(f'Finetuning iterations = {finetuning_iterations}')
print(f'LSTM layer size = {LSTM_layer_size}')
print(f'LSTM iterations = {LSTM_iterations}')

MASE = np.mean(abs_error)/np.mean(abs_naive_error)
print('\nMASE =',MASE)


train_pred = model.predict(features_set)

for j in range(window,split):
    train_pred[j-window] = train_pred[j-window]*(max(data[j-window:j,4])-min(data[j-window:j,4]))+min(data[j-window:j,4])

train_pred = np.reshape(train_pred,(train_pred.shape[0],))

plt.pyplot.figure(1)
plt.pyplot.plot(reality)
plt.pyplot.plot(predictions)
plt.pyplot.show()

predicted_change = predictions-data[split+window-1:split2-1,4]
real_change = data[split+window:split2,4]-data[split+window-1:split2-1,4]

change_error = predicted_change-real_change

buy = np.where(predicted_change>0,1,0)

assets = [1,0]
daily_profit = [1]
asset_value = 1
for i in range(0,buy.shape[0]):
    daily_profit.append((assets[0]+assets[1]*data[split+window-1+i,4])/asset_value)
    if buy[i]==1:
        assets = [0,assets[1]+assets[0]/data[split+window-1+i,4]]
    else:
        assets = [assets[0]+assets[1]*data[split+window-1+i,4],0]
    asset_value = assets[0]+assets[1]*data[split+window-1+i,4]
        
daily_profit = np.delete(np.delete(daily_profit,0,0),0,0)
temp = 1
for i in range(len(daily_profit)):
    temp = temp*daily_profit[i]

avgdaily_profit = math.pow(temp,1/(len(daily_profit)))
STD = np.sqrt(np.mean(np.square(daily_profit-np.array(avgdaily_profit))))

plt.pyplot.figure(2)
plt.pyplot.plot((daily_profit-np.array(1))*100)
plt.pyplot.show()

profit = round((assets[0]+assets[1]*data[split2-1,4]-1)*100,2)

long_term_profit = round((data[split2-1,4]/data[split+window-1,4]-1)*100,2)

tradeless_daily = math.pow(data[split2-1,4]/data[split+window-1,4],1/len(daily_profit))
print(f'\nAverage daily profit = {round((avgdaily_profit-1)*100,3)}%')
print(f'Average daily profit (no trade) = {round((tradeless_daily-1)*100,3)}%')
print(f'\nStandard deviation = {round(STD*100,3)} pp')
print(f'2 std confidence interval = [{round((avgdaily_profit-1)*100-2*STD*100,3)}%, {round((avgdaily_profit-1)*100+2*STD*100,3)}%]')
print(f'[min,max] = [{round((min(daily_profit)-1)*100,3)}%, {round((max(daily_profit)-1)*100,3)}%]')

print(f'\nProfit at the end of exercise = {profit}%')
print(f'Profit at the end of exercise without trading = {long_term_profit}%')

bassets = [1,0]
for i in range(0,buy.shape[0]):
    if real_change[i]>0:
        bassets = [0,bassets[1]+bassets[0]/data[split+window-1+i,4]]
    else:
        bassets = [bassets[0]+bassets[1]*data[split+window-1+i,4],0]


perfect_profit = round((bassets[0]+bassets[1]*data[split2-1,4]-1)*100,2)

print(f'Theoretical maximum profit at the end of exercise = {perfect_profit}%')

print('-----------------------------------------------------------------')


