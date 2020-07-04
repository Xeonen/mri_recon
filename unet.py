# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:37:11 2020

@author: Xeonen
"""


import dataOp
import metrics
import customLayers as cl
import helperFunctions as hf

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import layers, models, losses
import matplotlib.pyplot as plt






inputs = layers.Input((256, 256, 1))
c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = layers.Dropout(0.1) (c1)
c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = layers.Dropout(0.1) (c2)
c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = layers.Dropout(0.2) (c3)
c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = layers.Dropout(0.2) (c4)
c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = layers.Dropout(0.3) (c5)
c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = layers.Dropout(0.2) (c6)
c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = layers.Dropout(0.2) (c7)
c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = layers.Dropout(0.1) (c8)
c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = layers.Dropout(0.1) (c9)
c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
c10 = layers.Conv2D(1, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

model = models.Model(inputs = [inputs], outputs=[c10])

model.compile(loss="MSE", optimizer="Adam")





ifft2d = tf.signal.ifft2d

def processInput(arr):  
    def castBatchComplex(arr):
        real = arr[:, :, :, 0]
        imag = arr[:, :, :, 1]    
        complex_array = tf.complex(real, imag)
        
        return(complex_array)

    arr = castBatchComplex(arr)
    dImg = ifft2d(arr)
    dImg = tf.abs(dImg)
    dImg = tf.reshape(dImg, (-1, 256, 256, 1))
    
    return(dImg)

dVal = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 8, 4, 10, False)
dTrain = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Train", 8, 4, 10, False)




################ Train with Magnitude Only

trainList = list()
valList = list()
for _ in tqdm(range(15)):
    tLoss = []
    vLoss = []
    dTrain.on_epoch_end()
    dVal.on_epoch_end()
    for t in range(dTrain.__len__() -1):
        dT = dTrain.__getitem__(t)
        y_true = processInput(dT[0])
        x_true = processInput((dT[0]*dT[1]))
        tLoss.append(model.train_on_batch(x_true, y_true))
    
    for v in range(dVal.__len__() -1):
        dV = dVal.__getitem__(v)
        y_true = processInput(dV[0])
        x_true = processInput((dV[0]*dV[1]))
        vLoss.append(model.test_on_batch(x_true, y_true))
        
    print(np.mean(tLoss), np.mean(vLoss))
           
    trainList.append(np.mean(tLoss))
    valList.append(np.mean(vLoss))


model.save("unet.h5")

dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 4, 10, False)
d = dL.__getitem__(200)

def calcMetrics(model, mName):
    trueSSIM = list()
    valSSIM  = list() 
    
    truePSNR = list()
    valPSNR = list()
    
    
    for i in tqdm(range(dL.__len__())):
        d = dL.__getitem__(i)

        y_true = processInput(d[0])
        x_true = processInput((d[0]*d[1]))
        y_pred = model(x_true)
        
        
        y_true = np.reshape(y_true, (256, 256, 1))
        x_true = np.reshape(x_true, (256, 256, 1))
        y_pred = np.reshape(y_pred, (256, 256, 1))
        
        trueSSIM.append(metrics.calcSSIM(y_true, x_true))
        valSSIM.append(metrics.calcSSIM(y_true, y_pred))
        
        truePSNR.append(metrics.calcPSNR(y_true, x_true))
        valPSNR.append(metrics.calcPSNR(y_true, y_pred))
        

    
    model_results = pd.DataFrame({"trueSSIM": trueSSIM, "valSSIM": valSSIM, "truePSNR": truePSNR,
                                  "valPSNR": valPSNR})
    
    model_results.to_csv(f"{mName}.csv", sep=",", header=True, encoding="UTF-8")
    
    
# calcMetrics(model, "unet")
    
    
    


# y_true = processInput(d[0])
# x_true = processInput((d[0]*d[1]))    
# plt.imshow(y_true[0, :, :, 0], cmap="gray")
# print("xxxx")
# plt.imshow(x_true[0, :, :, 0], cmap="gray")   

# pred = model.predict(x_true)
# plt.imshow(pred[0, :, :, 0], cmap="gray")

