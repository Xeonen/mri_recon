# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:15:22 2020

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

dVal = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 8, 4, 10, False)
dTrain = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Train", 8, 4, 10, False)

dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 4, 10, False)
d = dL.__getitem__(200)


dIn = layers.Input(shape=(256, 256, 2))
mIn = layers.Input(shape=(256, 256, 2))
mul = cl.mulBlock()([dIn, mIn])


# Dumbell Image Domain
ifft = cl.ifftBlock()(mul)
dumbell = cl.dumbellS()(ifft)

fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)


# Dumbell Image Domain
dumbell = cl.dumbellS()(ifft)
fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)

# Dumbell Image Domain
dumbell = cl.dumbellS()(ifft)
fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)

# Dumbell Image Domain
dumbell = cl.dumbellS()(ifft)
fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)


# Dumbell Image Domain
dumbell = cl.dumbellS()(ifft)
fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)

# Dumbell Image Domain
dumbell = cl.dumbellS()(ifft)
fft = cl.fftBlock()(dumbell)
con = cl.conBlock()([mul, mIn, fft])



# Model Output
ifft = cl.ifftBlock()(con)



model = models.Model(inputs = [dIn, mIn], outputs = [ifft])
model.compile(loss="MSE", optimizer="Adam")

print(model.summary())


################ Train with Phase
ifftConv = hf.ifftModel()
a = ifftConv(d[0])


trainList = list()
valList = list()
minLoss = 0
oldminLoss = 0
minEpoch = 0
for i in tqdm(range(20)):
    tLoss = []
    vLoss = []
    dTrain.on_epoch_end()
    dVal.on_epoch_end()
    for t in range(dTrain.__len__() -1):
        dT = dTrain.__getitem__(t)
        y = ifftConv(dT[0])
        tLoss.append(model.train_on_batch(dT, y))
    
    for v in range(dVal.__len__() -1):
        dV = dVal.__getitem__(v)
        y = ifftConv(dV[0])
        vLoss.append(model.test_on_batch(dV, y))
        
    print(np.mean(tLoss), np.mean(vLoss))
    
    if minLoss == 0:
        minLoss = np.mean(vLoss)
    elif minLoss > np.mean(vLoss):
        oldminLoss = minLoss
        minLoss = np.mean(vLoss)
        minEpoch = i
        model.save("RCC_DC_Small_5x.h5")
        print(f"Loss improved from {np.round(oldminLoss, 6)} to {np.round(minLoss, 6)} at {i}th epoch")
                  
    trainList.append(np.mean(tLoss))
    valList.append(np.mean(vLoss))



df = pd.DataFrame({"trainLoss": trainList, "valLoss": valList})
df.to_csv("RCC_DC_Small_5x.csv", sep=",", header=True, encoding="UTF-8")



dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 4, 10, False)
d = dL.__getitem__(200)

out = model.predict(d)

out = np.reshape(out, (256, 256, 2))
out = out[:, :, 0] + 1j*out[:, :, 1]

plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.abs(out),cmap='gray')
plt.title('Magnitude')
plt.colorbar(orientation='horizontal',shrink=0.9)
plt.subplot(1,2,2)
plt.imshow(np.angle(out),cmap='gray')
plt.title('Phase')
plt.colorbar(orientation='horizontal',shrink=0.9)
plt.show()
