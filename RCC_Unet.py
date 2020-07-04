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







conv01 = layers.Conv2D(64, 3, padding="same", activation="relu")(ifft)
conv02 = layers.Conv2D(64, 3, padding="same", activation="relu")(conv01) # Add copy and crop
maxP01 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv02)
conv03 = layers.Conv2D(128, 3, padding="same", activation="relu")(maxP01)
conv04 = layers.Conv2D(128, 3, padding="same", activation="relu")(conv03) # Add copy and crop using crop03
maxP02 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv04)
conv05 = layers.Conv2D(256, 3, padding="same", activation="relu")(maxP02)
conv06 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv05) # Add copy and crop using crop02
maxP03 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv06)
conv07 = layers.Conv2D(512, 3, padding="same", activation="relu")(maxP03)
conv08 = layers.Conv2D(512, 3, padding="same", activation="relu")(conv07) # Copy and crop using crop01
maxP04 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv08)
conv09 = layers.Conv2D(1024, 3, padding="same", activation="relu")(maxP04)
conv10 = layers.Conv2D(1024, 3, padding="same", activation="relu")(conv09)

tConv01 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(conv10)
concat01 = layers.Concatenate()([tConv01, conv08])

conv11 = layers.Conv2D(512, 3, padding="same", activation="relu")(concat01)
conv12 = layers.Conv2D(512, 3, padding="same", activation="relu")(conv11)

tConv02 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv12)
concat02 = layers.Concatenate()([tConv02, conv06])  

conv13 = layers.Conv2D(256, 3, padding="same", activation="relu")(concat02)
conv14 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv13)

tConv03 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(conv14)
concat03 = layers.Concatenate()([tConv03, conv04])  

conv15 = layers.Conv2D(128, 3, padding="same", activation="relu")(concat03)
conv16 = layers.Conv2D(128, 3, padding="same", activation="relu")(conv15)

tConv04 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(conv16)
concat04 = layers.Concatenate()([tConv04, conv02])

conv17 = layers.Conv2D(64, 3, padding="same", activation="relu")(concat04)
conv18 = layers.Conv2D(64, 3, padding="same", activation="relu")(conv17)
conv19 = layers.Conv2D(2, 1, padding="same", activation="relu")(conv18)    






fft = cl.fftBlock()(conv18)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)



# Dumbell Image Domain
conv01 = layers.Conv2D(64, 3, padding="same", activation="relu")(ifft)
conv02 = layers.Conv2D(64, 3, padding="same", activation="relu")(conv01) # Add copy and crop
maxP01 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv02)
conv03 = layers.Conv2D(128, 3, padding="same", activation="relu")(maxP01)
conv04 = layers.Conv2D(128, 3, padding="same", activation="relu")(conv03) # Add copy and crop using crop03
maxP02 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv04)
conv05 = layers.Conv2D(256, 3, padding="same", activation="relu")(maxP02)
conv06 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv05) # Add copy and crop using crop02
maxP03 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv06)
conv07 = layers.Conv2D(512, 3, padding="same", activation="relu")(maxP03)
conv08 = layers.Conv2D(512, 3, padding="same", activation="relu")(conv07) # Copy and crop using crop01
maxP04 = layers.MaxPool2D(pool_size = 2, strides= 2)(conv08)
conv09 = layers.Conv2D(1024, 3, padding="same", activation="relu")(maxP04)
conv10 = layers.Conv2D(1024, 3, padding="same", activation="relu")(conv09)

tConv01 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(conv10)
concat01 = layers.Concatenate()([tConv01, conv08])

conv11 = layers.Conv2D(512, 3, padding="same", activation="relu")(concat01)
conv12 = layers.Conv2D(512, 3, padding="same", activation="relu")(conv11)

tConv02 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv12)
concat02 = layers.Concatenate()([tConv02, conv06])  

conv13 = layers.Conv2D(256, 3, padding="same", activation="relu")(concat02)
conv14 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv13)

tConv03 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(conv14)
concat03 = layers.Concatenate()([tConv03, conv04])  

conv15 = layers.Conv2D(128, 3, padding="same", activation="relu")(concat03)
conv16 = layers.Conv2D(128, 3, padding="same", activation="relu")(conv15)

tConv04 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(conv16)
concat04 = layers.Concatenate()([tConv04, conv02])

conv17 = layers.Conv2D(64, 3, padding="same", activation="relu")(concat04)
conv18 = layers.Conv2D(64, 3, padding="same", activation="relu")(conv17)
conv19 = layers.Conv2D(2, 1, padding="same", activation="relu")(conv18)    



fft = cl.fftBlock()(conv19)
con = cl.conBlock()([mul, mIn, fft])


# Model Output
ifft = cl.ifftBlock()(con)



model = models.Model(inputs = [dIn, mIn], outputs = [ifft])
model.compile(loss="MSE", optimizer="Adam")




################# Train with Phase
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
        model.save("RCC_UCUC.h5")
        print(f"Loss improved from {np.round(oldminLoss, 6)} to {np.round(minLoss, 6)} at {i}th epoch")
                  
    trainList.append(np.mean(tLoss))
    valList.append(np.mean(vLoss))



df = pd.DataFrame({"trainLoss": trainList, "valLoss": valList})
df.to_csv("RCC_UCUC.csv", sep=",", header=True, encoding="UTF-8")



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



# mulOut = cl.mulBlock()(d)
# outC = cl.fftBlock()(mulOut)

# conOut = cl.conBlock()([mulOut, d[1], outC])
# conOut = cl.ifftBlock()(conOut)
# out = model.predict(d)
# img = hf.castComplex(out[0])

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(np.abs(img),cmap='gray')
# plt.title('Magnitude')
# plt.colorbar(orientation='horizontal',shrink=0.9)
# plt.subplot(1,2,2)
# plt.imshow(np.angle(img),cmap='gray')
# plt.title('Phase')
# plt.colorbar(orientation='horizontal',shrink=0.9)
# plt.show()


