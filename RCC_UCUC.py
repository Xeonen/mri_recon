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







conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ifft)
conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = layers.Dropout(0.5)(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = layers.Dropout(0.5)(conv5)


us01 = layers.UpSampling2D(size = (2,2))(drop5)
up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us01)
merge6 = layers.concatenate([drop4,up6], axis = 3)
conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)


us02 = layers.UpSampling2D(size = (2,2))(conv6)
up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us02)
merge7 = layers.concatenate([conv3,up7], axis = 3)
conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


us03 = layers.UpSampling2D(size = (2,2))(conv7)
up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us03)
merge8 = layers.concatenate([conv2,up8], axis = 3)
conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)


us04 = layers.UpSampling2D(size = (2,2))(conv8)
up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us04)
merge9 = layers.concatenate([conv1,up9], axis = 3)
conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = layers.Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)







fft = cl.fftBlock()(conv9)
con = cl.conBlock()([mul, mIn, fft])
ifft = cl.ifftBlock()(con)



# Dumbell Image Domain
conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ifft)
conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = layers.Dropout(0.5)(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = layers.Dropout(0.5)(conv5)


us01 = layers.UpSampling2D(size = (2,2))(drop5)
up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us01)
merge6 = layers.concatenate([drop4,up6], axis = 3)
conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)


us02 = layers.UpSampling2D(size = (2,2))(conv6)
up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us02)
merge7 = layers.concatenate([conv3,up7], axis = 3)
conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


us03 = layers.UpSampling2D(size = (2,2))(conv7)
up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us03)
merge8 = layers.concatenate([conv2,up8], axis = 3)
conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)


us04 = layers.UpSampling2D(size = (2,2))(conv8)
up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(us04)
merge9 = layers.concatenate([conv1,up9], axis = 3)
conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = layers.Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)










fft = cl.fftBlock()(conv9)
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
        
    
    
    if minLoss == 0:
        minLoss = np.mean(vLoss)
    elif minLoss > np.mean(vLoss):
        oldminLoss = minLoss
        minLoss = np.mean(vLoss)
        minEpoch = i
        model.save("RCC_UCUC.h5")
        print(f"Loss improved from {np.round(oldminLoss, 6)} to {np.round(minLoss, 6)} at {i}th epoch")
        
    print(np.mean(tLoss), np.mean(vLoss), minLoss, oldminLoss)
                  
    trainList.append(np.mean(tLoss))
    valList.append(np.mean(vLoss))



# df = pd.DataFrame({"trainLoss": trainList, "valLoss": valList})
# df.to_csv("RCC_UCUC.csv", sep=",", header=True, encoding="UTF-8")



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


