# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:20:51 2020

@author: Xeonen
"""
import dataOp
import metrics
import customLayers as cl
import helperFunctions as hf
import dataGen as dg

import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob

from tqdm import tqdm
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt

def trainOp(model, optimizer, dLoader, it, train=True):
    data = dLoader.__getitem__(it)
    with tf.GradientTape() as tape:
        y_true, y_pred = model(data, training=train)
        loss = losses.MSE(y_true, y_pred)

    if train:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return(loss)

dL = layers.Input(shape=(218, 170, 24))
dM = layers.Input(shape=(218, 170, 24))

dLZ = layers.ZeroPadding2D(padding=((3, 3)))(dL)
dMZ = layers.ZeroPadding2D(padding=((3, 3)))(dM)

mul = cl.mulBlock()([dLZ, dMZ])

sConv01 = cl.convBlock(32, 3, 4)(mul)
sConv02 = cl.convBlock(128, 3, 2)(sConv01)
sConv03 = cl.convBlock(256, 3, 2)(sConv02)

res01 = cl.resBlock(256, 3)(sConv03)
res02 = cl.resBlock(256, 3)(res01)


tConv01 = cl.convTransBlock(128, 3, 2)(res02)
add = layers.Add()([tConv01, sConv02])
tConv02 = cl.convTransBlock(64, 3, 2)(add)
tConv03 = cl.convTransBlock(24, 3, 4)(tConv02)

synt = layers.Conv2D(24, 3, padding="same")(tConv03)

con = cl.conBlock()([mul, dMZ, synt])

ifft = cl.ifftBlock()(con)

sConv01 = cl.convBlock(32, 3, 4)(ifft)
sConv02 = cl.convBlock(128, 3, 2)(sConv01)
sConv03 = cl.convBlock(256, 3, 2)(sConv02)

res01 = cl.resBlock(256, 3)(sConv03)
res02 = cl.resBlock(256, 3)(res01)

tConv01 = cl.convTransBlock(128, 3, 2)(res02)
add = layers.Add()([tConv01, sConv02])
tConv02 = cl.convTransBlock(64, 3, 2)(add)
tConv03 = cl.convTransBlock(24, 3, 4)(tConv02)

synt = layers.Conv2D(24, 3, padding="same")(tConv03)



fft = cl.fftBlock()(synt)
con = cl.conBlock()([mul, dMZ, fft])

trimmedCon = layers.Cropping2D(cropping=((3, 3), (3, 3)))(con)

underChan = cl.fullChanBlock()(trimmedCon)
fullChan = cl.fullChanBlock()(dL)

model = models.Model(inputs=[dL, dM], outputs=[fullChan, underChan])
optimizer = optimizers.Adam()

train = glob(r"C:/Datasets/*.h5")
val = glob(r"C:/Datasets/*.h5")
R = 4 # *2
sample_n = 32 # //2 # 'calibration center' size
random = False # only applicable for uniform = False, really...
uniform = False
centered = False
dim = (218,170) # input dimensions
batch_size = 32
n_channels = 12*2 # 12-channels*2 (real and imaginary)
nslices = 256
crop = (30,30) # Crops slices with little anatomy???

tLoader = dg.DataGenerator(train, dim, R, sample_n, crop, batch_size, n_channels,
             nslices, centered, uniform, shuffle=True)
vLoader = dg.DataGenerator(val, dim, R, sample_n, crop, batch_size, n_channels,
             nslices, centered, uniform, shuffle=True)

trainList = list()
valList = list()
for _ in tqdm(range(1)):
    tLoss = list()
    vLoss = list()
    tLoader.on_epoch_end()
    vLoader.on_epoch_end()
    
    for t in range(tLoader.__len__() -1):
        dTy = trainOp(model, optimizer, tLoader, t)
        tLoss.append(dTy)
        
    for v in range(vLoader.__len__() -1):
        dVy = trainOp(model, optimizer, tLoader, t, train=False)
        vLoss.append(dVy)
        
    print((np.mean(tLoss), np.mean(vLoss)))
           
    trainList.append(np.mean(tLoss))
    valList.append(np.mean(vLoss))


