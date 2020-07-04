# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 05:57:40 2020

@author: Xeonen
"""

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, activations

def get_unet(k):
    # 256 + k for k = [204, 220, 236, 252]
    
    a = 256 + k
    inL = layers.Input(shape=(a, a, 2))
    conv01 = layers.Conv2D(64, 3, padding="same", activation="relu")(inL)
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
    
    
    model = models.Model(inputs=[inL], outputs=[conv19])
    
    return(model)


m = get_unet(0)
print(m.summary())

# x = list()
# for i in tqdm(range(200, 256)):
#     try:
#         _ = get_unet(k= i)
#         x.append(i)
#         print(i)
#     except:
#         continue

# print("Done")
    