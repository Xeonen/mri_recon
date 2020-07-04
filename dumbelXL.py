# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 05:57:40 2020

@author: Xeonen
"""
import customLayers as cl
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, activations

def dumbelXL():

    inL = layers.Input(shape=(256, 256, 2))
    
    conv01 = cl.convBlock(32, 3)(inL)
    conv02 = cl.convBlock(32, 3)(conv01)
    sConv01 = cl.convBlock(64, 3, 2)(conv02)
    
    conv03 = cl.convBlock(64, 3)(sConv01)
    conv04 = cl.convBlock(64, 3)(conv03)
    sConv02 = cl.convBlock(128, 3, 2)(conv04)
    
    conv05 = cl.convBlock(128, 3)(sConv02)
    conv06 = cl.convBlock(128, 3)(conv05)
    sConv03 = cl.convBlock(256, 3, 2)(conv06)
    
    conv07 = cl.convBlock(256, 3)(sConv03)
    conv08 = cl.convBlock(256, 3)(conv07)
    sConv04 = cl.convBlock(512, 3, 2)(conv08)



    
    res01 = cl.resBlock(512, 3)(sConv04)
    res02 = cl.resBlock(512, 3)(res01)
    res03 = cl.resBlock(512, 3)(res02)
    res04 = cl.resBlock(512, 3)(res03)
    
    
    add01 = layers.Add()([sConv04, res04])
    tConv01 = cl.convTransBlock(512, 3, 2)(add01)
    conv09 = cl.convBlock(256, 3)(tConv01)
    conv10 = cl.convBlock(256, 3)(conv09)
    
    add02 = layers.Add()([conv08, conv10])
    tConv02 = cl.convTransBlock(256, 3, 2)(conv10)
    conv11 = cl.convBlock(128, 3)(tConv02)
    conv12 = cl.convBlock(128, 3)(conv11)
    
    
    add03 = layers.Add()([conv06, conv12])
    tConv03 = cl.convTransBlock(128, 3, 2)(add03)
    conv13 = cl.convBlock(64, 3)(tConv03)
    conv14 = cl.convBlock(64, 3)(conv13)
    
    add04 = layers.Add()([conv04, conv14])
    tConv04 = cl.convTransBlock(64, 3, 2)(add04)
    conv15 = cl.convBlock(32, 3)(tConv04)
    conv16 = cl.convBlock(32, 3)(conv15)

    synt = layers.Conv2D(2, 3, padding="same", use_bias=False)(conv16)

    
    
    model = models.Model(inputs=[inL], outputs=[synt])
    
    return(model)


m = dumbelXL()
print(m.summary())
