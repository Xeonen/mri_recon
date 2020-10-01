# -*- coding: utf-8 -*-
"""
Has a longer residual block

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
    conv02 = cl.convBlock(32, 3)(conv02)
    sConv01 = cl.convBlock(64, 3, 2)(conv02)
    
    conv03 = cl.convBlock(64, 3)(sConv01)
    conv04 = cl.convBlock(64, 3)(conv03)
    conv04 = cl.convBlock(64, 3)(conv04)
    sConv02 = cl.convBlock(128, 3, 2)(conv04)
    
    conv05 = cl.convBlock(128, 3)(sConv02)
    conv06 = cl.convBlock(128, 3)(conv05)
    conv06 = cl.convBlock(128, 3)(conv06)
    sConv03 = cl.convBlock(256, 3, 2)(conv06)
    
    conv07 = cl.convBlock(256, 3)(sConv03)
    conv08 = cl.convBlock(256, 3)(conv07)
    conv08 = cl.convBlock(256, 3)(conv08)
    sConv04 = cl.convBlock(512, 3, 2)(conv08)



    
    res01 = cl.resBlock(512, 3)(sConv04)
    res02 = cl.resBlock(512, 3)(res01)
    res03 = cl.resBlock(512, 3)(res02)
    res04 = cl.resBlock(512, 3)(res03)
    res05 = cl.resBlock(512, 3)(res04)
    res06 = cl.resBlock(512, 3)(res05)
    
    

    tConv01 = cl.convTransBlock(256, 3, 2)(res06)
    add01 = layers.Add()([tConv01, conv08])
    conv09 = cl.convBlock(256, 3)(add01)
    conv09 = cl.convBlock(256, 3)(conv09)
    conv10 = cl.convBlock(256, 3)(conv09)
    
    tConv02 = cl.convTransBlock(128, 3, 2)(conv10)
    add02 = layers.Add()([tConv02, conv06])
    conv11 = cl.convBlock(128, 3)(add02)
    conv11 = cl.convBlock(128, 3)(conv11)
    conv12 = cl.convBlock(128, 3)(conv11)
    
    

    tConv03 = cl.convTransBlock(64, 3, 2)(conv12)
    add03 = layers.Add()([tConv03, conv04])
    conv13 = cl.convBlock(64, 3)(add03)
    conv13 = cl.convBlock(64, 3)(conv13)
    conv14 = cl.convBlock(64, 3)(conv13)
    

    tConv04 = cl.convTransBlock(32, 3, 2)(conv14)
    add04 = layers.Add()([tConv04, conv02])
    conv15 = cl.convBlock(32, 3)(add04)
    conv15 = cl.convBlock(32, 3)(conv15)
    conv16 = cl.convBlock(32, 3)(conv15)

    synt = layers.Conv2D(2, 3, padding="same", use_bias=False)(conv16)

    
    
    model = models.Model(inputs=[inL], outputs=[synt])
    
    return(model)


m = dumbelXL()
print(m.summary())
