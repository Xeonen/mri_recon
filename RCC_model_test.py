# -*- coding: utf-8 -*-
"""
Created on Sat May 30 06:38:58 2020

@author: Xeonen
"""


import metrics
import dataOp
import customLayers as cl
import helperFunctions as hf

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, losses


import matplotlib.pyplot as plt
from tqdm import tqdm

ssim = tf.image.ssim_multiscale
ifft2d = tf.signal.ifft2d
fft2d = tf.signal.fft2d


def fftModel():
    dIn = layers.Input(shape=(256, 256, 2))
    fft = cl.fftBlock()(dIn)
    model = models.Model(inputs=[dIn], outputs=[fft])   
    return(model)

def ifftModel():
    dIn = layers.Input(shape=(256, 256, 2))
    ifft = cl.ifftBlock()(dIn)
    model = models.Model(inputs=[dIn], outputs=[ifft])      
    return(model)


def inputModel():
    dIn = layers.Input(shape=(256, 256, 2))
    mIn = layers.Input(shape=(256, 256, 2))
    mul = cl.mulBlock()([dIn, mIn])
    model = models.Model(inputs=[dIn, mIn], outputs=[mul])   
    return(model)


ifft = ifftModel()
underSampler = inputModel()


def image_gradient_loss(y_true, y_pred):
    true_grad = tf.image.image_gradients(y_true)
    true_grad = tf.math.abs(true_grad)
    pred_grad = tf.image.image_gradients(y_pred)
    pred_grad = tf.math.abs(pred_grad)
    
    loss = tf.math.subtract(true_grad, pred_grad)
    loss = tf.math.abs(loss)
    lossA, lossB = tf.split(loss, 2, axis=0)
    loss = tf.add(lossA, lossB)
    
    return(loss)






custom_objects = {"mulBlock": cl.mulBlock, "ifftBlock": cl.ifftBlock,
                  "dumbellS": cl.dumbellS, "fftBlock": cl.fftBlock,
                  "conBlock": cl.conBlock, "resBlock": cl.resBlock
    }

model = models.load_model("RCC_DC_Small_6x.h5", custom_objects=custom_objects)

dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 4, 10, False)
d = dL.__getitem__(200)


img_dict, stat_dict = hf.getStats(model, d)
print(stat_dict)
hf.gen_comparison_graph(img_dict)





def calcMetrics(model, mType, mName):
    
    fft = fftModel()
    ifft = ifftModel()
    underSampler = inputModel()
    
    trueSSIM = list()
    valSSIM  = list() 
    
    truePSNR = list()
    valPSNR = list()
    
    trueIGL = list()
    valIGL = list()
    
    for i in tqdm(range(dL.__len__())):
        d = dL.__getitem__(i)

        x_true = ifft(underSampler(d))[0]
        y_true = ifft(d[0])[0]
        y_pred = model(d)[0]
        
        if mType == "magnitude":
            under = np.abs(hf.castComplex(x_true))
            y_true = np.abs(hf.castComplex(y_true))
            y_pred = np.abs(hf.castComplex(y_pred))
        elif mType == "phase":
            under = np.angle(hf.castComplex(x_true))
            y_true = np.angle(hf.castComplex(y_true))
            y_pred = np.angle(hf.castComplex(y_pred))
        else:
            print("Choose magnitude or phase as mType")
            pass
        
        y_true = np.reshape(y_true, (256, 256, 1))
        x_true = np.reshape(under, (256, 256, 1))
        y_pred = np.reshape(y_pred, (256, 256, 1))
        
        trueSSIM.append(metrics.calcSSIM(y_true, x_true))
        valSSIM.append(metrics.calcSSIM(y_true, y_pred))
        
        truePSNR.append(metrics.calcPSNR(y_true, x_true))
        valPSNR.append(metrics.calcPSNR(y_true, y_pred))
        
        
        y_true = np.reshape(y_true, (1, 256, 256, 1))
        x_true = np.reshape(under, (1, 256, 256, 1))
        y_pred = np.reshape(y_pred, (1, 256, 256, 1))
        

       
    
    model_results = pd.DataFrame({"trueSSIM": trueSSIM, "valSSIM": valSSIM, "truePSNR": truePSNR,
                                  "valPSNR": valPSNR})
    
    model_results.to_csv(f"{mName}.csv", sep=",", header=True, encoding="UTF-8")
    
calcMetrics(model, "magnitude", "RCC_DC_Small_6x_Mag_Results")
calcMetrics(model, "phase", "RCC_DC_Small_6x_Phase_Results")

