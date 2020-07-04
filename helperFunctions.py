# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:07:44 2020

@author: Xeonen
"""
import metrics
import customLayers as cl

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

ssim = tf.image.ssim_multiscale
ifft2d = tf.signal.ifft2d
fft2d = tf.signal.fft2d



def fftModel():
    dIn = layers.Input(shape=(256, 256, 2))
    fft = cl.fftBlock()(dIn)
    model = models.Model(inputs=[dIn], outputs=[fft])   
    return(fft)

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

def normalize(arr):
    minVal = np.min(arr)
    maxVal = np.max(arr)
    arr = (arr - minVal) / (maxVal - minVal)
    return(arr)

def castComplex(arr):
    real = arr[:, :, 0]
    imag = arr[:, :, 1]
    complexArray = tf.complex(real, imag)
    return(complexArray)

def castNormal(arr):
    real = arr.numpy().real
    imag = arr.numpy().imag
    normalArray = np.stack((real, imag), axis=-1)
    return(normalArray)


def gen_img_from_complex_array(arr, title="Complex Array"):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, dpi=240)
    fig.patch.set_visible(False)

    
    ax[0].set_title("Magnitude")
    ax[0].imshow(np.abs(arr), cmap="gray")
    ax[0].axis("off")
    
    ax[1].set_title("Phase")
    ax[1].imshow(np.angle(arr), cmap="gray")
    ax[1].axis("off")
    
    fig.suptitle(title)


def prepare_array(arr):
    arr = np.log(1+np.abs(arr))
    return(arr)


def gen_img(arr, title="Non-Complex Array"): 
    
    arr = tf.complex(arr[:, :, :, 0], arr[:, :, :, 1])
    
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, dpi=240)
    fig.patch.set_visible(False)
    
    fig.suptitle(title)
    
    ax[0].set_title("Magnitude")
    ax[0].imshow(np.abs(arr), cmap="gray")
    ax[0].axis("off")
    
    ax[1].set_title("Phase")
    ax[1].imshow(np.angle(arr), cmap="gray")
    ax[1].axis("off")



def gen_img_imgDomain(arr, title="Image Output"):
    arr = tf.reshape(arr, (256, 256))
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, dpi=240)
    ax.set_title(title)
    img = ax.imshow(arr, cmap="gray")
    fig.colorbar(img, orientation="vertical", shrink=0.9, ax=ax)
    ax.axis("off")
    




ifft = ifftModel()
underSampler = inputModel()

def getStats(model, d):
   
    x_true = ifft(underSampler(d))[0]
    y_true = ifft(d[0])[0]
    y_pred = model(d)[0]

    
    x_true = castComplex(x_true)
    y_true = castComplex(y_true)
    y_pred = castComplex(y_pred)
    y_error = np.subtract(y_pred, y_true)
    
    
    y_true_metric = np.reshape(np.abs(y_true), (256, 256, 1))
    x_true_metric = np.reshape(np.abs(x_true), (256, 256, 1))
    y_pred_metric = np.reshape(np.abs(y_pred), (256, 256, 1))
    

    
    trueSNR = metrics.calcPSNR(y_true_metric, x_true_metric)
    trueSSIM = metrics.calcSSIM(y_true_metric, x_true_metric)
    
    predSNR = metrics.calcPSNR(y_true_metric, y_pred_metric)
    predSSIM = metrics.calcSSIM(y_true_metric, y_pred_metric)
    
    
    # trueIGL = np.mean(metrics.image_gradient_loss(y_true, x_true).numpy()[0])
    # predIGL = np.mean(metrics.image_gradient_loss(y_true, y_pred).numpy()[0])
    

    # y_true = np.resize(y_true, (256, 256))
    # y_pred = np.resize(y_pred, (256, 256))
    # x_true = np.resize(x_true, (256, 256))  
    # y_error = np.abs(y_true - y_pred)
    # y_error = y_error  / np.max(y_error)
    # y_error = np.resize(y_error, (256, 256))
    
    img_dict = {"y_true": y_true, "y_pred": y_pred, "x_true": x_true, "y_error": y_error}
    stat_dict = {"refSNR": trueSNR, "refSSIM": trueSSIM,
                 "modelSNR": predSNR, "modelSSIM": predSSIM}
    

    return(img_dict, stat_dict)


def gen_comparison_graph(img_dict):
      
    
    # "y_true": y_true, "y_pred": y_pred, "x_true": x_true, "y_error": y_error
    
    y_true = img_dict["y_true"]
    y_pred = img_dict["y_pred"]
    x_true = img_dict["x_true"]
    y_error = img_dict["y_error"]


    fig, ax = plt.subplots(nrows=2, ncols=4, constrained_layout=True, dpi=320)
    fig.patch.set_visible(False)
    
    
    ax[0][0].set_title("US Input")
    uIM = ax[0][0].imshow(np.abs(x_true), cmap = "gray")
    fig.colorbar(uIM, orientation='horizontal', shrink=0.9, ax=ax[0][0])
    ax[0][0].axis("off")
    
    ax[0][1].set_title("Output")
    mOM = ax[0][1].imshow(np.abs(y_pred), cmap = "gray")
    fig.colorbar(mOM, orientation='horizontal', shrink=0.9, ax=ax[0][1])
    ax[0][1].axis("off")
    
    
    ax[0][2].set_title("FS Img")
    fSM = ax[0][2].imshow(np.abs(y_true) ,cmap = "gray")
    fig.colorbar(fSM, orientation='horizontal', shrink=0.9, ax=ax[0][2])
    ax[0][2].axis("off")
    
    ax[0][3].set_title("Error")
    fEM = ax[0][3].imshow(np.abs(y_error) ,cmap = "magma")
    fig.colorbar(fEM, orientation='horizontal', shrink=0.9, ax=ax[0][3])
    ax[0][3].axis("off")
       
    
    ax[1][0].set_title("US Input")
    uIP = ax[1][0].imshow(np.angle(x_true), cmap = "gray")
    fig.colorbar(uIP, orientation='horizontal', shrink=0.9, ax=ax[1][0])
    ax[1][0].axis("off")
    
    ax[1][1].set_title("Output")
    mOP = ax[1][1].imshow(np.angle(y_pred), cmap = "gray")
    fig.colorbar(mOP, orientation='horizontal', shrink=0.9, ax=ax[1][1])
    ax[1][1].axis("off")
    
    
    ax[1][2].set_title("FS Img")
    fSP = ax[1][2].imshow(np.angle(y_true) ,cmap = "gray")
    fig.colorbar(fSP, orientation='horizontal', shrink=0.9, ax=ax[1][2])
    ax[1][2].axis("off")
    
    ax[1][3].set_title("Error")
    fEP = ax[1][3].imshow(np.angle(y_error) ,cmap = "magma")
    fig.colorbar(fEP, orientation='horizontal', shrink=0.9, ax=ax[1][3])
    ax[1][3].axis("off")
    

    plt.show()

