# -*- coding: utf-8 -*-
"""
@author: Xeonen
"""


import pandas as pd
import tensorflow as tf
import numpy as np

from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.losses import MSE
from tensorflow.keras import layers


ssim_multiscale = tf.image.ssim_multiscale
psnr = tf.image.psnr 


def calcPSNR(y_true, y_pred):
    # dims are batch, x, y, channel
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    maxVal = tf.reduce_max([tf.reduce_max(y_true), tf.reduce_max(y_pred)])
    minVal = tf.reduce_min([tf.reduce_min(y_true), tf.reduce_min(y_pred)])
    dynamicRange = tf.math.subtract(maxVal, minVal)
    try:
        snr = psnr(y_true, y_pred, max_val=dynamicRange).numpy()
    except Exception as e:
        snr = psnr(y_true, y_pred, max_val=dynamicRange)
        # print(e)       
    return(snr)



def calcSSIM(y_true, y_pred):
    # dims are batch, x, y, channel
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    maxVal = tf.reduce_max([tf.reduce_max(y_true), tf.reduce_max(y_pred)])
    minVal = tf.reduce_min([tf.reduce_min(y_true), tf.reduce_min(y_pred)])
    dynamicRange = tf.math.subtract(maxVal, minVal)
    try:
        ssim = ssim_multiscale(y_true, y_pred, max_val=dynamicRange).numpy()
    except Exception as e:
        ssim = ssim_multiscale(y_true, y_pred, max_val=dynamicRange)
        # print(e)
    return(ssim)

def PSNR_loss(y_true, y_pred):
    return(-1*calcPSNR(y_true, y_pred))
  # return tf.reduce_mean(-1*calcPSNR(y_true, y_pred))


def SSIM_loss(y_true, y_pred):
  return tf.reduce_mean(1 - calcSSIM(y_true, y_pred))

def image_gradient_loss(y_true, y_pred):
    # dims are batch, x, y, channel
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    
    y_true = tf.math.abs(tf.image.image_gradients(y_true))
    y_pred = tf.math.abs(tf.image.image_gradients(y_pred))
    
    igd = tf.math.subtract(y_true, y_pred)
    igd = tf.math.abs(igd)
    
    # true_grad_Y = y_true[0]
    # true_grad_X = y_true[1]
    
    # pred_grad_Y = y_pred[0]
    # pred_grad_X = y_pred[1]

    # y_difference = tf.math.abs(tf.math.subtract(true_grad_Y, pred_grad_Y))
    # x_difference = tf.math.abs(tf.math.subtract(true_grad_X, pred_grad_X))
    
    # sum_loss = tf.math.add(y_difference, x_difference)


    return(igd)



    # loss = tf.reduce_mean(sum_loss)












# @tf.function
# def image_gradient_loss(y_true, y_pred):
#     # dims are batch, x, y, channel
#     y_true = tf.convert_to_tensor(y_true)
#     y_pred = tf.convert_to_tensor(y_pred)
    
#     true_grad_Y, true_grad_X = tf.math.abs(tf.image.image_gradients(y_true))
#     pred_grad_Y, pred_grad_X = tf.math.abs(tf.image.image_gradients(y_pred))
    
#     y_difference = tf.math.abs(tf.math.subtract(true_grad_Y, pred_grad_Y))
#     x_difference = tf.math.abs(tf.math.subtract(true_grad_X, pred_grad_X))
    
#     sum_loss = tf.math.add(y_difference, x_difference)
#     loss = tf.reduce_mean(sum_loss)

#     return(loss)
