# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:04:26 2020

@author: Xeonen
"""



import dataOp
import metrics
import customLayers as cl
import helperFunctions as hf

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, losses

dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 8, 15, False)
d = dL.__getitem__(200)


mask = tf.concat((d[1], d[1]), axis=-1)

under = d[0]*mask

img = tf.complex(under[0, :, :, 0], under[0, :, :, 1])

img = np.fft.ifft2(img.numpy())










# y_pred = cl.mulBlock()(d)

# img = tf.reshape(y_pred, (256, 256, 2))
# img = tf.complex(img[:, :, 0], img[:, :, 1])


plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.abs(img),cmap='gray')
plt.title('Magnitude')
plt.colorbar(orientation='horizontal',shrink=0.9)
plt.subplot(1,2,2)
plt.imshow(np.angle(img),cmap='gray')
plt.title('Phase')
plt.colorbar(orientation='horizontal',shrink=0.9)
plt.show()

# img = tf.signal.ifft2d(img)

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