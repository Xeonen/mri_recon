# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:13:14 2020

@author: Xeonen
"""

import tensorflow as tf
import numpy as np
import dataOp
import  matplotlib.pyplot as plt


dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 8, 15, False)
y, x = dL.__getitem__(200)

y = np.reshape(y, (256, 256, 2))
x = np.reshape(x, (256, 256))


img = np.fft.ifft2(np.squeeze(y[:,:,0])+1j*np.squeeze(y[:,:,1]))

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



img = tf.signal.ifft2d(tf.complex(y[:, :, 0], y[:, :, 1]))

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


comp = y[:,:,0]+1j*y[:,:,1]
compMask = x + 1j*x
comp = comp * compMask
img = np.fft.ifft2(comp)

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