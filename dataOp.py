# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:00:59 2020

@author: Xeonen
"""


import numpy as np
import tensorflow as tf

from glob import glob

import math
import mymath
from numpy.lib.stride_tricks import as_strided


ifft2d = tf.signal.ifft2d

class dataOp:
    def __init__(self):
        self.ifft2d = tf.signal.ifft2d
        self.fft2d = tf.signal.fft2d
        
    def cast_batch_complex(self, arr):
        real = arr[:, :, :, 0]
        imag = arr[:, :, :, 1]    
        complex_array = tf.complex(real, imag)       
        return(complex_array)
    
    def cast_batch_normal(self, arr):
        real = arr.numpy().real
        imag = arr.numpy().imag
        normal_array = tf.stack((real, imag))
        return(normal_array)

    def convert_to_img(self, arr):  
        arr = self.cast_batch_complex(arr)
        dImg = self.ifft2d(arr)
        dImg = tf.abs(dImg)
        dImg = tf.reshape(dImg, (-1, 256, 256, 1))        
        return(dImg)
    
    def under_sample(self, data, mask):
        undersampled = tf.math.multiply(data, mask)
        return(undersampled)
    
    def convert_to_underIMG(self, data, mask):
        under_k = self.under_sample(data, mask)
        under_img = self.convert_to_img(under_k)
        return(under_img)
    
    def fft(self, arr):
        complex_arr = self.cast_batch_complex(arr)
        transformed = self.fft2d(complex_arr)
        return(transformed)


class data_loader(tf.keras.utils.Sequence):
    def __init__(self, source, batch_size, acc, n_samples, centered, **kwargs):
        super().__init__(**kwargs)
        
        self.file_list = glob(f"{source}/*.npy")
        self.batch_size = batch_size
        self.maskGen = generate_mask((self.batch_size, 256, 256), acc, n_samples, centered)
        
    def __len__(self):
        return(math.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, idx):
        self.mask = self.get_mask()
        self.batch_y = self.file_list[idx * self.batch_size : (idx+1)*self.batch_size]
        
        self.img_y = np.array([np.load(file_name) for file_name in self.batch_y]).astype(np.float32)
        return(self.img_y, self.mask)
        
    def on_epoch_end(self):
         np.random.shuffle(self.file_list)
        
    def get_mask(self):
        mask = self.maskGen.produce_mask()
        return(mask)
        
        
        
class generate_mask:    
    
    def __init__(self, shape, acc, sample_n, centered):
        self.shape = shape
        self.acc = acc
        self.sample_n = sample_n
        self.centered = centered
        
    def produce_mask(self):
        mask = self.cartesian_mask(self.shape, self.acc, self.sample_n, self.centered).astype(np.float32)
        # mask = np.expand_dims(mask, -1)
        mask = np.stack([mask, mask], axis=3).astype(np.float32)
        return(mask)
    
    
    def normal_pdf(self, length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

    def cartesian_mask(self, shape, acc, sample_n=10, centred=False):
        """
        Sampling density estimated from implementation of kt FOCUSS
        shape: tuple - of form (..., nx, ny)
        acc: float - doesn't have to be integer 4, 8, etc..
        """
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
        pdf_x = self.normal_pdf(Nx, 0.5/(Nx/10.)**2)
        lmda = Nx/(2.*acc)
        n_lines = int(Nx / acc)
    
        # add uniform distribution
        pdf_x += lmda * 1./Nx
    
        if sample_n:
            pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
            pdf_x /= np.sum(pdf_x)
            n_lines -= sample_n
    
        mask = np.zeros((N, Nx))
        for i in range(N):
            idx = np.random.choice(Nx, n_lines, False, pdf_x)
            mask[i, idx] = 1
    
        if sample_n:
            mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
    
        size = mask.itemsize
        mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
    
        mask = mask.reshape(shape)
    
        if not centred:
            mask = mymath.ifftshift(mask, axes=(-1, -2))
    
        return mask


# dl = data_loader("C:/Datasets/MRI_Data/Recon_v4/Train", 4)
# a,b = dl.__getitem__(1)





