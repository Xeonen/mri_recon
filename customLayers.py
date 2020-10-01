# -*- coding: utf-8 -*-
"""
@author: Xeonen
"""



import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


ifft2d = tf.signal.ifft2d
fft2d = tf.signal.fft2d



class convBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides = 1, **kwargs):
        super().__init__(**kwargs)  
        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden = [
        layers.Conv2D(self.filters, self.kernel_size, strides= self.strides,
                      kernel_initializer='he_normal',
                      padding="same", kernel_regularizer=None, use_bias=True),
        layers.BatchNormalization(),
        layers.ReLU()
        ]
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'strides': self.strides,
            'hidden': self.hidden,
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config
 
        
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return(Z)
    
    
class resBlock(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs) 
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(self.filters, self.kernel_size, kernel_initializer='he_normal',
                      padding="same", kernel_regularizer=None, use_bias=True)
        self.batch = layers.BatchNormalization()
        self.activation = layers.ReLU()
        
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,

        })
        return config
 
        
    def call(self, inputs):
        Z = inputs
        conv01 = self.conv(inputs)
        batch01 = self.batch(conv01)
        act01 = self.activation(batch01)
        conv02 = self.conv(act01)
        batch02 = self.batch(conv02)
        add01 = batch02 + inputs
        Z = self.activation(add01)
        return(Z)

    


class convTransBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides=2, **kwargs):
        super().__init__(**kwargs) 
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.hidden = [
        layers.Conv2DTranspose(self.filters, self.kernel_size, kernel_initializer='he_normal',
                      padding="same", strides=self.strides, kernel_regularizer=None, use_bias=True),
        layers.BatchNormalization(),
        layers.ReLU()
        ]
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'strides': self.strides,
            'hidden': self.hidden,
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config
 
        
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return(Z)
 

# class fullChanBlock(layers.Layer):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)  
#         pass

       
#     def call(self, inputs):
#         data = inputs
      
#         # real = layers.Lambda(lambda Z : Z[:,:,:,0::2])(data)
#         # imag = layers.Lambda(lambda Z : Z[:,:,:,1::2])(data)
        
        
#         real = layers.Lambda(lambda Z : Z[:,:,:,0])(Z)
#         imag = layers.Lambda(lambda Z : Z[:,:,:,1])(Z)
        
#         complex_data = tf.complex(real, imag)
#         image_domain = tf.signal.ifft2d(complex_data)
        
#         Z = tf.math.abs(image_domain)
#         Z = tf.math.square(Z)
#         Z = tf.math.reduce_sum(Z, axis=-1)
#         Z =tf.math.sqrt(Z)

        # return(Z)


class combBlock(layers.layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.kConv = convBlock(self.filters, self.kernel_size)
        self.iConv = convBlock(self.filters, self.kernel_size)
        
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config
    
    def call(self, inputs):
        iIn = inputs[0]
        fIn = inputs[1]
        
        iAdd = fftBlock(iIn)
        fAdd = ifftBlock((fIn))
        
        i01 = iIn + fAdd
        f01 = fIn + iAdd
        
        i02 = self.iConv(i01)
        f02 = self.kConv(f01)
        
        iOut = iIn + i02
        fOut = fIn + f02
        
        return((iOut, fOut))
    


class mulBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        pass

       
    def call(self, inputs):
        data = inputs[0]
        mask = inputs[1]
        Z = tf.math.multiply(data, mask)
        return(Z)
    
class conBlock(layers.Layer):
    """
    First input is the undersampled data.
    Second input is the mask.
    Third input is the synthetic data.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        pass

       
    def call(self, inputs):
        undersampled_data = inputs[0]
        mask = inputs[1]
        synth_data = inputs[2]

        
        # Get complement mask, turn it to complex
        complement_mask = tf.math.logical_not(tf.cast(mask, tf.bool))
        complement_mask_float = tf.cast(complement_mask, tf.float32)        

  
        # Do the consistency
        inconmplete_data = tf.multiply(synth_data, complement_mask_float)           
        Z = layers.add([inconmplete_data, undersampled_data])
   
        return(Z)
    
     
class ifftBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        pass

       
    def call(self, inputs):
        Z = inputs
        
        real = layers.Lambda(lambda Z : Z[:,:,:,0])(Z)
        imag = layers.Lambda(lambda Z : Z[:,:,:,1])(Z)
        
        Z_complex = tf.complex(real, imag)
        
        image = ifft2d(Z_complex)

        
        real = tf.math.real(image)
        imag = tf.math.imag(image)
       

        
        Z = tf.stack((real, imag), axis=-1)
        
        return(Z)       
        

    
class fftBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        pass

       
    def call(self, inputs):
        Z = inputs
        
        real = layers.Lambda(lambda Z : Z[:,:,:,0])(Z)
        imag = layers.Lambda(lambda Z : Z[:,:,:,1])(Z)
        
        Z_complex = tf.complex(real, imag)
        
        k_space = fft2d(Z_complex)

        
        real = tf.math.real(k_space)
        imag = tf.math.imag(k_space)
              
        Z = tf.stack((real, imag), axis=-1)
        
        return(Z)    



class dumbell(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(16, 3)
        self.sConv01 = convBlock(16, 3, 2)
        self.conv02 = convBlock(32, 3)
        self.sConv02 = convBlock(32, 3, 2)
        self.conv03 = convBlock(64, 3)
        self.sConv03 = convBlock(128, 3, 2)
        
        self.res01 = resBlock(128, 3)
        self.res02 = resBlock(128, 3)
        self.res03 = resBlock(128, 3)
        
        self.tConv01 = convTransBlock(64, 3, 2)
        self.tConv02 = convTransBlock(32, 3, 2)
        self.tConv03 = convTransBlock(16, 3, 2)
        self.conv04 = convBlock(16, 3)
        self.conv05 = convBlock(16, 3)
    
        self.synt = layers.Conv2D(2, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        
        conv01 = self.conv01(inputs)
        sConv01 = self.sConv01(conv01)
        conv02 = self.conv02(sConv01)
        sConv02 = self.sConv02(conv02)
        conv03 = self.conv03(sConv02)
        sConv03 = self.sConv03(conv03)

        
        res01 = self.res01(sConv03)      
        res02 = self.res02(res01)
        res03 = self.res03(res02)  

        

        tConv01 = self.tConv01(res03)
        add01 = layers.Add()([tConv01, conv03])
        tConv02 = self.tConv02(add01)
        add02 = layers.Add()([tConv02, conv02])
        tConv03 = self.tConv03(add02)
        conv04 = self.conv04(tConv03)
        conv05 = self.conv05(conv04)
        synt = self.synt(conv05)

        return(synt)




class dumbellXL(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(32, 3)
        self.conv02 = convBlock(32, 3)
        self.sConv01 = convBlock(64, 3, 2)
        
        self.conv03 = convBlock(64, 3)
        self.conv04 = convBlock(64, 3)
        self.sConv02 = convBlock(128, 3, 2)
        
        self.conv05 = convBlock(128, 3)
        self.conv06 = convBlock(128, 3)
        self.sConv03 = convBlock(256, 3, 2)
        
        self.conv07 = convBlock(256, 3)
        self.conv08 = convBlock(256, 3)
        self.sConv04 = convBlock(512, 3, 2)


        
        self.res01 = resBlock(512, 3)
        self.res02 = resBlock(512, 3)
        self.res03 = resBlock(512, 3)
        self.res04 = resBlock(512, 3)
        
        self.tConv01 = convTransBlock(256, 3, 2)
        self.conv09 = convBlock(256, 3)
        self.conv10 = convBlock(256, 3)
        
        
        self.tConv02 = convTransBlock(128, 3, 2)
        self.conv11 = convBlock(128, 3)
        self.conv12 = convBlock(128, 3)
        
        self.tConv03 = convTransBlock(64, 3, 2)
        self.conv13 = convBlock(64, 3)
        self.conv14 = convBlock(64, 3)
        
        self.tConv04 = convTransBlock(32, 3, 2)
        self.conv15 = convBlock(32, 3)
        self.conv16 = convBlock(32, 3)
    
        self.synt = layers.Conv2D(2, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        conv01 = self.conv01(inputs)
        conv02 = self.conv02(conv01)
        sConv01 = self.sConv01(conv02)

        conv03 = self.conv03(sConv01)
        conv04 = self.conv04(conv03)
        sConv02 = self.sConv02(conv04)        
        
        conv05 = self.conv05(sConv02)
        conv06 = self.conv06(conv05)
        sConv03 = self.sConv03(conv06)
        
        conv07 = self.conv07(sConv03)
        conv08 = self.conv08(conv07)
        sConv04 = self.sConv04(conv08)

        
        res01 = self.res01(sConv04)
        res02 = self.res02(res01)
        res03 = self.res03(res02)
        res04 = self.res04(res03)
        

        tConv01 = self.tConv01(res04)
        add01 = layers.Concatenate()([tConv01, conv08])
        conv09 = self.conv09(add01)
        conv10 = self.conv10(conv09)
        
        

        tConv02 = self.tConv02(conv10)
        add02 = layers.Concatenate()([tConv02, conv06])
        conv11 = self.conv11(add02)
        conv12 = self.conv12(conv11)
        

        tConv03 = self.tConv03(conv12)
        add03 = layers.Concatenate()([tConv03, conv04])
        conv13 = self.conv13(add03)
        conv14 = self.conv14(conv13)
        
        

        tConv04 = self.tConv04(conv14)
        add04 = layers.Concatenate()([tConv04, conv02])
        conv15 = self.conv15(add04)
        conv16 = self.conv16(conv15)
        
        synt = self.synt(conv16)
        

        return(synt)
    
    
    
class dumbellXLv4(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.conv01 = convBlock(32, 3)
        self.conv02 = convBlock(32, 3)
        self.conv03 = convBlock(32, 3)
        self.sConv01 = convBlock(64, 3, 2)
        
        self.conv04 = convBlock(64, 3)
        self.conv05 = convBlock(64, 3)
        self.conv06 = convBlock(64, 3)
        self.sConv02 = convBlock(128, 3, 2)
        
        self.conv07 = convBlock(128, 3)
        self.conv08 = convBlock(128, 3)
        self.conv09 = convBlock(128, 3)
        self.sConv03 = convBlock(256, 3, 2)
        
        self.conv10 = convBlock(256, 3)
        self.conv11 = convBlock(256, 3)
        self.conv12 = convBlock(256, 3)
        self.sConv04 = convBlock(512, 3, 2)


        
        self.res01 = resBlock(512, 3)
        self.res02 = resBlock(512, 3)
        self.res03 = resBlock(512, 3)
        self.res04 = resBlock(512, 3)
        self.res05 = resBlock(512, 3)
        self.res06 = resBlock(512, 3)



        
        self.tConv01 = convTransBlock(256, 3, 2)
        self.conv13 = convBlock(256, 3)
        self.conv14 = convBlock(256, 3)
        self.conv15 = convBlock(256, 3)
        
        
        self.tConv02 = convTransBlock(128, 3, 2)
        self.conv16 = convBlock(128, 3)
        self.conv17 = convBlock(128, 3)
        self.conv18 = convBlock(128, 3)
        
        self.tConv03 = convTransBlock(64, 3, 2)
        self.conv19 = convBlock(64, 3)
        self.conv20 = convBlock(64, 3)
        self.conv21 = convBlock(64, 3)
        
        self.tConv04 = convTransBlock(32, 3, 2)
        self.conv22 = convBlock(32, 3)
        self.conv23 = convBlock(32, 3)
        self.conv24 = convBlock(32, 3)
    
        self.synt = layers.Conv2D(2, 3, padding="same", use_bias=False)
       
    
    def call(self, inputs):
        conv01 = self.conv01(inputs)
        conv02 = self.conv02(conv01)
        conv03 = self.conv03(conv02)
        sConv01 = self.sConv01(conv03)

        conv04 = self.conv04(sConv01)
        conv05 = self.conv05(conv04)
        conv06 = self.conv06(conv05)
        sConv02 = self.sConv02(conv06)        
        
        conv07 = self.conv07(sConv02)
        conv08 = self.conv08(conv07)
        conv09 = self.conv09(conv08)
        sConv03 = self.sConv03(conv09)
        
        conv10 = self.conv10(sConv03)
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)
        sConv04 = self.sConv04(conv12)

        
        res01 = self.res01(sConv04)
        res02 = self.res02(res01)
        res03 = self.res03(res02)
        res04 = self.res04(res03)
        res05 = self.res05(res04)
        res06 = self.res06(res05)
        

        tConv01 = self.tConv01(res06)
        add01 = layers.Concatenate()([tConv01, conv12])
        conv13 = self.conv13(add01)
        conv14 = self.conv14(conv13)
        conv15 = self.conv15(conv14)
        
        

        tConv02 = self.tConv02(conv15)
        add02 = layers.Concatenate()([tConv02, conv09])
        conv16 = self.conv16(add02)
        conv17 = self.conv17(conv16)
        conv18 = self.conv18(conv17)
        

        tConv03 = self.tConv03(conv18)
        add03 = layers.Concatenate()([tConv03, conv06])
        conv19 = self.conv19(add03)
        conv20 = self.conv20(conv19)
        conv21 = self.conv21(conv20)
        
        

        tConv04 = self.tConv04(conv21)
        add04 = layers.Concatenate()([tConv04, conv03])
        conv22 = self.conv22(add04)
        conv23 = self.conv23(conv22)
        conv24 = self.conv24(conv23)
        
        synt = self.synt(conv24)
        

        return(synt)
  