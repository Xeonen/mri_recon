# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:52:50 2020

@author: Xeonen
"""


from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit([1,2,3], [1,2,3])


# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(filters=1, kernel_size=(3,3), input_shape=(10,10,1), use_bias=True))
# model.add(keras.layers.Conv2D(filters=1, kernel_size=(1,1), input_shape=(10,10,1), use_bias=False))
# model.compile(optimizer="adam", loss="mse")

# model.fit([1,2,3], [1,2,3])

# print(model.summary())