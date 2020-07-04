# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:49:57 2020

@author: Xeonen
"""


import numpy as np

import dataOp
import metrics
import customLayers as cl
import helperFunctions as hf

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt


n = np.zeros(shape=(1, 218, 170, 24))

x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(n)








# nvalidArgumentError: The first dimension of paddings must be the rank of inputs[4,2] [218,170,24] [Op:Pad]