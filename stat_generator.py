# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:31:04 2020

@author: Xeonen
"""


import dataOp
import metrics
import customLayers as cl
import helperFunctions as hf

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import layers, models, losses



custom_objects = {"mulBlock": cl.mulBlock, "ifftBlock": cl.ifftBlock,
                  "dumbell": cl.dumbell, "fftBlock": cl.fftBlock,
                  "conBlock": cl.conBlock, "image_gradient_loss": metrics.image_gradient_loss
    }

model = models.load_model("RCC_00_IGL.h5", custom_objects=custom_objects)




dL = dataOp.data_loader("C:/Datasets/MRI_Data/Recon_v4/Val", 1, 4, 15, False)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# hf.gen_comparison_graph(model, d)
# hf.getStats_magOnly(model, d)

# statDF = pd.DataFrame()
# for i in tqdm(range(dL.__len__())):
#     d = dL.__getitem__(i)
#     _ , stat_dict = hf.getStats(model, d)
#     tempDF = pd.DataFrame(stat_dict, index=[i])
#     statDF = pd.concat((statDF, tempDF))
    
# statDF.to_csv("RCC_00_MSE_FULL.csv", sep=",", header=0, encoding="UTF-8")

d = dL.__getitem__(200)
img_dict, stat_dict = hf.getStats(model, d)
hf.gen_comparison_graph(img_dict)
print(stat_dict)
