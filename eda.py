# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:33:00 2019

@author: Lisa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('petfinder-adoption-prediction/train/train.csv')

#%%

plt.hist(data['Name'])