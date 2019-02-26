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

import collections

name_counter = collections.Counter(data['Name'].values)
lst = name_counter.most_common(20)
df = pd.DataFrame(lst, columns = ['Name', 'Count'])
df.plot.bar(x='Name',y='Count')

#%%

adoptlst = []

for l in lst:
    idx = data.loc[:,  'Name'] == l[0]
    adoptlst.append(data[idx]['AdoptionSpeed'].values)
    
    