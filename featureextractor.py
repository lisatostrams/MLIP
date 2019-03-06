# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:37:30 2019

@author: Casper
"""
from PIL import Image
import pandas as pd
import os
import json
from collections import Counter

def contrast_value(image):
    g_image = image.convert('LA')
    return g_image.mean()


def get_metadata_label(petid):
    image_dir = 'petfinder-adoption-prediction/train_metadata/'
    
    images = [i for i in os.listdir(image_dir) if i.startswith(petid)]
    toplabels = []
    for imagename in images:
        file_txt = open(image_dir + imagename)
        data = json.loads(file_txt.read())
        try:
            toplabels.append(data["labelAnnotations"][0]['description'])
        except Exception as e:
            print(e)
    if toplabels:
        return (max(set(toplabels), key=toplabels.count))
    else:
        return ""
    


def main():
    
    orig_data = pd.read_fwf('petfinder-adoption-prediction/train/train.csv')
    filenames = orig_data['PetID']
    
    
    #generate metadata label feature
    for petid in filenames:
        print(get_metadata_label(petid))
    
    #generate occurrences rescuerID
    #RescuerIDs = origdata['RescuerID']
    #dict = Counter(RescuerIDs)
    #print(dict)
    
            


if __name__ == '__main__':
    main()