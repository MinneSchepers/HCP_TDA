#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:48:24 2022

@author: m.schepers
"""

import numpy as np
from sklearn.metrics import mutual_info_score
import os
import pandas as pd
from mpire import WorkerPool
import tqdm


# Path with all participant names
path = "/data/public_data/HCPAgingDerivatives"

# Make list with participants to process
names = []
for file in os.listdir(path):
    if file.startswith("HCA") and not file.startswith("HCA_"):
        names.append(file)

# Delete participants from list
# 'HCA8239378': no timeseries file available in folder
to_remove = ['HCA8239378'] # is missing
names.remove(*to_remove)


#### Define functions ####

def calc_MI(x, y, bins):

    # Calculates mutual information score

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    
    return mi


def scaled(X):
    
    # Resscales the mutual information matrix
    
    Xmax=1
    Xmin=0
    X_std = (X - np.min(X)) / (np.max(X) - np.min(X))
    X_scaled = X_std * (Xmax - Xmin) + Xmin
    return X_scaled


def import_timeseries(person):
    
    # Imports the timeseries of a person
    
    path = f"/data/public_data/HCPAgingDerivatives/{person}/V1/rfMRI_REST1/GlasserFreesurfer/{person}_V1_rfMRI_REST1_Atlas_MSMAll_hp0_clean_GlasserFreesurfer.txt"
    timeseries = pd.read_csv(path,sep='\t',header=None)
    
    return timeseries


def MI_matrix(person):
    
    # Opens timeseries, constructs and exports sclaed mutual information matrix 
    
    temp = import_timeseries(person)
    
    MI = np.zeros((temp.shape[0],temp.shape[0]))
    for i in range(0,temp.shape[0]):
        #MI[i,i]=0
        for j in range(i,temp.shape[0]):
            if i!=j:
                value=calc_MI(temp[i].tolist(),temp[j].tolist(),10)
                MI[i,j]=value#calc_MI(test[i].tolist(),test[j].tolist(),10)
                MI[j,i]=value
                
    MI_scaled = scaled(MI)      
    np.savetxt(f'/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_MI/{person}', MI_scaled)
    
    return MI_scaled


    

#### Process data ####

n_workers = 10

print("Creating randomized matrices ...")
print(f'     Number of files: {len(names)}')
print(f'     Number of workers: {n_workers}')

pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(MI_matrix, names), total=len(names)):
    pass







