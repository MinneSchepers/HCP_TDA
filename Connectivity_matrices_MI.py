#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate connectivity matrices using mutual information score

   Python script for calculating connectivity matrices from fMRI timeseries
   using the mutual information as the scoring method. 
   The functions below were adjusted from codes by Fernando Nobrega 
   (f.nobregasantos@amsterdamumc.nl). 

"""

__author__ = "Minne Schepers"
__contact__ = "minneschepers@gmail.com"
__date__ = "24/02/2021"   ### Date it was created
__status__ = "Production" ### Production = still being developed. Else: Concluded/Finished.

####################
# Review History   #
####################

# Reviewed by Name Date ### e.g. Eduarda Centeno 20200909


####################
# Libraries        #
####################

# Standard imports 
import os

# Third party imports
from mpire import WorkerPool
import numpy as np # version 1.20.3
import pandas as pd # version 1.4.1
from sklearn.metrics import mutual_info_score # version 1.0.2
import tqdm# version 4.55.0


####################
# Import data      #
####################

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


####################
# Functions        #
####################

def calc_MI(x, y, bins):
    """ Calculates mutual information score between two parameters
    
    
    Parameters
    ----------
    x: value 1 in matrix
    y: value 2 in matrix
    bins: number of bins for discretizing the continuous data before 
        calculating mutual ifnormation score. There is debate on the number 
        ,some suggest 10 is the best option
        
    
    Returns
    -------
    mi: mutual information score, float
    
      
    
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    
    return mi


def scaled(X):
    """ Scales the connectivity matrix. 
    
    
    Parameters
    ----------
    X: connectivity matrix
        
    
    Returns
    -------
    X_scaled: scaled connectivity matrix
        
    
    """
    
    Xmax=1
    Xmin=0
    X_std = (X - np.min(X)) / (np.max(X) - np.min(X))
    X_scaled = X_std * (Xmax - Xmin) + Xmin
    
    return X_scaled


def import_timeseries(subject):
    """ Imports fMRI timeseries of a subject
    
    
    Parameters
    ----------
    subject: name of current subject. Is used to complete the path to the
        timeseries file. 
        
    
    Returns
    -------
    timeseries: the fMRI timeseries as a pandas dataframe
    
    
    
    """
    # Imports the timeseries of a subject
    
    path = f"/data/public_data/HCPAgingDerivatives/{subject}/V1/rfMRI_REST1/GlasserFreesurfer/{subject}_V1_rfMRI_REST1_Atlas_MSMAll_hp0_clean_GlasserFreesurfer.txt"
    timeseries = pd.read_csv(path,sep='\t',header=None)
    
    return timeseries


def MI_matrix(subject):
    """ Calculates connectivity matrix with mutual information scores
    
    
    Parameters
    ----------
    subject: name of current subject, used for importing timeseries and 
        for exporting the matrix. 
        
    
    Returns
    -------
    MI_scaled: exports scaled mutual information connectivity matrix
    
    
    
    """
    # Opens timeseries, constructs and exports sclaed mutual information matrix 
    
    temp = import_timeseries(subject)
    
    MI = np.zeros((temp.shape[0],temp.shape[0]))
    for i in range(0,temp.shape[0]):
        #MI[i,i]=0
        for j in range(i,temp.shape[0]):
            if i!=j:
                value=calc_MI(temp[i].tolist(),temp[j].tolist(),10)
                MI[i,j]=value#calc_MI(test[i].tolist(),test[j].tolist(),10)
                MI[j,i]=value
                
    MI_scaled = scaled(MI)      
    np.savetxt(f'/data/KNW/KNW-stage/m.schepers/HCP/Connectivity_Matrices/HCP_REST1_conn_mat_MI/{subject}',
               MI_scaled)
    
    return MI_scaled


####################
# Run Functions    #
####################

n_workers = 10

print("Creating randomized matrices ...")
print(f'     Number of files: {len(names)}')
print(f'     Number of workers: {n_workers}')

pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(MI_matrix, names), total=len(names)):
    pass







