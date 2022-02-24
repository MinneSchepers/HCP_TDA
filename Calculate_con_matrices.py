# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:06:18 2021

@author: Minne Schepers

To convert timeseries to connectivity matrices
    

"""

import os
import numpy as np
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
to_remove = ['HCA8239378']
names.remove(*to_remove)
print(len(names))


# Define function to calculate and export correlation matrix
def calculate_corr_mat(name):
        
    # Path to look for timeseries
    path_2 = f"/data/public_data/HCPAgingDerivatives/{name}/V1/rfMRI_REST1/GlasserFreesurfer/{name}_V1_rfMRI_REST1_Atlas_MSMAll_hp0_clean_GlasserFreesurfer.txt"

    # Import timeseries
    timeseries = pd.read_csv(path_2, sep='\t', header=None)

    # Calculate correlation matrix
    corr_matrix = np.array(timeseries.T.corr(method='pearson'))
    
    # Absolutize, zscore and rescale
    matrixabs = abs(np.array(corr_matrix))
    np.fill_diagonal(matrixabs, np.nan)
    zmatrix = (matrixabs - np.nanmean(matrixabs))/np.nanstd(matrixabs)
    rescaled = (zmatrix - np.nanmin(zmatrix))/(np.nanmax(zmatrix)- np.nanmin(zmatrix))
    
    # Export matrix as pandas dataframe
    output_matrix = pd.DataFrame(rescaled)    
    output_matrix.to_csv(f"/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat/{name}.csv",
                          index=False, sep=' ', na_rep = 'NaN')


# Perform the function above pooled with n_workers
n_workers=10
pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(calculate_corr_mat, names), total=len(names)):
    pass







