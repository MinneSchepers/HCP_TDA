# -*- coding: utf-8 -*-
"""Calculate connectivity matrices

   To convert timeseries to connectivity matrices

"""

__author__ = "Minne Schepers"
__contact__ = "minneschepers@gmail.com"
__date__ = "10/05/2021"   ### Date it was created
__status__ = "Production" ### Production = still being developed. Else: Concluded/Finished.

####################
# Review History   #
####################

# Reviewed by Name Date ### e.g. Eduarda Centeno 20200909


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
import os

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
from mpire import WorkerPool # version 2.3.3
import numpy as np # version 1.20.3
import pandas as pd # version 1.4.1
import tqdm # version 4.55.0


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

# Delete participants if necessary
# 'HCA8239378': no timeseries file available in folder
to_remove = ['HCA8239378']
names.remove(*to_remove)
print(len(names))


####################
# Functions        #
####################

# Define function to calculate and export correlation matrix
def calculate_corr_mat(name):
    """ Calculates correlation matrix from timeseries for one subject
    
    
    Parameters
    ----------
    name: name of participant. Is used to locate the timeseries and also to 
        name the correlation matrix file csv file which is exported. 
        
    
    Returns
    -------
    Exports output_matrix. Correlation matrix. Is absolutized, diagonal is 
    filled with np.nan, z-scored and rescaled. 
    
    Notes
    -------
    
    
    """
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
 
    output_matrix.to_csv(f"/data/KNW/KNW-stage/m.schepers/HCP/Connectivity_Matrices/HCP_REST1_conn_mat/{name}.csv",
                          index=False, sep=' ', na_rep = 'NaN')


####################
# Run Functions    #
####################

# Perform the function above pooled with n_workers
n_workers=10
pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(calculate_corr_mat, names), total=len(names)):
    pass







