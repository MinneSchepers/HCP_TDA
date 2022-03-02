#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate connectivity matrices 

   Pythons script for calculating connectivity matrices from fMRI 
   timeseries. This is the first step of calculating filtered pvalued
   connectivity matrices. 
   

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
import numpy as np # version 1.20.3
import pandas as pd # version 1.4.1


####################
# Import data      #
####################

# Path to directory with all subject names
path = "/data/public_data/HCPAgingDerivatives"

# Make list with all subject names
names = []
for file in os.listdir(path):
    if file.startswith("HCA") and not file.startswith("HCA_"):
        names.append(file)

# Remove missing data
to_remove = ['HCA8239378'] # is missing
names.remove(*to_remove)

print("Getting all timeseries ...")

# Make list with all paths to the timeseries of all subjects
Cohort = []     
for i in names:
    path_to_timeseries = f"/data/public_data/HCPAgingDerivatives/{i}/V1/rfMRI_REST1/GlasserFreesurfer/{i}_V1_rfMRI_REST1_Atlas_MSMAll_hp0_clean_GlasserFreesurfer.txt"
    df = pd.read_csv(path_to_timeseries,sep='\t',header=None)
    Cohort.append(df)


#%%

# Import all subnetworks to exclude all subcortical regions 
path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'

regions = pd.read_csv(path_regions, header=None)
names = pd.read_csv(path_region_names, header=None)

subnetworks = pd.DataFrame()
subnetworks["region"] = list(regions[0])
subnetworks["subnetwork"] = list(names[0])

subcortical = list(subnetworks[subnetworks['subnetwork']=='SC'].index.values)


print("Converting to correlation matrices ...")

# For every subject create connectivity matrix
Corr_Matrix=[]

for individual in Cohort:
    corr_matrix = np.array(individual.T.corr(method='pearson'))
    corr_matrix = np.delete(corr_matrix, subcortical, axis=0)
    corr_matrix = np.delete(corr_matrix, subcortical, axis=1)
    #Making the diagonal elements zero
    np.fill_diagonal(corr_matrix, 0)
    matrixabs = abs(np.array(corr_matrix))
    np.fill_diagonal(matrixabs, np.nan)
    zmatrix = (matrixabs - np.nanmean(matrixabs))/np.nanstd(matrixabs)
    rescaled = (zmatrix - np.nanmin(zmatrix))/(np.nanmax(zmatrix)- np.nanmin(zmatrix))
    Corr_Matrix.append(np.abs(corr_matrix))

# Export connectivity matrix
for i in range(len(names)):
    print(i)
    name = names[i]
    matrix = Corr_Matrix[i]

    np.save(f'/data/KNW/KNW-stage/m.schepers/HCP/Connectivity_Matrices/HCP_REST1_conn_mat_pvalued/connectivity_matrices/{name}.npz',
            matrix)

