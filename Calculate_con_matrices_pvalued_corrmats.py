#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 23:28:46 2022

@author: m.schepers
"""


import os
import numpy as np
import pandas as pd


path = "/data/public_data/HCPAgingDerivatives"

names = []
for file in os.listdir(path):
    if file.startswith("HCA") and not file.startswith("HCA_"):
        names.append(file)

to_remove = ['HCA8239378'] # is missing
names.remove(*to_remove)

files = []     
for i in names:
    files.append(f"/data/public_data/HCPAgingDerivatives/{i}/V1/rfMRI_REST1/GlasserFreesurfer/{i}_V1_rfMRI_REST1_Atlas_MSMAll_hp0_clean_GlasserFreesurfer.txt")

# names = names[0:2]
# files = files[0:2]


#%%
print("Getting all timeseries ...")

# The whole 100 individuals Cohort is in this list, each individual is in Cohort[i]
Cohort=[]
for i in range(0,len(files)):
    df=pd.read_csv(files[i],sep='\t',header=None)
    Cohort.append(df)

#%%
print("Importing subnetworks ...")

def import_subnetworks(path_regions, path_region_names):
    
    regions = pd.read_csv(path_regions, header=None)
    names = pd.read_csv(path_region_names, header=None)
    
    subnetworks = pd.DataFrame()
    subnetworks["region"] = list(regions[0])
    subnetworks["subnetwork"] = list(names[0])
    
    FPN = list(subnetworks[subnetworks['subnetwork']=='FP'].index.values)
    DMN = list(subnetworks[subnetworks['subnetwork']=='DMN'].index.values)
    subcortical = list(subnetworks[subnetworks['subnetwork']=='SC'].index.values)
    FPN_names = list(subnetworks['region'][FPN])
    DMN_names = list(subnetworks['region'][DMN])

    all_node_names = list(subnetworks['subnetwork'])
    
    return FPN, DMN, subcortical, FPN_names, DMN_names, all_node_names
path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
FPN, DMN, subcortical, FPN_names, DMN_names, all_node_names = import_subnetworks(path_regions, path_region_names)

#%%
print("Converting to correlation matrices ...")

#I used similar function that Eduarda and Minne used for their analysis - found in file Calculate_con_matrices.py in Minnes folder
Corr_Matrix=[]
Rescaled_Matrix=[]

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
    Rescaled_Matrix.append(rescaled)

    
#%%

for i in range(len(names)):
    print(i)
    name = names[i]
    matrix = Corr_Matrix[i]

    np.save(f'/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/intermediate_matrices/{name}.npz', matrix)

#%%
