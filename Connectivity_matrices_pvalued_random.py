#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate randomized connectivity matrices 

   Pythons script for calculating randomized n connectivity matrices from 
   connectivity matrices. This is the second step of calculating filtered 
   pvalued connectivity matrices. The randomized connectivity matrices
   are used to 
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
import bct # version 0.5.2
from mpire import WorkerPool# version 2.3.3
import numpy as np # version 1.20.3
from numpy import savez_compressed # version 1.20.3
import pandas as pd # version 1.4.1
import tqdm # version 4.55.0


####################
# Import data      #
####################

# Path of intermediate matrices
path_matrices = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/corr_matrices/'
# Path to export npz files of random_matrices to
path_export = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/random_matrices/'

# Make list with subjects
file_names = []
for file in os.listdir(path_matrices):
    if file.endswith('.npy'):
        file_names.append(file)

# Only include males and females from the training data
path_females = '/data/KNW/KNW-stage/m.schepers/HCP/Cog_data/females_train.csv'
path_males = '/data/KNW/KNW-stage/m.schepers/HCP/Cog_data/Males_exp_TDAandBD.csv'

females = pd.read_csv(path_females)
females = list(females['subject'])
males = pd.read_csv(path_males)
males = list(males['subject'])

males = [i+'.npz.npy' for i in males]
females = [i+'.npz.npy' for i in females]
to_include = males + females
file_names = [i for i in file_names if i in to_include]

# Exclude already completed files
files_completed = []
for file in os.listdir(path_export):
    if file.endswith('.npz'):
        files_completed.append(file+'.npy')

file_names = [i for i in file_names if i not in files_completed]


####################
# Functions        #
####################

def create_random_matrices(file_name):
    """ Creates randomized connectivity matrices
    
    
    Parameters
    ----------
    file_name: name of subject 
        
    
    Returns
    -------
    random_arrays: 100 randomized connectivity matrices. Exported as 
        a .npz file. 
        
    """
        
    matrix = np.load(path_matrices + file_name)
    # matrix = matrix[0:10, 0:10]
    
    random_arrays = []
    
    for i in range(0, 100):
    #creating a surrogate matrix based in the real data
        rand_array=bct.null_model_und_sign(matrix)[0]
        random_arrays.append(rand_array)
        
    savez_compressed(f'{path_export}{file_name[0:-8]}.npz', *random_arrays)
    

#####

n_workers = 10

print("Creating randomized matrices ...")
print(f'     Number of files: {len(file_names)}')
print(f'     Number of workers: {n_workers}')

pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(create_random_matrices, file_names), total=len(file_names)):
    pass


