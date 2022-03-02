#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate filtered pvalued connectivity matrices

   Pythons script for calculating randomized n connectivity matrices from 
   connectivity matrices. This is the third and lest step of calculating 
   filtered pvalued connectivity matrices. 
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
import matplotlib.pyplot as plt # version 3.3.2
from mpire import WorkerPool# version 2.3.3
import numpy as np # version 1.20.3
from scipy import stats # version 1.6.2
import tqdm # version 4.55.0


####################
# Import data      #
####################

# Path to randomized matrices 
path_rand = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/random_matrices/'
all_random_matrices = os.listdir(path_rand)

# Path of correlation matrices from step 1
path_corrmat = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/corr_matrices/'

# Path for exporting filtered pvalued matrices
path_filtered_matrices = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/filtered_matrices/'
# Path for exporting pvalued matrices (not yet filtered)
path_pvalued_matrices = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/pvalued_matrices/'

# Exclude already filtered matrices
all_filtered_matrices = os.listdir(path_filtered_matrices)
all_filtered_matrices = [i + '.npz' for i in all_filtered_matrices]
matrices_to_filter = [i for i in all_random_matrices if i not in all_filtered_matrices]

# Exclude missing data
to_exclude = ['HCA6075263.npz'] #invalid block type in array nr 75 in 100 random matrices
matrices_to_filter.remove(*to_exclude)


####################
# Functions        #
####################


def get_links(ensemble,i,j):
    """ Retrieves links from ensemble of matrices. For example, with i=4 and 
    j = 6, this function returns a list with all values from the value 
    with row value 4 and column value 6 from all matrices in the ensemble. 
    
    
    Parameters
    ----------
    ensemble: list of connectivity matrices
    i: index from row in matrix
    j: index from column in matrix
        
    
    Returns
    -------
    links: list of all values from links i, j
    
    
    """
    links = []
    for i in range(0, len(ensemble)):
        matrix = ensemble[i]
        links.append(matrix[i,j])
        
    return links


def random_vs_original(corr_matrix, random_data, i, j, verbose):
    """ Compares the original matrix to the randomized matrices. Performs a 
    Kolmogorov-Smirnov (KS) test to test for significance. Can also plot the 
    original versus the random links. 
    
    Parameters
    ----------
    corr_matrix: 
    random_data: 
    i: index from row in matrix
    j: index from column in matrix
    verbose: if True, make plot of the original link versus the randomized 
        links
    
    
    Returns
    -------
    temp[-1]: the p-value from the KS--test
    
    
    """    
    # The original link 
    original_edge = corr_matrix[i, j]
    # List of all the random links
    random_edges = get_links(random_data, i, j)
    
    # Perform Kolmogorov Smirnov test. 
    temp=stats.kstest(np.array([original_edge]), random_edges)
    
    # Plot original versus randomized links
    if verbose == True:
        plt.hist(random_edges, bins=5, label='Randomized Edge')
        plt.hist(original_edge, bins=20, label='Original Edge')
        plt.title(f'Edge index: {i}, {j} \n {temp}')
        plt.xlabel('strength')
        plt.ylabel('frequency')
        plt.legend()
        plt.show()

    return temp[-1]
    

def p_value_matrix(corr_matrix, random_data):
    """ Constructs a connectivity matrix with only p-values. The p-values are
    from the Kolmogorov-Smirnov test which indicates whether the original 
    link is significantly different from the randomized links. 
    
    
    Parameters
    ----------
    corr_matrix: original correlation matrix
    random_data: the randomized correlation matrices
    
        
    Returns
    -------
    p_matrix: pvalued connectivity matrix
    
      
    
    """
    p_matrix = np.zeros((corr_matrix.shape))    

    for i in range(0, p_matrix.shape[0]):
        for j in range(i, p_matrix.shape[0]):
            if i != j:
                p_value = random_vs_original(corr_matrix, random_data, i, j, verbose=False)
                p_matrix[i,j] = p_value
                p_matrix[j,i] = p_value
                
    return p_matrix


def import_data(subject):
    
    """ Imports the original connectivity matrix and the randomized data
    for one subject

    
    Parameters
    ----------
    subject: name of subject
        
    
    Returns
    -------
    corr_matrix: original connectivity matrix
    random_data: list of randomized connectivity matrices
    
    
    """
    
    corr_matrix = np.load(path_corrmat + subject + '.npy')
    random_data = np.load(path_rand + subject)
    random_data = [v for k, v in random_data.items()]
    
    return corr_matrix, random_data
                

def filter_corr(subject):
    """ Creates a filtered pvalued correlation matrix. If the value in the 
    pvalued matrix is significant (p < 0.05), the original value is included
    in the filtered connectivity matrix. If not, the value is set to 0. 
    
    
    
    Parameters
    ----------
    subject: name of subject
        
    
    Returns
    -------
    filtered_corr: the filtered pvalued matrix
    p_matrix: the pvalued matrix
    
    
    """
    corr_matrix, random_data = import_data(subject)
    
    filtered_corr = np.copy(corr_matrix)
    p_matrix = p_value_matrix(corr_matrix, random_data)
   
    for index, values in np.ndenumerate(p_matrix):
        if p_matrix[index[0],index[1]]>0.05:
            filtered_corr[index[0],index[1]]=0
        
    np.savetxt(path_filtered_matrices + subject[:-4], filtered_corr)
    np.savetxt(path_pvalued_matrices + subject[:-4], p_matrix)
        
    return filtered_corr, p_matrix


####################
# Run Functions    #
####################

n_workers = 10

print("Creating filtered p-valued matrices ...")
print(f'     Number of files: {len(matrices_to_filter)}')
print(f'     Number of workers: {n_workers}')

pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(filter_corr, matrices_to_filter), total=len(matrices_to_filter)):
    pass
