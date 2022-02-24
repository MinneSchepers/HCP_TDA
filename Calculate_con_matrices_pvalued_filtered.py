#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:58:16 2022

@author: m.schepers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpire import WorkerPool
import tqdm

#%%

path_rand = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/random_matrices/'
all_random_matrices = os.listdir(path_rand)

path_corrmat = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/corr_matrices/'
path_filtered_matrices = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/filtered_matrices/'
path_pvalued_matrices = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat_pvalued/pvalued_matrices/'

all_filtered_matrices = os.listdir(path_filtered_matrices)
all_filtered_matrices = [i + '.npz' for i in all_filtered_matrices]

matrices_to_filter = [i for i in all_random_matrices if i not in all_filtered_matrices]

to_exclude = ['HCA6075263.npz'] #invalid block type in array nr 75 in 100 random matrices
matrices_to_filter.remove(*to_exclude)


#%%


def get_links(ensemble,i,j):
    links = []
    for i in range(0, len(ensemble)):
        matrix = ensemble[i]
        links.append(matrix[i,j])
    return links


def random_vs_original(corr_matrix, random_data, i, j, verbose):
    
    original_edge = corr_matrix[i, j]
    random_edges = get_links(random_data, i, j)
    
    temp=stats.kstest(np.array([original_edge]), random_edges)
    
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

    p_matrix = np.zeros((corr_matrix.shape))    

    for i in range(0, p_matrix.shape[0]):
        for j in range(i, p_matrix.shape[0]):
            if i != j:
                p_value = random_vs_original(corr_matrix, random_data, i, j, verbose=False)
                p_matrix[i,j] = p_value
                p_matrix[j,i] = p_value
                
    return p_matrix


def import_data(person):
    
    corr_matrix = np.load(path_corrmat + person + '.npy')
    random_data = np.load(path_rand + person)
    random_data = [v for k, v in random_data.items()]
    
    return corr_matrix, random_data
                

def filter_corr(person):
    
    corr_matrix, random_data = import_data(person)
    
    filtered_corr = np.copy(corr_matrix)
    p_matrix = p_value_matrix(corr_matrix, random_data)
   
    for index, values in np.ndenumerate(p_matrix):
        if p_matrix[index[0],index[1]]>0.05:
            filtered_corr[index[0],index[1]]=0
        
    np.savetxt(path_filtered_matrices + person[:-4], filtered_corr)
    np.savetxt(path_pvalued_matrices + person[:-4], p_matrix)
        
    return filtered_corr, p_matrix



n_workers = 10

print("Creating filtered p-valued matrices ...")
print(f'     Number of files: {len(matrices_to_filter)}')
print(f'     Number of workers: {n_workers}')

pool = WorkerPool(n_jobs=n_workers)
for _ in tqdm.tqdm(pool.imap_unordered(filter_corr, matrices_to_filter), total=len(matrices_to_filter)):
    pass

#%%

# def plot_summary(person):
#     #Makes a summary of raw connectivity and other things
#     corr_matrix, random_data = import_data(person)
    
#     plt.imshow(corr_matrix,cmap='viridis')
#     plt.title(f'Raw_Corr Individual {person}')
#     plt.colorbar()
#     # plt.savefig(f'Figs/Raw_Corr Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     filtered_corr, p_matrix = filter_corr(person)
#     plt.imshow(filtered_corr,cmap='Greys')
#     plt.title(f'Filtered Correlations Individual {person}')
#     plt.colorbar()
#     # plt.savefig('Figs/Filtered_Corr Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.imshow(p_matrix,cmap='viridis')
#     plt.title(f'p_values Individual {person}')
#     plt.colorbar()
#     # plt.savefig('Figs/P_values_Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
# plot_summary(person)
