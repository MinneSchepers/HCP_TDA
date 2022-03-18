#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:58:16 2022

@author: m.schepers - adapted to HCP Young Adult
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#from mpire import WorkerPool
#import tqdm

#%%

import os
import numpy as np
#setting the current working directory
#os.chdir('/Users/boltzmann/Dropbox/VUmc/Topology_Behavior/Backup_Pierre')
os.getcwd()
import glob
#Getting the files addresses no need / in the begging
files = glob.glob('HCP_AAL_Young_Adult/functional_connectivity/*functional_connectivity.txt')
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import bct


#%%

# The whole 100 individuals Cohort is in this list, each individual is in Cohort[i]
import pandas as pd
Cohort=[]
for i in range(0,len(files)):
    df=pd.read_csv(files[i],header=None)
    #df = df.iloc[:-1 , :]
    # if needed, some cleaning step could be done in this stage here
    Cohort.append(np.abs(df))
    
def Ensemble(size,ind):
    #size = number of copies of surrogate matrixes - I fixed 100 - but we can change is needed
    #ind individual in your data
    Ensemble=[]
    for i in range(0,100):
    #creating a surrogate matrix based in the real data - I had to include.to_numpy in comparison with Minne's code
        W0=bct.null_model_und_sign(Cohort[ind].to_numpy())
        Ensemble.append(W0)
    return Ensemble




def get_links_r(random_data,i,j):
    #Create the random_data before using this function
    data=random_data
    links=[]
    # Here we get all the 1000 (or more) random links with index i and j
    for ind in range(0,len(data)):
        matrix=data[ind][0]
        links.append(matrix[i,j])
    
    return links



def random_vs_original_r(random_data,ind,i,j,verbose=True):
    #Random_data is the surrogate data for individual ind
    #ind is the individual lable
    #i, and j are the links we want to compute the p_value
    
    
    # Getting the distribution of surrogate data for link i,j
    data=get_links_r(random_data,i,j)
   

    data2= Cohort[ind].to_numpy()#original_data
    if verbose == True:
        plt.hist(get_links_r(random_data,i,j),bins=5, label = 'Randomized Edge')
        plt.hist(data2[i,j],bins=20, label = 'Original Edge')
   
    
    x =np.array([data2[i,j]]) 
    #Computin the KS test and geting the p_value
    temp=stats.kstest(x, data)
    if verbose == True:
        plt.title('Edge index:'+str(i)+', '+str(j)+'\n' + str(temp))
        plt.xlabel('strength')
        plt.ylabel('frequency')
        plt.legend()
        #plt.savefig('Edge_links'+str(i)+'_'+str(j)+'.png')
        plt.show()
    #print(temp[-1])      
    return temp[-1]#print(stats.kstest(x, data))


def p_values_M(ind):
    # Here I created 1000 surrogated matrixes 
    random_data=Ensemble(1000,ind)
    real_data=Cohort[ind].to_numpy()
    p_M=np.zeros((real_data.shape[0],real_data.shape[0]))
    #p_values=[]
    for i in range(0,real_data.shape[0]):
        for j in range(i,real_data.shape[0]):
            # Notice that the matrix is symetric, I only need to run on the upper part of the matrix
            if i!=j:
            
                p_value=random_vs_original_r(random_data,ind,i,j,verbose=False)
                
                p_M[i,j]=p_value
                p_M[j,i]=p_value
    return p_M

def filter_corr(ind,save=False):
    #This function creates the filtered correlation matrixes based on the p_value
    
    new_corr=np.copy(Cohort[ind].to_numpy())
    p_matrix=p_values_M(ind)
    for index, values in np.ndenumerate(p_matrix):
        if p_matrix[index[0],index[1]]>0.05:
            new_corr[index[0],index[1]]=0
    if save == True:
        #Saving the filtered correlation matrix
        np.savetxt('Filtered_Corr_Matrixes/Filtered_p_values_subj_'+files[ind][-34:],new_corr)
        #Saving the p_value matrix
        np.savetxt('P_value_Matrixes/P_value_matrix_subj_'+files[ind][-34:],p_matrix)
        

    return new_corr, p_matrix

def plot_summary(ind):
    #Makes a summary of raw connectivity and other things
    plt.figure()
    plt.ioff()
    plt.imshow(Cohort[ind].to_numpy(),cmap='viridis')
    plt.title('Raw_Corr Individual '+str(ind))
    plt.colorbar()
    plt.savefig('Figs/Raw_Corr Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()
    plt.figure()
    plt.ioff()
    filt_corr,p_values=filter_corr(ind,save=True)
    np.savetxt('Filtered_Corr_Matrixes/Filtered_p_values_subj_'+files[ind][-34:],filt_corr)
    np.savetxt('P_value_Matrixes/P_value_matrix_subj_'+files[ind][-34:],p_values)
    plt.imshow(filt_corr,cmap='Greys')
    plt.title('Filtered Correlations Individual '+str(ind))
    plt.colorbar()
    plt.savefig('Figs/Filtered_Corr Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure()
    #plt.show()
    plt.ioff()
    plt.imshow(p_values,cmap='viridis')
    plt.title('p_values Individual '+str(ind))
    plt.colorbar()
    plt.savefig('Figs/P_values_Individual_'+str(ind)+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

from multiprocessing import Pool


if __name__ == '__main__':
        with Pool(12) as p:
            #EulerC.append(p.map(Euler_helper4,lista(j)))
            p.map(plot_summary,range(0,998))
        
        p.close()
        p.join()




#n_workers = 10

#print("Creating filtered p-valued matrices ...")
#print(f'     Number of files: {len(matrices_to_filter)}')
#print(f'     Number of workers: {n_workers}')

#pool = WorkerPool(n_jobs=n_workers)
#for _ in tqdm.tqdm(pool.imap_unordered(filter_corr, matrices_to_filter), total=len(matrices_to_filter)):
#    pass

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





#n_workers = 10

#print("Creating filtered p-valued matrices ...")
#print(f'     Number of files: {len(matrices_to_filter)}')
#print(f'     Number of workers: {n_workers}')

#pool = WorkerPool(n_jobs=n_workers)
#for _ in tqdm.tqdm(pool.imap_unordered(filter_corr, matrices_to_filter), total=len(matrices_to_filter)):
#    pass

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
