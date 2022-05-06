#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculate Topological Data Analysis (TDA) features

   Pythons script for calculating TDA features from connectivity matrices
   derived from fMRI timeseries. 
   Makes use of GUDHI Python TDA package and code by Fernando Santos. 

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
import csv
from time import time

# Third party imports 
import gudhi as gd # version 3.3.0
from gudhi.representations import Landscape # version 3.3.0
from gudhi.representations import BettiCurve # version 3.3.0
from gudhi.representations import TopologicalVector # version 3.3.0
from gudhi.representations import Entropy # version 3.3.0
from mpire import WorkerPool # version 2.3.3
import numpy as np # version 1.20.3
from numpy import savez_compressed # version 1.20.3
import pandas as pd # version 1.4.1
from scipy.signal import find_peaks # version 1.6.2
from scipy.stats import kurtosis # version 1.6.2
from scipy.stats import skew # version 1.6.2
from scipy.stats import entropy # version 1.6.2
from sklearn import metrics # version 1.0.2
import tqdm # version 4.55.0

# Internal imports
import TDA_Fernando


####################
# Functions        #
####################
  
  
def timer_func(func):
    """ Can be used as a decorator to a function to measure time it takes to
    complete a function. To do this, put @timer_func on the line directly 
    before where the function is defined. 

    
    Parameters
    ----------
    func: the function which is measured. Do not set the parameter, only put
    @timer_func before where function is defined
    
    
    Returns
    -------
    prints name of function and the time it takes to execute in seconds 
           
        
    """
    
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    
    return wrap_func


def import_data(path_data, remove_completed):
    """Makes list of all connectivity matrices to include
    
    
    Parameters
    ----------
    path_dir (str): path to directory with all connectivity matrices
    remove_completed (bool): remove the already processed subjects from analyses
    
    
    Returns
    -------
    data: list containing all names of connectivity matrices
        
    """
    data = os.listdir(path_data)
    
    if remove_completed == True:
        # Export any already completed subjects in the same file from a previous run
        # Make file if file does not exist
        open(path_export+export_filename, 'a+')
        # If file is not empty: remove completed subjects from data
        if os.stat(path_export+export_filename).st_size != 0:
            export_data = pd.read_csv(path_export+export_filename)
            completed_subjects = export_data["Subject"]
            completed_subjects = [i + ".csv" for i in completed_subjects]
            
            data = [i for i in data if i not in completed_subjects]

    return data


def import_cmatrix(path_dir, subject):
    """Imports single connectivity matrix
    
    
    Parameters
    ----------
    path_dir (str): path to directory with all connectivity matrices
    subject  (str): a single subject from the data-list
    
    
    Returns
    -------
    matrix (numpy.array): connectivity matrix as numpy array
        
    
    Notes
    ----------
    Checks if the connectivity matrix is square (because some datasets
    have a header which needs to be excluded), if not, raises exception
    with error message    
    
    """    
    path = path_dir + subject
    matrix = np.loadtxt(path, skiprows=1)

    if matrix.shape[0] != matrix.shape[1]:
        raise Exception(
            f'Error: imported connectitivy matrix is not square: {matrix.shape} \n'
            f'Consider removing the top rows if it contains a header \n'
            f'See function import_cmatrix: adjust skiprows'
            )

    return matrix


def export_TDA_data(path_export, export_filename, outcomes_to_export):
    """ Exports dictionary from single subject as row to csv file 
    
    
    Parameters
    ----------
    path_export (str): the path of the csv file to export to
    outcomes_to_export (dict): dict with all TDA data from a single subject
    
    Notes
    -----------
    If a file with the same name already exists due to a previous run, 
    the new data is added to this file. Does not overwrite. 
    
        
    """
    with open(path_export+export_filename, 'a+') as csvfile:  
        writer = csv.DictWriter(csvfile, fieldnames=outcomes_to_export.keys())    
        # If file is empty, make row with headers
        if os.stat(path_export+export_filename).st_size == 0:
            writer.writeheader()        
        # Write data as row
        writer.writerow(outcomes_to_export)
        

def calculate_persistence(matrix):
    """ Performs filtration (persistent homology) process 
    
    
    Parameters
    ----------
    matrix (numpy.array): connectivity matrix as numpy array
        
    
    Returns
    -------
    pers: persistence of simplicial complex. 
          Type:list of pairs(dimension, pair(birth, death))
    simplex_tree: data structure for representing simplicial complexes
        
        
    """
    rips_complex = gd.RipsComplex(distance_matrix=matrix, max_edge_length=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=nr_dimensions)
    pers = simplex_tree.persistence()

    return pers, simplex_tree


def persistence_per_dim(simplex_tree, nr_dimension):
    """ Performs filtration (persistent homology) process per dimension
    
    
    Parameters
    ----------
    simplex_tree: data structure for representing simplicial complexes
    nr_dimension: parameter which specifies the number of dimensions to 
        investigate. E.g. 1, 2 or 3. 
    
        
    Returns
    -------
    pers_per_dim: persistence of simplicial complex per dimension. E.g., if 
        3 dimensions: list of 3 lists of pairs(dimension, pair(birth, death))
    
    
    Notes
    -------
    Infinite values likely exist in the filtration process, because some 
    structures never cease existing. This leads to problems for computing 
    TDA metrics per dimension. Therefore, infinite values are substituted
    for the maximum finite values for other structures in the persistence
    in the same dimension. 
    
        
    """

    pers_per_dim = []

    for dimension in range(nr_dimensions):
        pers = simplex_tree.persistence_intervals_in_dimension(dimension)

        # The code below replaces any infinite values with the maximum value
        # from the persistence in dimension 0.

        # Find the maximum value which is not infinite
        if dimension == 0:
            pers = np.where(np.isinf(pers), -1, pers)
            max_val_dim0 = np.max(pers)

        # If non-empty array: replace infinite with maximum value
        if pers.shape != (0,):
            pers = np.where(np.isinf(pers), -1, pers)
            pers = np.where(pers == -1, max_val_dim0, pers)
            pers_per_dim.append(pers)

        # if no topological structures present, append empty array to prevent error
        else:
            pers_per_dim.append(np.zeros([0, 2]))

    return pers_per_dim


def betti_curves(pers_per_dim):
    """ Calculates Betti curves per dimension

    
    Parameters
    ----------
    pers_per_dim (list): list of persistence per dimension
        
    
    Returns
    -------
    outcomes_to_export (dict): dictionary with all TDA outcomes per subject. 
        keys:    
        bc_AUC_dim#: the area under the curve for the betti curve per 
            dimension, where # is the number of dimensions
        bc_max_dim#: the maximum of the betti curve per dimension. 
         
        
    """

    betti_curves = BettiCurve(resolution=resolution,
                              sample_range=[0, 1]).transform(pers_per_dim)
    
    for i in range(nr_dimensions):
        
        betti_curve_AUC = metrics.auc(
            np.linspace(0, resolution, num=resolution), betti_curves[i])
        betti_curve_max = np.max(betti_curves[i])
        
        outcomes_to_export[f'bc_AUC_dim{i}'] = betti_curve_AUC
        outcomes_to_export[f'bc_max_dim{i}'] = betti_curve_max
    
    return outcomes_to_export


def persistence_landscape(pers_per_dim):
    """ Calculates persistence landscape per dimension

    
    Parameters
    ----------
    pers_per_dim(list): list of persistence per dimension
        
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
        pl_auc_dim: the area under the curve for the persistence landscape
            per dimension. 
         
        
    """
    
    landscapes = Landscape(num_landscapes=1,
                           resolution=resolution).fit_transform(pers_per_dim)

    for i in range(nr_dimensions):
        landscape_AUC = metrics.auc(
            np.linspace(0, resolution, num=resolution), landscapes[i])
        outcomes_to_export[f'pl_AUC_dim{i}'] = landscape_AUC
    
    return outcomes_to_export


def topological_vector(pers_per_dim):
    """ Calculates topological vector per dimension

    
    Parameters
    ----------
    pers_per_dim (list): list of persistence per dimension
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
        topological vector is added as column per dimension
         
        
    """
    for i in range(nr_dimensions):
        # If no topolofical components present, set to 0 to prevent errors
        if pers_per_dim[i].size == 0:
            outcomes_to_export[f'top_vec_dim{i}'] = 0

        else:
            pers = [pers_per_dim[i]]
            tv = TopologicalVector(threshold=1).transform(pers)
            outcomes_to_export[f'top_vec_dim{i}'] = float(tv)

    return outcomes_to_export


def shannon_entropy(pers_per_dim):
    """ Calculates Shannon entropy per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
         shannon entropy is added as column per dimension
         
        
    """
    for i in range(nr_dimensions):
        # If no topolofical components present, set to 0 to prevent errors
        if pers_per_dim[i].size == 0:
            outcomes_to_export[f'S_entropy_dim{i}'] = 0

        else:
            pers = [pers_per_dim[i]]
            entropy = Entropy(mode='scalar').fit_transform(pers)
            outcomes_to_export[f'S_entropy_dim{i}'] = float(entropy)

    return outcomes_to_export


@timer_func
def euler_characeristic(matrix, subject, max_density, 
                                 Euler_resolution, k_max):
    """ Calculates Euler characteristic across a density range

    
    Parameters
    ----------
    matrix (numpy.array): connectivity matrix as numpy array
    subject (str): name of subject, needed for exporting data
    max_density (float): float between 0 and 1 to set the maximum of the density at which 
        the Euler will be sampled. If 0.1, then the Euler is sampled from 0% 
        to 10% density. 
    Euler_resolution (int): the number of samples to estimate the Euler with. With large
        connectivity matrix, i.e. from HCP at least 360 regions, use minimal 
        resolution of 1000.
    k_max (int): the maximum number of k-cliques to calculate. Lower values lowers
        computation time. 
    
    
    Returns
    -------
    outcomes_to_export (dict): dictionary with all TDA outcomes per subject. 
         The sum of the Euler characteristic per subject, the total number of 
         cliques and total number of triangles are added as a column
         per subject
    outcomes_to_plot (dict): dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook. 
    Euler (list): list of all Euler values across the density range 
        
    
    """
    
    # Get absolute value of maximum density
    density_value = int(max_density * Euler_resolution)
    
    outcomes = TDA_Fernando.Eulerange_dens(matrix, density_value, 
                                           Euler_resolution, k_max)

    Euler = [i[0] for i in outcomes]
    total_cliques = [i[1] for i in outcomes]
    triangles = [i[5] for i in outcomes]

    outcomes_to_plot[f'Eulerrange_d{max_density}_{subject[:-4]}'] = Euler

    outcomes_to_export['Euler_sum'] = sum(Euler)
    outcomes_to_export['total_cliques_sum'] = sum(total_cliques)
    outcomes_to_export['triangles_sum'] = sum(triangles)

    return outcomes_to_export, outcomes_to_plot, Euler

@timer_func
def curvature(matrix, subject, *args, **kwargs):
    """ Calculates node curvatures at certain density values

    
    Parameters
    ----------
    matrix (numpy.array): connectivity matrix as numpy array
    subject (str): name of subject, needed for exporting data
    args: list of density values at which the sample the curvature values
        density value of 0.01 means 1%, density value of 0.1 means at 10% density
    kwargs: dictionary with density values which are sampled at 
        (a distance to) a phase transition. This means that each subject 
        will then have a different density value for sampling. 
        Values: density values. Keys: name of density value. E.g. p1: first
        phase transition, or p2: second phase transition.
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
         curvature mean, standard deviation, kurtosis, skewness and entropy
         are added, either for all nodes, or DMN nodes, or FPN nodes. 
    outcomes_to_plot: dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook. 
        
    
    """
    
    # Make dictionary for looping through all nodes, only DMN nodes, 
    # or only FPN nodes
    all_nodes = range(0, matrix.shape[0])
    dict_to_loop = {"all": all_nodes, "DMN": DMN, "FPN": FPN}

    # loop through fixed density values
    for dens in args:
        # calculate curvature at density value
        curv = TDA_Fernando.Curv_density(dens, matrix)
        # export for plotting
        outcomes_to_plot[f'curv_d{dens}_{subject[:-4]}'] = curv
        # calculate curvature outcomes (mean, sandard deviation, kurtosis, 
        # skewness, entropy) for all nodes, DMN nodes and FPN nodes
        for k, v in dict_to_loop.items():
            outcomes_to_export[f'curv_mean_{k}_{dens}'] = curv[v].mean()
            outcomes_to_export[f'curv_std_{k}_{dens}'] = np.std(curv[v])
            outcomes_to_export[f'curv_kur_{k}_{dens}'] = kurtosis(curv[v])
            outcomes_to_export[f'curv_skew_{k}_{dens}'] = skew(curv[v])
            outcomes_to_export[f'curv_ent_{k}_{dens}'] = entropy(abs(curv[v]))

    # Loop through kwargs separately to prevent overwriting other variables.
    for name, dens in kwargs.items():
        curv = TDA_Fernando.Curv_density(dens, matrix)
        for k, v in dict_to_loop.items():
            outcomes_to_export[f'curv_mean_{k}_{name}'] = curv[v].mean()
            outcomes_to_export[f'curv_std_{k}_{name}'] = np.std(curv[v])
            outcomes_to_export[f'curv_kur_{k}_{name}'] = kurtosis(curv[v])
            outcomes_to_export[f'curv_skew_{k}_{name}'] = skew(curv[v])
            outcomes_to_export[f'curv_ent_{k}_{name}'] = entropy(abs(curv[v]))

    return outcomes_to_export, outcomes_to_plot


def participation_cliques(matrix, clique_nr, *args, **kwargs):
    """ Calculates participation for DMN and FPN in n-cliques
    at chosen density values

    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
    clique_nr: integer value which sets the value for n. E.g.: 3 means
        that participation in 3-cliques/triangles is counted for each node
        in the network
    args: list of density values at which the sample the curvature values
    kwargs: dictionary with density values which are sampled at 
        (a distance to) a phase transition. This means that each subject 
        will then have a different density value for sampling. 
        Values: density values. Keys: name of density value. E.g. p1: first
        phase transition, or p2: second phase transition. 
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
         Participation in n-cliques is measured for DMN and FPN, is exported
         as the sum of participation in cliques and the entropy
        
    
    """
    
    for dens in args:
        # If dens is 0, set outcomes at 0 to prevent errors
        if dens == 0:
            p_cliques_sum_DMN = 0
            p_cliques_sum_FPN = 0
            DMN_clique_entropy = 0
            FPN_clique_entropy = 0
        else:
            # calculate particpation in cliques
            part_cliques = TDA_Fernando.Participation_in_cliques(dens,
                                                                 matrix,
                                                                 clique_nr,
                                                                 verbose=False)
            p_cliques_sum_DMN = part_cliques[DMN].sum()
            p_cliques_sum_FPN = part_cliques[FPN].sum()
            DMN_clique_entropy = p_cliques_sum_DMN * np.log(p_cliques_sum_DMN)
            FPN_clique_entropy = p_cliques_sum_FPN * np.log(p_cliques_sum_FPN)

        outcomes_to_export[f"p{clique_nr}cliques_DMN_ent_{dens}"] = DMN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_FPN_ent_{dens}"] = FPN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_DMN_sum_{dens}"] = p_cliques_sum_DMN
        outcomes_to_export[f"p{clique_nr}cliques_FPN_sum_{dens}"] = p_cliques_sum_FPN
    
    # Deo separately for kwargs to prevent overwriting other variables
    for name, dens in kwargs.items():
        if dens == 0:
            p_cliques_sum_DMN = 0
            p_cliques_sum_FPN = 0
            DMN_clique_entropy = 0
            FPN_clique_entropy = 0
        else:
            part_cliques = TDA_Fernando.Participation_in_cliques(dens,
                                                                 matrix,
                                                                 clique_nr,
                                                                 verbose=False)
            p_cliques_sum_DMN = part_cliques[DMN].sum()
            p_cliques_sum_FPN = part_cliques[FPN].sum()
            DMN_clique_entropy = p_cliques_sum_DMN * np.log(p_cliques_sum_DMN)
            FPN_clique_entropy = p_cliques_sum_FPN * np.log(p_cliques_sum_FPN)

        outcomes_to_export[f"p{clique_nr}cliques_DMN_ent_{name}"] = DMN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_FPN_ent_{name}"] = FPN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_DMN_sum_{name}"] = p_cliques_sum_DMN
        outcomes_to_export[f"p{clique_nr}cliques_FPN_sum_{name}"] = p_cliques_sum_FPN

    return outcomes_to_export


def preprocess_matrix(path_data, subject):
    """ Imports and preprocesses connectivity matrix
    
    
    Parameters
    ----------
    path_data: path to location of all connectivity matrices
    subject: name of current subject from data
    
    Returns
    -------
    matrix: preprocessed matrix as numpy array. This is the input for the 
        functions by Gudhi. 
    distance_matrix: preprocessed distance matrix. This is the input for 
        the functions by Fernando
     
    
    Notes
    -------
    If still present, any subcortical areas are deleted. Matrix is then 
    absolutized and converted to distance matrix, and diagonal filled with 1. 
    And short quality check: matrix with NaN values are printed. 
    
    """   
    
    # Import connectivity matrix
    matrix = import_cmatrix(path_data, subject)

    # Remove subcortical regions
    matrix = np.delete(matrix, subcortical, axis=0)
    matrix = np.delete(matrix, subcortical, axis=1)

    # Absolutize matrix
    matrix = abs(matrix)

    # Convert to distance matrix
    dist_matrix = 1 - matrix

    # Fill diagional with 1
    np.fill_diagonal(matrix, 1)
    np.fill_diagonal(dist_matrix, 1)

    # Quality check
    if np.any(np.isnan(matrix)):
        print(f'{subject} connectivity matrix contains nan value')

    return matrix, dist_matrix


def import_subnetworks(path_regions, path_region_names):
    """ Imports subnetworks
    
    
    Parameters
    ----------
    path_regions: path to file which contains all region names in the brain
    path_region_names: path to file which contains all names of subnetworks
        to which each region belongs
    
    Returns
    -------
    FPN: list of indexes of all regions which are part of FPN
    DMN: list of indexes of all regions which are part of DMN
    subcortical: list of indexes of all regions which are subcortical. 
        Necessary for excluding any subcortical regions from analysis. 
    
    """ 
    regions = pd.read_csv(path_regions, header=None)
    names = pd.read_csv(path_region_names, header=None)

    subnetworks = pd.DataFrame()
    subnetworks["region"] = list(regions[0])
    subnetworks["subnetwork"] = list(names[0])

    FPN = list(subnetworks[subnetworks['subnetwork'] == 'FP'].index.values)
    DMN = list(subnetworks[subnetworks['subnetwork'] == 'DMN'].index.values)
    subcortical = list(subnetworks[subnetworks['subnetwork'] == 'SC'].index.values)

    return FPN, DMN, subcortical


def export_plotting_data(plotting_data):
    """ THIS IS AN OLD FUNCTION. Replaced by export_plot_data, which exports
    the plotting data per subject separately in separate files. Prevents 
    loss of data when an error occurs. 
    
    Exports TDA features for plotting
    
    
    Parameters
    ----------
    plotting_data: list with all numpy arrays for plotting
    
    
    Returns
    -------
    Exports data per feature to plot as a npz file (multiple numpy arrays 
    zipped into a single file). 
    E.g. a file called curv_d0.005.npz, containing all numpy arrays
    with curvatures at density 0.005 for all subjects. 
    These npz files can be unpacked as a dictionary, so keys are the names
    of the subjects, e.g. curv_d0.005_subject1, while the value is the 
    numpy array with curvatures. 
     
    
    Notes
    -------
    Can be plotted using a Jupyter Notebook, see GitHub. 
    
    
    """   
    # Merge all data from all participants into one dictionary
    merged = {}
    for i in plotting_data:
        merged.update(i)

    # Make a list of all feature_names to enable grouping and exporting 
    # per feature
    feature_names = []
    for i in plotting_data[0].keys():
        elements = i.split("_")
        feature = str(f"{elements[0]}_{elements[1]}")
        feature_names.append(feature)
        
    # Per feature, make a new dictionary with only data from that feature
    for feature in feature_names:
        print(feature)
        # Make list with all keys to include in new dictionary. For example: 
        # for curv_d0.05, include curv_d0.005_subject1 and curv_d0.05_subject2
        all_keys = [key for key,value in merged.items() if feature in key]
        # Include data from merged with the keys selected above
        new_dict = {}
        for i in all_keys:
            new_dict[i] = merged[i]
        # Export dictionary as npz file
        savez_compressed(f'{path_plots}To_Plot_{feature}.npz', **new_dict)
    

def euler_peaks(Euler):
    """ Calculates the position of the Euler peaks / phase transitions
    
    
    Parameters
    ----------
    Euler: list with all Euler values for each density in density range
        
    
    Returns
    -------
    peaks: list of locations of Euler peaks
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
        Location of phase transition 1 and 2 (if present) is added
             
    Notes
    -------
    Only calculates up to the second phase transition. 
    
    """

    peaks = find_peaks(-np.log(np.abs(Euler)), prominence=1)[0]

    # If no peaks present, make phase transition 1 Nan value 
    # to prevent errors
    if len(peaks) == 0:
        outcomes_to_export['Phase_transition_1'] = float('NaN')
    if len(peaks) > 0: 
        outcomes_to_export['Phase_transition_1'] = float(peaks[0])
        
    # If only one peak present, make phase_transition 2 NaN value
    # to prevent errors

    if len(peaks) > 1:
        outcomes_to_export['Phase_transition_2'] = float(peaks[1])
    else:
        outcomes_to_export['Phase_transition_2'] = float('NaN')

    return peaks, outcomes_to_export


def phase_transition_densities(peaks):
    """ Calculates values of densities located around phase transition

    
    Parameters
    ----------
    peaks: list of locations of Euler peaks
    
    
    Returns
    -------
    pt_peak: dictionary with density values at fixed distances around 
        phase transitions. 
        
        
    Notes
    -------
    These distances to phase transitions can be set at different values,
    depending on how this affects performance (i.e. correlation with 
    cognition). Because during exploratory analyses we found that 
    TDA metrics which are sampled at phase transitions (e.g. curvature 
    or participation in cliques) perform better when sampled not at the 
    phase transition, but slightly around it. 
    
        
    """
    # Define absolute distances to phase transitions
    distances = {
        'd1': 0.00015,
        'd2': 0.0015,
        'd3': 0.015
        }
    
    # Make dictionary with phase transition names as keys and densities as
    # values. Make all values 0. 
    pt_peak = {}
    for k, v in distances.items(): 
        pt_peak[f'pt1_{v}_low'] = 0
        pt_peak[f'pt1_{v}_high'] = 0
        pt_peak[f'pt2_{v}_low'] = 0
        pt_peak[f'pt2_{v}_high'] = 0
    
    # If first phase transition present, change values 
    if len(peaks) == 1:
        for k, v in distances.items(): 
            pt_peak[f'pt1_{v}_low'] = abs(round(float(peaks)/1000 - v, 5))
            pt_peak[f'pt1_{v}_high'] = abs(round(float(peaks)/1000 + v, 5))
        
    # If two phase transitions present, change two values
    if len(peaks) > 1:
        for k, v in distances.items(): 
            pt_peak[f'pt1_{v}_low'] = abs(round(float(peaks[0])/1000 - v, 5))
            pt_peak[f'pt1_{v}_high'] = abs(round(float(peaks[0])/1000 + v, 5))
            pt_peak[f'pt2_{v}_low'] = abs(round(float(peaks[1])/1000 - v, 5))
            pt_peak[f'pt2_{v}_high'] = abs(round(float(peaks[1])/1000 + v, 5))

    return pt_peak


def export_plot_data(path_plots, export_filename, outcomes_to_plot):
    
    """ Exports data which can be plotted later
    
    
    Parameters
    ----------
    path_plots: path to directory with all plotting data
    export_filename: the name of the final TDA file. Used for naming the
        directory within the path_plots directory to put all the plotting
        files in
    outcomes_to_plot: a dictionary with the feature names as keys (e.g. 
        curv_d0.01_subject1) and a numpy array or list as value
        
    
    Returns
    -------
    Saves the plotting data in separate subfolders as a txt file. For example, 
    all files named curv_d0.01_subjectx are place in the folder curv_d0.01 with
    with their original names as file names (so within this folder are 
    curv_d0.01_subject1.csv and curv_d0.01_subjectx.csv)
        
        
    """
    
    # Make directory to put all plotting data 
    directory_name = export_filename[:-4] # Remove the .csv from filename
    if not os.path.exists(path_plots+directory_name):
        os.makedirs(path_plots+directory_name)
    
    # Also make subfolders to save data per featre in separate folder
    for feature, plot_matrix in outcomes_to_plot.items():
        # Get common feature names. E.g. from curv_d0.01_subject1 to curv_d0.01
        splitted = feature.split('_')
        feature_name = f'{splitted[0]}_{splitted[1]}'
        # If no directory with common feature name, make one
        path_feature = f'{path_plots}{directory_name}/{feature_name}'
        if not os.path.isdir(path_feature):
            os.makedirs(path_feature, exist_ok=True)

        # Save matrix in subfolder as txt file
        np.savetxt(f'{path_feature}/{feature}.csv', plot_matrix, delimiter=",")


def calculate_features(subject):
    """ Combines all functions above to calculate TDA features for each subject
    
    
    Parameters
    ----------
    subject: name of subject from data
        
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
        Location of phase transition 1 and 2 (if present) is added
    outcomes_to_plot: dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook.     
        Format: list of dictionaries. For each subject a dictionary is 
        generated. Key is names of variables to plot + name subject, value 
        is numpy array (in the case of curvatures) or list (in the case
        of Euler range)
    
    
    Notes
    -------
    
    
    """
    # print(subject)
    # Import and preproces connectivity matrix
    matrix, dist_matrix = preprocess_matrix(path_data, subject)

    # Perform persistence and persistence per dimension
    pers, simplex_tree = calculate_persistence(dist_matrix)
    pers_per_dim = persistence_per_dim(simplex_tree, nr_dimensions)
    
    # Calculate Gudhi TDA outcomes
    outcomes_to_export = betti_curves(pers_per_dim)
    outcomes_to_export = persistence_landscape(pers_per_dim)
    outcomes_to_export = topological_vector(pers_per_dim)
    outcomes_to_export = shannon_entropy(pers_per_dim)
    


    ##### Calculate noise-related features
    # Calculate Euler characteristic
    (outcomes_to_export, outcomes_to_plot, Euler
      ) = euler_characeristic(dist_matrix, subject, max_density=0.10,
                                       Euler_resolution=500, k_max=15)
                              
    # Calculate phase transitions and densities located around transitions                                     
    peaks, outcomes_to_export = euler_peaks(Euler)
    pt_peak = phase_transition_densities(peaks)
    
    (outcomes_to_export, outcomes_to_plot
      ) = curvature(dist_matrix, subject, *curvatures_to_plot)
    # Calculate and save curvatures

    (outcomes_to_export, outcomes_to_plot
      ) = curvature(dist_matrix, subject, **pt_peak)
    
    # Calculate and save participation in 3-cliques
    outcomes_to_export = participation_cliques(dist_matrix, 3, **pt_peak)
    # Calculate and save participation in 4-cliques
    outcomes_to_export = participation_cliques(dist_matrix, 4, **pt_peak)
    
    ##### Calculate noise-related features
    # Calculate Euler characteristic
    # (outcomes_to_export, outcomes_to_plot, Euler
    #   ) = euler_characeristic(matrix, subject, max_density=0.50,
    #                                    Euler_resolution=100, k_max=10)
                              
    # Calculate phase transitions and densities located around transitions                                     
    # peaks, outcomes_to_export = euler_peaks(Euler)
    # pt_peak = phase_transition_densities(peaks)
    
    # # Calculate and save curvatures
    # (outcomes_to_export, outcomes_to_plot
    #   ) = curvature(matrix, subject, *curvatures_to_plot,**pt_peak)
    
    # # Calculate and save participation in 3-cliques
    # outcomes_to_export = participation_cliques(matrix, 3, **pt_peak)
    # # Calculate and save participation in 4-cliques
    # outcomes_to_export = participation_cliques(matrix, 4, **pt_peak)
    
    # Add name Subject to dictionary to export
    outcomes_to_export['Subject'] = subject[:-4] 
    
    export_TDA_data(path_export, export_filename, outcomes_to_export)
    export_plot_data(path_plots, export_filename, outcomes_to_plot)
    
    return outcomes_to_export, outcomes_to_plot


####################
# Run Functions    #
####################

# Ignore divide by infinity error 
np.seterr(divide='ignore', invalid='ignore')

# Specify paths
path_data = '/mnt/resource/m.schepers/Connectivity_Matrices/HCP_REST1_conn_mat/'
path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
path_test = '/data/KNW/KNW-stage/m.schepers/HCP/Data/Cog_data/Cog_All_test.csv'
# Paths for exporting
path_export = '/mnt/resource/b.maciel/minne_code_review/'
path_plots = '/mnt/resource/b.maciel/minne_code_review/plotting/'
export_filename = 'Test_data_run_review.csv'

# Set variables
nr_dimensions = 2 # number of dimensions in filtration process to analyze
resolution = 100 # resolution for calculating area under curve
curvatures_to_plot = [0.005, 0.01, 0.02, 0.05, 0.10] # fixed densities for plotting
# curvatures_to_plot = [0.005, 0.01]
# and calculating curvatures
density_Euler = 0.10 # In percentage, e.g. 0.10 means 10% density
Euler_resolution = 100
n_workers = 10 # number of cores to run scripts on 

# Import subnetworks
FPN, DMN, subcortical = import_subnetworks(path_regions, path_region_names)

# Import data
data = import_data(path_data, remove_completed=True)
# data = ['HCA7552478.csv', 'HCA9546594.csv', 'HCA7056264.csv',
#       'HCA6375275.csv']
# data = ['HCA7552478.csv']

test = pd.read_csv(path_test)
test_subjects = test['subject']
test_subjects = [i + '.csv' for i in test_subjects]
only_test = [i for i in data if i in test_subjects]
data = only_test

# Create variables for exporting
outcomes_to_export = {}
outcomes_to_plot = {}

# Print basic statistics for keeping progress
print(f'N = {len(data)}, n_workers = {n_workers}', flush=True)
print("--- Generating features ---", flush=True)

# Run code, tqdm will show progress bar
pool = WorkerPool(n_jobs=n_workers)
results = list(tqdm.tqdm(pool.imap_unordered(calculate_features, data),
                          total=len(data)))

plot_data = [i[1] for i in results] 


