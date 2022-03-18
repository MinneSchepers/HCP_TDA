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

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
import os

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import gudhi as gd # version 3.3.0
from gudhi.representations import Landscape # version 3.3.0
from gudhi.representations import BettiCurve # version 3.3.0
from gudhi.representations import TopologicalVector # version 3.3.0
from gudhi.representations import Entropy # version 3.3.0
#from mpire import WorkerPool # version 2.3.3
import numpy as np # version 1.20.3
from numpy import savez_compressed # version 1.20.3
import pandas as pd # version 1.4.1
from scipy.signal import find_peaks # version 1.6.2
from scipy.stats import kurtosis # version 1.6.2
from scipy.stats import skew # version 1.6.2
from scipy.stats import entropy # version 1.6.2
from sklearn import metrics # version 1.0.2
import tqdm # version 4.55.0

# Internal imports ### (Put here imports that are related to internal codes from the lab)
import TDA_Fernando

# FIne

####################
# Functions        #
####################


def import_data(path_dir):
    """Makes list of all connectivity matrices to include
    
    
    Parameters
    ----------
    path_dir: path to directory with all connectivity matrices
        
    
    Returns
    -------
    data: list containing all names of connectivity matrices
        
    """
    data = []
    for filename in os.listdir(path_dir):
        data.append(filename)

    return data

# need to create a files list to open the person
def import_cmatrix(path_dir, person):
    """Imports single connectivity matrix
    
    
    Parameters
    ----------
    path_dir: path to directory with all connectivity matrices
    person: a single person from the data-list
    
    
    Returns
    -------
    matrix: connectivity matrix as numpy array
        
    
    Notes
    ----------
    Checks if the connectivity matrix is square (because some datasets
    have a header which needs to be excluded), if not, raises exception
    with error message    
    
    """    
    path = path_dir + person
    matrix = pd.read_csv(path,header=None).to_numpy()###np.loadtxt(path)#, skiprows=1)

    #for filtered
    #matrix = pd.read_csv(path,header=None,sep=' ').to_numpy()###np.loadtxt(path)#, skiprows=1)

    if matrix.shape[0] != matrix.shape[1]:
        raise Exception(
            f'Error: imported connectitivy matrix is not square: {matrix.shape} \n'
            f'Consider removing the top rows if it contains a header \n'
            f'See function import_cmatrix: adjust skiprows'
            )

    return matrix

def make_df(TDA_features):
    """ Constructs a dataframe with all TDA features for all persons 
    for exporting
    
    
    Parameters
    ----------
    TDA_features: list with all TDA features for one person
        
    
    Returns
    -------
    df_export: pandas dataframe with all persons as rows and their 
               corresponding TDA data in columns
        
    """
    
    df_export = pd.DataFrame([TDA_features])

    # Apppend each index of results as row of df
    #for i in TDA_features:
    #    df_export = df_export.append(i, ignore_index=True)

    # Bring subject name forward to first column of dataframe
    df_columns = df_export.columns.tolist()
    df_columns.insert(0, df_columns.pop(df_columns.index('Subject')))
    df_export = df_export.reindex(columns=df_columns)

    return df_export


def calculate_persistence(matrix):
    """ Performs filtration (persistent homology) process 
    
    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
        
    
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


def calculate_betti_curves(pers_per_dim):
    """ Calculates Betti curves per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
        
    
    Returns
    -------
    betti_curves: list of Betti curves per dimension
         
        
    """

    betti_curves = BettiCurve(resolution=resolution,
                              sample_range=[0, 1]).transform(pers_per_dim)

    return betti_curves


def calculate_betti_curves_AUC(betti_curves):
    """ Calculates area under curve (AUC) for Betti curve per dimension

    
    Parameters
    ----------
    betti_curves: list of Betti curves
    resolution: integer for the number of samples to take AUC of 
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
        Betti curve AUC is added as column per dimension
         
        
    """
    for i in range(nr_dimensions):
        betti_curve_AUC = metrics.auc(
            np.linspace(0, resolution, num=resolution), betti_curves[i])
        outcomes_to_export[f'bc_AUC_dim{i}'] = betti_curve_AUC

    return outcomes_to_export


def calculate_persistence_landscape(pers_per_dim):
    """ Calculates persistence landscape per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
        
    
    Returns
    -------
    landscapes: list of persistence landscape per dimension
         
        
    """
    
    landscapes = Landscape(num_landscapes=1,
                           resolution=resolution).fit_transform(pers_per_dim)

    return landscapes


def calculate_persistence_landscape_AUC(landscapes):
    """ Calculates area under curve (AUC) for persistence landscapes
    per dimension

    
    Parameters
    ----------
    landscapes: list of persistence landscape per dimension
    resolution: integer for the number of samples to take AUC of 
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
        persistence landscaoe AUC is added as column per dimension
         
        
    """

    for i in range(nr_dimensions):
        landscape_AUC = metrics.auc(
            np.linspace(0, resolution, num=resolution), landscapes[i])
        outcomes_to_export[f'pl_AUC_dim{i}'] = landscape_AUC

    return outcomes_to_export


def calculate_TopologicalVector(pers_per_dim):
    """ Calculates topological vector per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
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


def calculate_ShannonEntropy(pers_per_dim):
    """ Calculates Shannon entropy per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
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


def calculate_EulerCharaceristic(matrix, person, max_density):
    """ Calculates Euler characteristic across a density range

    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
    person: name of person, needed for exporting data
    max_density: integer which sets the max of the density range for the Euler
        function
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
         The sum of the Euler characteristic per person, the total number of 
         cliques and total number of triangles are added as a column
         per person
    outcomes_to_plot: dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook. 
    Euler: list of all Euler values across the density range 
        
    
    """
    outcomes = TDA_Fernando.Eulerange_dens(matrix, max_density)

    Euler = [i[0] for i in outcomes]
    total_cliques = [i[1] for i in outcomes]
    triangles = [i[5] for i in outcomes]

    outcomes_to_plot[f'Eulerrange_d{max_density}_{person[4:10]}'] = Euler

    outcomes_to_export['Euler_sum'] = sum(Euler)
    outcomes_to_export['total_cliques_sum'] = sum(total_cliques)
    outcomes_to_export['triangles_sum'] = sum(triangles)

    return outcomes_to_export, outcomes_to_plot, Euler


def calculate_curvature(matrix, person, *args, **kwargs):
    """ Calculates node curvatures at certain density values

    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
    person: name of person, needed for exporting data
    args: list of density values at which the sample the curvature values
    kwargs: dictionary with density values which are sampled at 
        (a distance to) a phase transition. This means that each person 
        will then have a different density value for sampling. 
        Values: density values. Keys: name of density value. E.g. p1: first
        phase transition, or p2: second phase transition. 
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
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
        outcomes_to_plot[f'curv_d{dens}_{person[4:10]}'] = curv
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


def calculate_cliques(matrix, clique_nr, *args, **kwargs):
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
        (a distance to) a phase transition. This means that each person 
        will then have a different density value for sampling. 
        Values: density values. Keys: name of density value. E.g. p1: first
        phase transition, or p2: second phase transition. 
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
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


def preprocess_matrix(path_data, person):
    """ Imports and preprocesses connectivity matrix
    
    
    Parameters
    ----------
    path_data: path to location of all connectivity matrices
    person: name of current person from data
    
    Returns
    -------
    matrix: preprocessed matrix as numpy array
     
    
    Notes
    -------
    If still present, any subcortical areas are deleted. Matrix is then 
    absolutized and converted to distance matrix, and diagonal filled with 1. 
    And short quality check: matrix with NaN values are printed. 
    
    """   
    
    # Import connectivity matrix
    matrix = import_cmatrix(path_data, person)
    # Took out here
    # Remove subcortical regions
    matrix = np.delete(matrix, subcortical, axis=0)
    matrix = np.delete(matrix, subcortical, axis=1)

    # Absolutize matrix
    matrix = abs(matrix)

    # Convert to distance matrix
    matrix = 1 - matrix

    # Fill diagional with 1
    np.fill_diagonal(matrix, 1)

    # Quality check
    if np.any(np.isnan(matrix)):
        print(f'{person} connectivity matrix contains nan value')

    return matrix


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
    DMN=list(regions[regions[0] == 'DMN'].index.values)
    FPN=list(regions[regions[0] == 'FP'].index.values)
    #adapted
    #FPN = list(subnetworks[subnetworks['subnetwork'] == 'FP'].index.values)
    #DMN = list(subnetworks[subnetworks['subnetwork'] == 'DMN'].index.values)
    subcortical = list(subnetworks[subnetworks['subnetwork'] == 'SC'].index.values)

    return FPN, DMN, subcortical


def export_plotting_data(plotting_data):
    """ Exports TDA features for plotting
    
    
    Parameters
    ----------
    plotting_data: list with all numpy arrays for plotting
    
    
    Returns
    -------
    Exports data per feature to plot as a npz file (multiple numpy arrays 
    zipped into a single file). 
    E.g. a file called curv_d0.005.npz, containing all numpy arrays
    with curvatures at density 0.005 for all persons. 
    These npz files can be unpacked as a dictionary, so keys are the names
    of the persons, e.g. curv_d0.005_person1, while the value is the 
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
        # Make list with all keys to include in new dictionary. For example: 
        # for curv_d0.05, include curv_d0.005_person1 and curv_d0.05_person2
        all_keys = [key for key,value in merged.items() if feature in key]
        # Include data from merged with the keys selected above
        new_dict = {}
        for i in all_keys:
            new_dict[i] = merged[i]
        # Export dictionary as npz file 
        savez_compressed(f'{path_plots}To_Plot_{feature}.npz', **new_dict)
    

def calculate_Euler_peaks(Euler):
    """ Calculates the position of the Euler peaks / phase transitions
    
    
    Parameters
    ----------
    Euler: list with all Euler values for each density in density range
        
    
    Returns
    -------
    peaks: list of locations of Euler peaks
    outcomes_to_export: dictionary with all TDA outcomes per person. 
        Location of phase transition 1 and 2 (if present) is added
             
    Notes
    -------
    Only calculates up to the second phase transition. 
    
    """

    peaks = find_peaks(-np.log(np.abs(Euler)), prominence=1)[0]
    #peaks = np.where(Euler == np.min(((Euler))))[0][0]
    print(peaks)
    outcomes_to_export['Phase_transition_1'] = float(peaks[0])
    # If only one peak present, make phase_transition 2 NaN value
    # to prevent errors
    if len(peaks) > 1:
        outcomes_to_export['Phase_transition_2'] = float(peaks[1])
    else:
        outcomes_to_export['Phase_transition_2'] = float('NaN')

    return peaks, outcomes_to_export


def get_phase_transition_peaks(peaks):
    """ Calculates values of densities located around phase transition

    
    Parameters
    ----------
    peaks: list of locations of Euler peaks
    
    
    Returns
    -------
    all_peaks: dictionary with density values at fixed distances around 
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
    all_peaks = {}
    
    # If second phase transition also present: 
    if len(peaks) > 1:
        peak1_low = round(peaks[0]/1000 - 0.0015, 5)
        peak1_high = round(peaks[0]/1000 + 0.0015, 5)
        peak2_low = round(peaks[1]/1000 - 0.015, 5)
        peak2_high = round(peaks[1]/1000 + 0.015, 5)

        all_peaks['p1low'] = peak1_low
        all_peaks['p1high'] = peak1_high
        all_peaks['p2low'] = peak2_low
        all_peaks['p2high'] = peak2_high
    
    # If only first phase transition present: 
    else:
        peak1_low = round(float(peaks)/1000 - 0.0015, 5)
        peak1_high = round(float(peaks)/1000 + 0.0015, 5)

        all_peaks['p1low'] = peak1_low
        all_peaks['p1high'] = peak1_high
        all_peaks['p2low'] = 0
        all_peaks['p2high'] = 0

    return all_peaks


def calculate_features(person):
    """ Combines all functions above to calculate TDA features for each person
    
    
    Parameters
    ----------
    person: name of person from data
        
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per person. 
        Location of phase transition 1 and 2 (if present) is added
    outcomes_to_plot: dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook.     
        Format: list of dictionaries. For each person a dictionary is 
        generated. Key is names of variables to plot + name person, value 
        is numpy array (in the case of curvatures) or list (in the case
        of Euler range)
    
    
    Notes
    -------
    
    
    """
    # Define outcomes_to_export for saving variables
    # global outcomes_to_export
    # outcomes_to_export = {}

    # Import and preproces connectivity matrix
    matrix = preprocess_matrix(path_data, person)
    matrix_2 = np.abs(import_cmatrix(path_data,person))# preprocess_matrix(path_data, person)

    # Perform persistence and persistence per dimension
    pers, simplex_tree = calculate_persistence(matrix)
    pers_per_dim = persistence_per_dim(simplex_tree, nr_dimensions)
    
    # Calculate Betti curves and save area under curve
    betti_curves = calculate_betti_curves(pers_per_dim)
    outcomes_to_export = calculate_betti_curves_AUC(betti_curves)
    
    # Calculate persistence landscapes and save area under curve
    landscapes = calculate_persistence_landscape(pers_per_dim)
    outcomes_to_export = calculate_persistence_landscape_AUC(landscapes)
    
    # Calculate and save topological vector and Shannon entropy
    outcomes_to_export = calculate_TopologicalVector(pers_per_dim)
    outcomes_to_export = calculate_ShannonEntropy(pers_per_dim)

    # Calculate Euler characteristic
    (outcomes_to_export, outcomes_to_plot, Euler
     ) = calculate_EulerCharaceristic(matrix_2, person,
                                      max_density=density_Euler)

    # Calculate phase transitions and densities located around transitions                                     
    peaks, outcomes_to_export = calculate_Euler_peaks(Euler)
    pt_peak = get_phase_transition_peaks(peaks)

    # Calculate and save curvatures
    (outcomes_to_export, outcomes_to_plot
     ) = calculate_curvature(matrix, person,
                             *curvatures_to_plot,
                             p1low=pt_peak['p1low'],
                             p1high=pt_peak['p1high'])
    # Calculate and save participation in 3-cliques
    outcomes_to_export = calculate_cliques(matrix, 3,
                                           p1low=pt_peak['p1low'],
                                           p1high=pt_peak['p1high'],
                                           p2low=pt_peak['p2low'],
                                           p2high=pt_peak['p2high'])
    # Calculate and save participation in 4-cliques
    outcomes_to_export = calculate_cliques(matrix, 4,
                                           p2low=pt_peak['p2low'],
                                           p2high=pt_peak['p2high'])

    # Add name Subject to dictionary to export
    outcomes_to_export['Subject'] = person[:-4]

    return outcomes_to_export, outcomes_to_plot

# I need to open for one person and input stuff


####################
# Run Functions    #
####################

# # Ignore divide by infinity error 
np.seterr(divide='ignore', invalid='ignore')

# # Specify paths
#path_data = 'Filtered_Corr_Matrixes/'# Fernando - had to change for the Filtereded!!! #it was: 
path_data ='HCP_AAL_Young_Adult/functional_connectivity/'
#path_export = 'TDA_all_nonglob_Train.csv'
#path_plots = 'Results/'
path_regions = 'subnet_ordernames_AAL.txt'
#'/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = 'AAL_region_names_abbrev.txt'
#'/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
# path_train = '/data/KNW/KNW-stage/m.schepers/HCP/GitHub/All_exp.csv'


# path_data = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat/'
# path_export = '/data/KNW/KNW-stage/m.schepers/HCP/TDA_all_nonglob_Train.csv'
# path_plots = '/data/KNW/KNW-stage/m.schepers/HCP/Plots/Train_all_nonglob'
# path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
# path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
# path_train = '/data/KNW/KNW-stage/m.schepers/HCP/GitHub/All_exp.csv'

# # Set variables
# nr_dimensions = 2 # number of dimensions in filtration process to analyze
# resolution = 100 # resolution for calculating area under curve
# curvatures_to_plot = [0.005, 0.01, 0.02, 0.05] # fixed densities for plotting
# # curvatures_to_plot = [0.005, 0.01]
# # and calculating curvatures
# density_Euler = 100 # the maximum density of density range to calculate Euler
# # density_Euler = 10
# n_workers = 10 # number of cores to run scripts on 

# # Import subnetworks
# FPN, DMN, subcortical = import_subnetworks(path_regions, path_region_names)

# # Import data
# data = import_data(path_data)
# # data = data[0:100]
# # Only select females in this case
# train = pd.read_csv(path_train)
# train_subjects = train['subject']
# train_subjects = [i + '.csv' for i in train_subjects]
# only_train = [i for i in data if i in train_subjects]
# data = only_train

# # Create variables for exporting
# outcomes_to_export = {}
# outcomes_to_plot = {}

# # Print basic statistics for keeping progress
# print(f'N = {len(data)}, n_workers = {n_workers}', flush=True)
# print("--- Generating features ---", flush=True)

# # Run code, tqdm will show progress bar
# pool = WorkerPool(n_jobs=n_workers)
# results = list(tqdm.tqdm(pool.imap_unordered(calculate_features, data),
#                           total=len(data)))

# # Export plotting data
# plotting_data = [i[1] for i in results]
# export_plotting_data(plotting_data)

# # Export TDA features
# TDA_features = [i[0] for i in results]
# df_export = make_df(TDA_features)
# df_export.to_csv(path_export, index=False)




# # Set variables
nr_dimensions = 2 # number of dimensions in filtration process to analyze
resolution = 100 # resolution for calculating area under curve
# curvatures_to_plot = [0.005, 0.01, 0.02, 0.05] # fixed densities for plotting
# # curvatures_to_plot = [0.005, 0.01]
# # and calculating curvatures
#density_Euler = 10 # the maximum density of density range to calculate Euler: did 10 to go quick!!!
# # density_Euler = 10
# n_workers = 10 # number of cores to run scripts on 

# # Create variables for exporting
outcomes_to_export = {}
outcomes_to_plot = {}

density_Euler = 150 # the maximum density of density range to calculate Euler: did 10 to go quick!!!

curvatures_to_plot = [0.005, 0.01]



# # Import subnetworks
FPN, DMN, subcortical = import_subnetworks(path_regions, path_region_names)
files=import_data(path_data)
#
def f(data):
  try:
    return calculate_features(data)
  except Exception as e:
        return e

from mpire import WorkerPool # version 2.3.3
#files=import_data(path_data)
#pool = WorkerPool(n_jobs=14)
#results = list(tqdm.tqdm(pool.imap_unordered(f, files)))
#list_results=[]


#new_r=[]
#for i in results:
#    if type(i)==tuple:
#        new_r.append(i)

#for result in new_r:
    #temp=calculate_features(files[i])
#    df=make_df(result[0])
##    list_results.append(df)
#Final=pd.concat(list_results)
#Final.to_csv('TDA_features_full_all.csv', index=False)





#for result in results:
    #temp=calculate_features(files[i])
#    df=make_df(result[0])
#    list_results.append(df)
#Final=pd.concat(list_results)
#Final.to_csv('TDA_features_full.csv', index=False)





#from mpire import WorkerPool # version 2.3.3
#new_path_data='Filtered_Corr_Matrixes/'




pool = WorkerPool(n_jobs=12)
results = list(tqdm.tqdm(pool.imap_unordered(f, files[0:12])))
list_results=[]


new_r=[]
for i in results:
    if type(i)==tuple:
        new_r.append(i)

for result in new_r:
    #temp=calculate_features(files[i])
    df=make_df(result[0])
    list_results.append(df)
    
    
Final=pd.concat(list_results)
Final.to_csv('Test_TDA_features_all.csv', index=False)


##

#path_filt='Filtered_Corr_Matrixes/'
#files2=import_data(path_filt)

#pool = WorkerPool(n_jobs=12)
#results_f = list(tqdm.tqdm(pool.imap_unordered(calculate_features, files2)))
#list_results_f=[]
#for result in results_f:
    #temp=calculate_features(files[i])
#    df=make_df(result[0])
#    list_results.append(df)
#Final_f=pd.concat(list_results)
#Final_f.to_csv('TDA_features_filtered.csv', index=False)



