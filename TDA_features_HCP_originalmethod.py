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


def import_cmatrix(path_dir, subject):
    """Imports single connectivity matrix
    
    
    Parameters
    ----------
    path_dir: path to directory with all connectivity matrices
    subject: a single subject from the data-list
    
    
    Returns
    -------
    matrix: connectivity matrix as numpy array
        
    
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


def make_df(TDA_features):
    """ Constructs a dataframe with all TDA features for all subjects 
    for exporting
    
    
    Parameters
    ----------
    TDA_features: list with all TDA features for one subject
        
    
    Returns
    -------
    df_export: pandas dataframe with all subjects as rows and their 
               corresponding TDA data in columns
        
    """
    
    df_export = pd.DataFrame()

    # Apppend each index of results as row of df
    for i in TDA_features:
        df_export = df_export.append(i, ignore_index=True)

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
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
        bc_AUC_dim: the area under the curve for the betti curve per 
            dimension. 
        bc_max_dim: the maximum of the betti curve per dimension. 
         
        
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


def calculate_persistence_landscape(pers_per_dim):
    """ Calculates persistence landscape per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
        
    
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


def calculate_TopologicalVector(pers_per_dim):
    """ Calculates topological vector per dimension

    
    Parameters
    ----------
    pers_per_dim: list of persistence per dimension
    
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


def calculate_ShannonEntropy(pers_per_dim):
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


def calculate_EulerCharaceristic(matrix, subject, max_density):
    """ Calculates Euler characteristic across a density range

    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
    subject: name of subject, needed for exporting data
    max_density: integer which sets the max of the density range for the Euler
        function
    
    
    Returns
    -------
    outcomes_to_export: dictionary with all TDA outcomes per subject. 
         The sum of the Euler characteristic per subject, the total number of 
         cliques and total number of triangles are added as a column
         per subject
    outcomes_to_plot: dictionary with outcomes to be exported which can be 
        plotted later using a Jupyter Notebook. 
    Euler: list of all Euler values across the density range 
        
    
    """
    outcomes = TDA_Fernando.Eulerange_dens(matrix, max_density)

    Euler = [i[0] for i in outcomes]
    total_cliques = [i[1] for i in outcomes]
    triangles = [i[5] for i in outcomes]

    outcomes_to_plot[f'Eulerrange_d{max_density}_{subject[4:10]}'] = Euler

    outcomes_to_export['Euler_sum'] = sum(Euler)
    outcomes_to_export['total_cliques_sum'] = sum(total_cliques)
    outcomes_to_export['triangles_sum'] = sum(triangles)

    return outcomes_to_export, outcomes_to_plot, Euler


def calculate_curvature(matrix, subject, *args, **kwargs):
    """ Calculates node curvatures at certain density values

    
    Parameters
    ----------
    matrix: connectivity matrix as numpy array
    subject: name of subject, needed for exporting data
    args: list of density values at which the sample the curvature values
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
        outcomes_to_plot[f'curv_d{dens}_{subject[4:10]}'] = curv
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
    """ Exports TDA features for plotting
    
    
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
        # Make list with all keys to include in new dictionary. For example: 
        # for curv_d0.05, include curv_d0.005_subject1 and curv_d0.05_subject2
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


def get_phase_transition_peaks(peaks):
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
    distance_p1 = 0.015
    distance_p2 = 0.15
    
    pt_peak = {}
    
    # If no phase transitions present:
    if len(peaks) == 0:
        peak1_low = 0
        peak1_high = 0
        peak2_low = 0
        peak2_high = 0
    
    # If only first phase transition present: 
    if len(peaks) == 1:
        peak1_low = abs(round(float(peaks)/1000 - distance_p1, 5))
        peak1_high = abs(round(float(peaks)/1000 + distance_p1, 5))
        peak2_low = 0
        peak2_high = 0
    
    # If also second phase transition present: 
    if len(peaks) > 1:
        peak1_low = abs(round(peaks[0]/1000 - distance_p1, 5))
        peak1_high = abs(round(peaks[0]/1000 + distance_p1, 5))
        peak2_low = abs(round(peaks[1]/1000 - distance_p2, 5))
        peak2_high = abs(round(peaks[1]/1000 + distance_p2, 5))
        
    pt_peak['p1low'] = peak1_low
    pt_peak['p1high'] = peak1_high
    pt_peak['p2low'] = peak2_low
    pt_peak['p2high'] = peak2_high

    return pt_peak


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
    # Define outcomes_to_export for saving variables
    # global outcomes_to_export
    # outcomes_to_export = {}
    # outcomes_to_export = {}

    # Import and preproces connectivity matrix
    matrix, dist_matrix = preprocess_matrix(path_data, subject)

    # Perform persistence and persistence per dimension
    pers, simplex_tree = calculate_persistence(dist_matrix)
    pers_per_dim = persistence_per_dim(simplex_tree, nr_dimensions)
    
    # Calculate Gudhi TDA outcomes
    outcomes_to_export = calculate_betti_curves(pers_per_dim)
    outcomes_to_export = calculate_persistence_landscape(pers_per_dim)
    outcomes_to_export = calculate_TopologicalVector(pers_per_dim)
    outcomes_to_export = calculate_ShannonEntropy(pers_per_dim)

    # Calculate Euler characteristic
    (outcomes_to_export, outcomes_to_plot, Euler
      ) = calculate_EulerCharaceristic(matrix, subject, density_Euler)

    # Calculate phase transitions and densities located around transitions                                     
    peaks, outcomes_to_export = calculate_Euler_peaks(Euler)
    pt_peak = get_phase_transition_peaks(peaks)
    
    # Calculate and save curvatures
    (outcomes_to_export, outcomes_to_plot
      ) = calculate_curvature(matrix, subject, *curvatures_to_plot, **pt_peak)
    
    # Calculate and save participation in 3-cliques
    outcomes_to_export = calculate_cliques(matrix, 3, **pt_peak)
    # Calculate and save participation in 4-cliques
    outcomes_to_export = calculate_cliques(matrix, 4, **pt_peak)
    
    # Add name Subject to dictionary to export
    outcomes_to_export['Subject'] = subject[:-4] 

    return outcomes_to_export, outcomes_to_plot


####################
# Run Functions    #
####################

# Ignore divide by infinity error 
np.seterr(divide='ignore', invalid='ignore')

# Specify paths
path_data = '/data/KNW/KNW-stage/m.schepers/HCP/Connectivity_Matrices/HCP_REST1_conn_mat/'
path_export = '/data/KNW/KNW-stage/m.schepers/HCP/TDA_data/TDA_features_HCP_training_20_originalmethod.csv'
path_plots = '/data/KNW/KNW-stage/m.schepers/HCP/Plots/Training_all_test'
path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
path_train = '/data/KNW/KNW-stage/m.schepers/HCP/GitHub/All_exp_MI.csv'

# Set variables
nr_dimensions = 1 # number of dimensions in filtration process to analyze
resolution = 100 # resolution for calculating area under curve
curvatures_to_plot = [0.0005, 0.001] # fixed densities for plotting
# curvatures_to_plot = [0.005, 0.01]
# and calculating curvatures
# density_Euler = 100 # the maximum density of density range to calculate Euler
density_Euler = 20
n_workers = 10 # number of cores to run scripts on 

# Import subnetworks
FPN, DMN, subcortical = import_subnetworks(path_regions, path_region_names)

# Import data
data = import_data(path_data)
# data = data[0:5]
data = ['HCA9546594.csv', 'HCA7552478.csv', 'HCA6119257.csv', 'HCA9044570.csv',
        'HCA7907287.csv', 'HCA8623581.csv', 'HCA7056264.csv', 'HCA6606571.csv',
        'HCA7502059.csv', 'HCA8657598.csv', 'HCA7764594.csv', 'HCA7268178.csv',
        'HCA9688312.csv', 'HCA6937998.csv', 'HCA6120646.csv', 'HCA9099999.csv',
        'HCA8126971.csv', 'HCA6750477.csv', 'HCA6474782.csv', 'HCA6668391.csv']


train = pd.read_csv(path_train)
train_subjects = train['Subject']
train_subjects = [i + '.csv' for i in train_subjects]
only_train = [i for i in data if i in train_subjects]
data = only_train

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

# Export plotting data
plotting_data = [i[1] for i in results]
export_plotting_data(plotting_data)

# Export TDA features
TDA_features = [i[0] for i in results]
df_export = make_df(TDA_features)
df_export.to_csv(path_export, index=False)
