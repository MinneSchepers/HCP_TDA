#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A short description or title.

   Longer description.

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

# Internal imports ### (Put here imports that are related to internal codes from the lab)
import TDA_Fernando


####################
# Functions        #
####################


def import_data(path_dir):
    
    data = []
    for filename in os.listdir(path_dir):
        data.append(filename)

    return data


def import_cmatrix(path_dir, person):
    path = path_dir + person
    cmatrix = np.loadtxt(path, skiprows=1)

    if cmatrix.shape[0] != cmatrix.shape[1]:
        raise Exception(
            f'Error: imported connectitivy matrix is not square: {cmatrix.shape} \n'
            f'Consider removing the top rows if it contains a header \n'
            f'See function import_cmatrix: adjust skiprows'
            )

    return cmatrix


def make_df(results):
    df_export = pd.DataFrame()

    # Apppend each index of results as row of df
    for i in results:
        df_export = df_export.append(i, ignore_index=True)

    # Bring subject name forward to first column of dataframe
    df_columns = df_export.columns.tolist()
    df_columns.insert(0, df_columns.pop(df_columns.index('Subject')))
    df_export = df_export.reindex(columns=df_columns)

    return df_export


def calculate_persistence(matrix):
    rips_complex = gd.RipsComplex(distance_matrix=matrix, max_edge_length=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=nr_dimensions)
    diag = simplex_tree.persistence()

    return rips_complex, diag, simplex_tree


def persistence_per_dim(tr, nr_dimension):
    # Calculates persistence per dimension.
    # Outputs list with all persistences per dimension.

    diag_per_dim = []

    for dimension in range(nr_dimensions):
        diag = tr.persistence_intervals_in_dimension(dimension)

        # The code below replaces any infinite values with the maximum value
        # from the persistence in dimension 0.

        # Find the maximum value which is not infinite
        if dimension == 0:
            diag = np.where(np.isinf(diag), -1, diag)
            max_val_dim0 = np.max(diag)

        # If non-empty array: replace infinite with maximum value
        if diag.shape != (0,):
            diag = np.where(np.isinf(diag), -1, diag)
            diag = np.where(diag == -1, max_val_dim0, diag)
            diag_per_dim.append(diag)

        # if no topological structures present, append empty array to prevent error
        else:
            diag_per_dim.append(np.zeros([0, 2]))

    return diag_per_dim


def calculate_betti_curves(diag_per_dim):

    # Calculate Betti curves per dimension
    betti_curves = BettiCurve(resolution=resolution,
                              sample_range=[0, 1]).transform(diag_per_dim)

    return betti_curves


def calculate_betti_curves_AUC(betti_curves):

    # Calculates area under curve for each Betti curve
    for i in range(nr_dimensions):
        outcomes_to_export[f'bc_AUC_dim{i}'] = metrics.auc(
            np.linspace(0, resolution, num=resolution), betti_curves[i])

    return outcomes_to_export


def calculate_persistence_landscape(diag_list):

    landscapes = Landscape(num_landscapes=1,
                           resolution=resolution).fit_transform(diag_list)

    return landscapes


def calculate_persistence_landscape_AUC(landscapes):

    for i in range(nr_dimensions):
        outcomes_to_export[f'pl_AUC_dim{i}'] = metrics.auc(
            np.linspace(0, resolution, num=resolution), landscapes[i])

    return outcomes_to_export


def calculate_TopologicalVector(diag_per_dim):

    for i in range(nr_dimensions):
        # If no topolofical components present, set to 0 to prevent errors
        if diag_per_dim[i].size == 0:
            outcomes_to_export[f'top_vec_dim{i}'] = 0

        else:
            diag = [diag_per_dim[i]]
            tv = TopologicalVector(threshold=1).transform(diag)
            outcomes_to_export[f'top_vec_dim{i}'] = float(tv)

    return outcomes_to_export


def calculate_ShannonEntropy(diag_per_dim):

    for i in range(nr_dimensions):
        # If no topolofical components present, set to 0 to prevent errors
        if diag_per_dim[i].size == 0:
            outcomes_to_export[f'S_entropy_dim{i}'] = 0

        else:
            diag = [diag_per_dim[i]]
            entropy = Entropy(mode='scalar').fit_transform(diag)
            outcomes_to_export[f'S_entropy_dim{i}'] = float(entropy)

    return outcomes_to_export


def calculate_EulerCharaceristic(matrix, person, max_density):

    # Calculates outcomes using the function made by Fernando across a density range.
    outcomes = TDA_Fernando.Eulerange_dens(matrix, max_density)

    Euler = [i[0] for i in outcomes]
    total_cliques = [i[1] for i in outcomes]
    triangles = [i[5] for i in outcomes]

    outcomes_to_plot[f'Eulerrange_d{max_density}_{person[4:10]}'] = Euler

    outcomes_to_export['Euler_sum'] = sum(Euler)
    outcomes_to_export['total_cliques_sum'] = sum(total_cliques)
    outcomes_to_export['triangles_sum'] = sum(triangles)

    return outcomes_to_export, outcomes_to_plot, Euler


def calculate_Euler_for_plotting(matrix, person, max_density):

    outcomes = TDA_Fernando.Eulerange_dens(matrix, max_density)
    Euler = [i[0] for i in outcomes]

    outcomes_to_plot[f'Eulerrange_d{max_density}_{person[4:10]}'] = Euler

    return outcomes_to_plot


def calculate_curvature(matrix, person, *args, **kwargs):
    all_nodes = range(0, matrix.shape[0])
    dict_to_loop = {"all": all_nodes, "DMN": DMN, "FPN": FPN}

    for dens in args:
        curv = TDA_Fernando.Curv_density(dens, matrix)
        outcomes_to_plot[f'curv_d{dens}_{person[4:10]}'] = curv
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


def calculate_curvature_for_plotting(matrix, person, *args):
    for dens in args:
        curv = TDA_Fernando.Curv_density(dens, matrix)
        outcomes_to_plot[f'curv_d{dens}_{person[4:10]}'] = curv

    return outcomes_to_plot


def calculate_cliques(matrix, clique_nr, *args, **kwargs):
    for dens in args:
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

        outcomes_to_export[f"p{clique_nr}cliques_DMN_ent_{dens}"] = DMN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_FPN_ent_{dens}"] = FPN_clique_entropy
        outcomes_to_export[f"p{clique_nr}cliques_DMN_sum_{dens}"] = p_cliques_sum_DMN
        outcomes_to_export[f"p{clique_nr}cliques_FPN_sum_{dens}"] = p_cliques_sum_FPN

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
    matrix = import_cmatrix(path_data, person)

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
    regions = pd.read_csv(path_regions, header=None)
    names = pd.read_csv(path_region_names, header=None)

    subnetworks = pd.DataFrame()
    subnetworks["region"] = list(regions[0])
    subnetworks["subnetwork"] = list(names[0])

    FPN = list(subnetworks[subnetworks['subnetwork'] == 'FP'].index.values)
    DMN = list(subnetworks[subnetworks['subnetwork'] == 'DMN'].index.values)
    subcortical = list(subnetworks[subnetworks['subnetwork'] == 'SC'].index.values)
    FPN_names = list(subnetworks['region'][FPN])
    DMN_names = list(subnetworks['region'][DMN])

    all_node_names = list(subnetworks['subnetwork'])

    return FPN, DMN, subcortical, FPN_names, DMN_names, all_node_names


def export_plotting_data(results):
    merged = {}
    for i in results:
        merged.update(i)

    feature_names = []
    for i in results[0].keys():
        elements = i.split("_")
        feature = str(f"{elements[0]}_{elements[1]}")
        feature_names.append(feature)

    for feature in feature_names:

        all_keys = [key for key, value in merged.items() if feature in key]

        new_dict = {}
        for i in all_keys:
            new_dict[i] = merged[i]

        savez_compressed(f'{path_plots}To_Plot_{feature}.npz', **new_dict)


def calculate_Euler_peaks(Euler):
    # Calculates the location of the Euler peaks,
    # for the first and second peak/phase transition
    peaks = find_peaks(-np.log(np.abs(Euler)), prominence=1)[0]
    outcomes_to_export['Phase_transition_1'] = float(peaks[0])
    # If only one peak present, make phase_transition 2 NaN value
    # to prevent errors
    if len(peaks) > 1:
        outcomes_to_export['Phase_transition_2'] = float(peaks[1])
    else:
        outcomes_to_export['Phase_transition_2'] = float('NaN')

    return peaks, outcomes_to_export


def get_phase_transition_peaks(peaks):

    all_peaks = {}

    if len(peaks) > 1:
        peak1_low = round(peaks[0]/1000 - 0.0015, 5)
        peak1_high = round(peaks[0]/1000 + 0.0015, 5)
        peak2_low = round(peaks[1]/1000 - 0.015, 5)
        peak2_high = round(peaks[1]/1000 + 0.015, 5)

        all_peaks['p1low'] = peak1_low
        all_peaks['p1high'] = peak1_high
        all_peaks['p2low'] = peak2_low
        all_peaks['p2high'] = peak2_high

    else:
        peak1_low = round(float(peaks)/1000 - 0.0015, 5)
        peak1_high = round(float(peaks)/1000 + 0.0015, 5)

        all_peaks['p1low'] = peak1_low
        all_peaks['p1high'] = peak1_high
        all_peaks['p2low'] = 0
        all_peaks['p2high'] = 0

    return all_peaks


def calculate_features(person):
    global outcomes_to_export
    outcomes_to_export = {}

    matrix = preprocess_matrix(path_data, person)

    rips_complex, diag, simplex_tree = calculate_persistence(matrix)
    diag_per_dim = persistence_per_dim(simplex_tree, nr_dimensions)

    betti_curves = calculate_betti_curves(diag_per_dim)
    outcomes_to_export = calculate_betti_curves_AUC(betti_curves)

    landscapes = calculate_persistence_landscape(diag_per_dim)
    outcomes_to_export = calculate_persistence_landscape_AUC(landscapes)

    outcomes_to_export = calculate_TopologicalVector(diag_per_dim)
    outcomes_to_export = calculate_ShannonEntropy(diag_per_dim)

    (outcomes_to_export, outcomes_to_plot, Euler
     ) = calculate_EulerCharaceristic(matrix, person,
                                      max_density=density_Euler)

    peaks, outcomes_to_export = calculate_Euler_peaks(Euler)
    pt_peak = get_phase_transition_peaks(peaks)

    (outcomes_to_export, outcomes_to_plot
     ) = calculate_curvature(matrix, person,
                             *curvatures_to_plot,
                             p1low=pt_peak['p1low'],
                             p1high=pt_peak['p1high'])

    outcomes_to_export = calculate_cliques(matrix, 3,
                                           p1low=pt_peak['p1low'],
                                           p1high=pt_peak['p1high'],
                                           p2low=pt_peak['p2low'],
                                           p2high=pt_peak['p2high'])

    outcomes_to_export = calculate_cliques(matrix, 4,
                                           p2low=pt_peak['p2low'],
                                           p2high=pt_peak['p2high'])

    outcomes_to_export['Subject'] = person[:-4]

    return outcomes_to_export, outcomes_to_plot


####################
# Run Functions    #
####################


# Specify paths
path_data = '/data/KNW/KNW-stage/m.schepers/HCP/HCP_REST1_corr_mat/'
path_export = '/data/KNW/KNW-stage/m.schepers/HCP/TDA_features_HCP_Females_Train.csv'
path_plots = '/data/KNW/KNW-stage/m.schepers/HCP/Plots/'
path_regions = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_region_names_full.txt'
path_region_names = '/data/KNW/f.tijhuis/Atlases_CIFTI/Glasser/Cortical+Freesurfer/GlasserFreesurfer_subnet_order_names.txt'
path_females = '/data/KNW/KNW-stage/m.schepers/HCP/Cog_data/females_train.csv'

# Set variables
np.seterr(divide='ignore', invalid='ignore')
nr_dimensions = 2
resolution = 100
curvatures_to_plot = [0.005, 0.01, 0.02, 0.05]
density_Euler = 100
n_workers = 10

# Import subnetworks
(FPN, DMN, subcortical, FPN_names, DMN_names, all_node_names
 ) = import_subnetworks(path_regions, path_region_names)

# Import data
data = import_data(path_data)
females = pd.read_csv(path_females)
female_subjects = females['subject']
female_subjects = [i + '.csv' for i in female_subjects]
only_females = [i for i in data if i in female_subjects]
data = only_females

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
