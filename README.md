# HCP_TDA

This repository contains code for topological analysis of HCP data. This readme contains, step by step, the codes we used for our project. We focused on the HCP aging database. 

# 1 - Create file for cognitive/behavioral data
We combined all behavioral data files from the HCP project into a single behavioral data file:

- Make_cog_file_HCP.ipynb

# 2 - Split data into exploration/training set and testing set.
We created two groups, training and validation sets. We performed feature selection in the training data set, and tested our findings in the validation dataset. We splitted the database based on age and on years of education. Data check based on visual inspection of distributions of these variables in both groups, as well as Mann Whitney U test.

Files:
- Split_HCP_data_Females.ipynb
- Split_HCP_data_Males.ipynb

# 3 - From the fMRI-timeseriesm we created 4 types of functional connectivity matrices:
1. Correlation matrices - i.e., Pearson connectivity matrices: 
o Connectivity_matrices.py
2. Filtered pvalued correlation matrices.
Is created in three steps:
- Calculate correlation matrices (is bit different than in Connectivity_matrices.py, so is a necessary step) 
Connectivity_matrices_pvalued.py
- Calculate 100 randomized correlation matrices for comparisons:
Connectivity_matrices_pvalued_random.py
- Create filtered correlation matrices with only significant links:
Connectivity_matrices_pvalued_filtered.py
3. Mutual information matrices:
Creates correlation matrices with mutual information scores
Connectivity_matrices_MI.py
4. Filtered mutual information matrices
Creates mutual information matrices with only significant links
To be created

# Produce TDA features from correlation matrices:
Input is correlation matrices, as well as files for the region names and subnetworks. Output is csv file with all TDA features for all persons, and npz files for numpy arrays, which can then be plotted afterwards using the HCP_Data_checking_plotting.ipynb file.
Files:
- TDA_features_HCP.py

# Create dataframes with TDA and cognitive data
Files:
- Construct_dataframe.ipynb

# Data exploration
Look for significant predictors of WM and EF using automatized codes from the Functions.ipynb file.
Files:
- Functions.ipynb
- HCP_exploratory_Analysis_xxx.ipynb
