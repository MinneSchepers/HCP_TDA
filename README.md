# HCP_TDA
 Contains code for analyzing HCP data


# Create file for cognitive/behavioral data
Combine all behavioral data files from the HCP project into a single behavioral data file:
Files:
- Make_cog_file_HCP.ipynb

# Split data into exploration/training set and testing set.
Splitting is done based on age and on years of education. Data check based on visual inspection of distributions of these variables in both groups, as well as Mann Whitney U test.
Files:
- Split_HCP_data_Females.ipynb
- Split_HCP_data_Males.ipynb

# fMRI-timeseries to 4 types of correlation matrices:
1. Correlation matrices
o Calculate_con_matrices.py
2. Filtered correlation matrices.
Is created in three steps:
- Calculate correlation matrices (is bit different than in calculate_con_matrices.py, so is a necessary step) Calculate_con_matrices_pvalued_corrmats.py
- Calculate 100 randomized correlation matrices for comparisons:
Calculate_con_matrices_pvalued.py
- Create filtered correlation matrices with only significant links:
Calculate_con_matrices_pvalued_filtered.py
3. Mutual information matrices:
Creates correlation matrices with mutual information scores
Calculate_con_matrices_MI.py
4. Filtered mutual information matrices
Creates mutual information matrices with only significant links
To be created

# Produce TDA features from correlation matrices:
Input is correlation matrices, as well as files for the region names and subnetworks. Output is csv file with all TDA features for all persons, and npz files for numpy arrays, which can then be plotted afterwards using the HCP_Data_checking_plotting.ipynb file.
Files:
- TDA_features_HCP.py
- TDA_features_HCP_MI_females.py
- TDA_features_HCP_MI_males.py
- TDA_features_HCP_pvalued.py
- TDA_features_HCP_pvalued_Females.py
- TDA_features_HCP_pvalued_Males.py

# Create dataframes with TDA and cognitive data
Files:
- Construct_dataframe.ipynb

# Data exploration
Look for significant predictors of WM and EF using automatized codes from the Functions.ipynb file.
Files:
- Functions.ipynb
- HCP_exploratory_Analysis_xxx.ipynb
