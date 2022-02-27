# KMFInitializedChimericE_Databasematching

This repository contains the code used to generate synthetic datasets and the implementation of proposed Chimeric encoder method on synthetic and real datasets.
Real datasets are not available here due to privacy issues.

Following are the file names and how to use them:

1) Synthetic_data_generation_code.py : This file contains the parameters and the seeds used to generate the synthetic datasets. Please change the dataset number in the code based on your needs.
2) Syn_data_Correlation matrices.py : Generates correlation heat maps from the generated synthetic data. Need to change the dataset number and path of the data files.
3) ACTFAST_Real_Data_exploration.py : Generates correlation heat map for ACTFAST data. 
4) MIMIC__correlation_plot.py : Generates correlation heat map for processed MIMIC-III dataset.
5) Syn_data_implementation_codes/Final_Chimeric2stage_vsothers_vs_Mappedfeatures_Syn_Perm.py :  Evaluates the performance of Chimeric Encoder and KMF in comparison to Kang method (https://pages.cs.wisc.edu/~naughton/includes/papers/schemaMatching.pdf) and RadialGAN (https://arxiv.org/abs/1802.06403) in bijective mapping case. Input (argumentative) is dataset number and whether the experiment is for continuous (c) or binary (b) data. Make sure the path for input files generated in (1) is correct. The output would be saved in the directories that are generated in the code. Other values can be played with too. Major output needed would be the comparison plots and the F1 scores from all the combinations of the trials and permutaions across all the fixed mapped features.
6) Syn_data_implementation_codes/Final_Chimeric2stage_vsothers_vsSamplesize_Syn_Perm.py : Evaluates the performance of Chimeric Encoder and KMF in comparison to Kang and RadialGAN as the number of sample size increases (bijective mapping case). Input and output are same as (5).
7) Syn_data_implementation_codes/Final_Chimeric2stage_vs_others_PartialMapping_Syn.py : Evaluates the performance of Chimeric Encoder and KMF in comparison to Kang in the partial and onto mapping case.  For onto mapping, the variable 'num_xtra_feat_inX1' is zero. Input same as (5). The output plots denote the change in F1 score as the difference between the size of two databases increases. In onto case, feature set of one database is a subset but in partial mapping case, the two databases do have non-overlapping features too.
8)Syn_data_implementation_codes/Final_Chimeric2stage_vsOthers_vs_Mappedfeatures_Syn_Perm_SqTran.py : Same evaluation as (5) when some of features ('num_feat_sq_trans') have been squared tranformed in the second dataset. Input output remains the same.
9) Syn_data_implementation_codes/Final_Chimeric2stage_vsothers_vs_Mappedfeatures_Syn_Perm_Binarized.py : Same as (5) when the continuous features have been binarized (use 'b' as the second argument) by thresholding the data at 0. Input and output remains the same. 

Package dependencies : torch, numpy, sklearn, pandas, matplotlib, matching, pingouin


ACTFAST_data_implementation_codes directory containes the python scripts used to run the similar experiments as in the case of syn data.

MIMI_data_implementation_codes contains 1) text file that has the sql commands used to extract the data from psql database, 2) jupyter notebook to format the extracted data based on certain constraints, 3) python script to perform exhaustive hyperparameter tuning, 4) python script to perform the roundrobin hyperparameter tuning using only mapped features, and 5) python script to that compares the performance of Chimeric encoder, KMF and Kang method on the partial mapping case of MIMIC dataset.
