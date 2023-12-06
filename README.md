# KMFInitializedChimericE_Databasematching

This repository contains the code used to generate synthetic datasets and the implementation of proposed Chimeric encoder method on synthetic and real datasets.
Real datasets are not available here due to privacy issues.

Following are the file names and how to use them:

1) Synthetic_data_generation_code.py : This file contains the parameters and the seeds used to generate the synthetic datasets. Please change the dataset number in the code based on your needs.

2) Syn_data_Correlation matrices.py : Generates correlation heat maps from the generated synthetic data. Need to change the dataset number and path of the data files.

3) MIMIC_correlation_plot.py : Generates correlation heat map for processed MIMIC-III dataset.

4) F1_score_Significance_test_from_violin_plots_files.py : Python script that reads the result file containing F1 scores, MatchMSE score and ReconMSE scores from each combination that the code ran for (avalable in Results folder) and average out the final result. It also has an snippet for testing for the statistical significance of the presented results.

### Synthetic data and Non-EHR data implementation

1) CL_level1_multiple_runs.py : Evaluates the performance of all the proposed methods on different synthetic and non-EHR datasets (Nomao and Superconductivity). The baselines are Kang method (https://pages.cs.wisc.edu/~naughton/includes/papers/schemaMatching.pdf) and RadialGAN (https://arxiv.org/abs/1802.06403). This code can be used for experimenting the bijective mapping case, binarized bijective case and square transformed bijective case. The arguments for this setup are obtained by running 'CL_1-1_mapping_arg_based_HP_tuning.py' over the possible combinations given in 'bsub_command_*.txt' file based on the respective setting. The output would be saved in the directories that are generated in the code.

2)  CL_partial_matching_multiple_runs.py: Input and comparison is same as the bijective mapping but this is used for testing the onto and partial mapping performance of the methods. For onto mapping, the variable 'num_xtra_feat_inX1' is zero. The output plots denote the change in F1 score as the difference between the size of two databases increases. In onto case, feature set of one database is a subset but in partial mapping case, the two databases do have non-overlapping features too.The arguments for this setup are obtained by running 'CL_partial_mappingargbased_HP_tuning.py' over the possible combinations given in 'bsub_command_*.txt' file based on the respective setting.

3) CL_1-1_vs_sample_size_multiple_runs.py:  Evaluates the performance of all the proposed methods on different synthetic and non-EHR datasets in comparison to Kang and RadialGAN as the number of sample size increases (bijective mapping case). The best setting used for bijective mapping in 'CL_level1_multiple_runs.py' is also used in this setup.

4) HP_running_file.py : This file generates the Sobol sequence based HP grid in various setups.

### MIMIC-III data implementation

1) MIMIC_data_extraction_sql.txt: This is a text file that has the sql commands used to extract the MIMIC-III data from psql database.

2) MIMIC_dataformatting.ipynb: This is a jupyter notebook to format the extracted data based on certain constraints in terms of missingness, datatype etc.

3) MIMIC_CL_Comparison_withDecoder_multiple_partial.py : Evaluates the performance of all the proposed methods on MIMIC-III dataset in comparison to Kang. Here the labs are known mapped features and chartevent features are unmapped features that need to be mapped. Since the two eras in MIMIC-III (Carevue and Metavision) had non-overlapping chartevents, this is a partial mapping setup. The best set of hyperparameters are chosen by running  'MIMIC_CL_withDec_HP_tuning.py' on the settings in 'bsub_command_CL_DatasetMIMIC-withDec.txt'.

4) MIMIC_CL_Comparison_withDecoder_multiple_partial_DS.py : Evaluates the performance of all the proposed methods on the MIMIC-III dataset in comparison to Kang. Here the discharge summary (DS) embeddings are set as the known mapped features and chartevent features are unmapped features that need to te mapped. The best set of hyperparameters are chosen by running 'MIMIC_CL_Comparison_withDecoder_multiple_partial_DS.py' on the settings in 'bsub_command_CL_DatasetMIMIC-withDec_DS.txt'.

5) HP_running_file_MIMIC_CL.py : This file generates the Sobol sequence based HP grid for CL based methods and are saved in the 'bsub_command_CL_DatasetMIMIC-with*.txt'.

6) MIMIC_summary_Statistic_methods.py : This file implements the two univariate parametric (Sum-Stat) and non-parametric (KS-Stat) statistic based methods on MIMIC-III's Carevue and Metavision charts. These methods do not use known mapped features (labs).

7) SMAT/train.py : This is a metadata based implementation adapted from the [SMAT](https://link.springer.com/chapter/10.1007/978-3-030-82472-3_19) paper's  [code](https://github.com/JZCS2018/SMAT/tree/main). The authors had used MIMIC but the setup was little different. For the sake of completeness, we add the MIMIC chartname dataset that we used too in SMAT/datasets subdirectory. To use SMAt in our setup, the dataset name is 'mimic_updated'. Glove embeddings needed for implementation can be downloaded from [here](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip).

### ACTFAST data implementation

1) Different_view_creation_ActFast.py : This file is used to preprocess and create the mapped and unmapped set of features, i.e., preops and labs for the ACTFAST data experiment where the task is to map the labs from Metavision and Epic era EHRs. This also saves any metadata and partitions the data into train and set at the start only.

2) ActFast_Preops_vs_Labs_comp_partial.py : Evaluates the performance of all the proposed methods on the ACTFAST dataset in comparison to Kang method. Here the preoperative faetures are treated as mapped set and the labs are treated as the unmapped set. Clearly, there were non-overlapping features but after the expert provided gold standard maps, Epic labs were a subset of the MV labs and hence this was a onto setup. The best set of hyperparameters are chosen by running 'ActFast_preops_vs_Labs_HP_tuning.py' on the settings in 'bsub_command_CL_DatasetACTFAST_preops_vs_Labs.txt'.

3) HP_running_file_ActFAST_CL.py : This file generates the Sobol sequence based HP grid for CL based methods and are saved in the  bsub file.

4) ACTFAST_univariate_statistic_methods.py : This file implements the two univariate parametric (Sum-Stat) and non-parametric (KS-Stat) statistic based methods on ACTFAST dataset's Metavision and Epic era labs. These methods do not use known mapped features (preops).

### Results

1) 'F1_For_violin_*.txt' : These files have the F1 score for all the partition-trial-mapped feature combinations and are used to obtain the results saved in 'F1_results_savedfile_Aug2023.txt'.

2) 'MatchMSE_For_violin_*.txt' : These files have the MatchMSE values for all the partition-trial-mapped feature combinations and are used to obtain the results saved in 'MatchMSE_results_savedfile_Aug2023.txt'.

3) 'ReconMSE_For_violin_*.txt' : These files have the ReconMSE values for all the partition-trial-mapped feature combinations and are used to obtain the results saved at the end of 'MatchMSE_results_savedfile_Aug2023.txt'.

### Package dependencies

1) The HP tuning experiments were all run inside a docker container using the docker121720/pytorch-for-ts:0.5 image.
2) Python package dependencies: torch, numpy, sklearn, pandas, matplotlib, matching, pingouin, scipy


Old_code_files folder has the codes from the previous version of experiments and only contains results for the KMF and Chimeric methods.
