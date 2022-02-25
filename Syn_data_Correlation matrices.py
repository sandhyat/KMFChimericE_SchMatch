"""
Correlation and exploration of the synthetic datasets

"""

# importing packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, linear_model, model_selection, metrics, ensemble
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from scipy import linalg, stats
import pickle
from datetime import datetime
import os.path
import math
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from collections import Counter
import random
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from matching.games import StableMarriage
import seaborn as sns
from sklearn.cluster import SpectralCoclustering


# data details
outcome = "Y"
dataset_number = 6

# reading the data
# filename = "./Syn_data/2021-04-16Syn_Data_" + str(dataset_number) + "_size_20_10000_for_AE_balanced.csv"  # for dataset 1 and 2
# filename = "./Syn_data/SD_" + str(dataset_number) + "/2021-05-20Syn_Data_" + str(dataset_number) +"_size_20_10000_for_AE_balanced.csv"
filename = "./Syn_data/2021-05-24Syn_Data_" + str(dataset_number) + "_size_30_10000_for_AE_balanced.csv"  # for dataset 6

full_Data = pd.read_csv(filename)  # for dataset 4
num_sample = full_Data.shape[0]
num_features = full_Data.shape[1] - 1

saving_dir = './Syn_' + str(dataset_number)+"_exploration"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


# collecting only the features
Feature_matrix = full_Data.iloc[:,:-1]


Cor_from_df = Feature_matrix.corr()
Cor = np.corrcoef(Feature_matrix.values.T)
print(Cor_from_df)
# variation in the correlation matrix
sns_plot = sns.heatmap(Cor_from_df, cmap="YlGnBu", annot=False, xticklabels=True, yticklabels=True)
sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=8)
sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=8)
fig = sns_plot.get_figure()
fig.savefig(saving_dir + "/Syn_Original_correlation_" + str(dataset_number) + "_with_feature_names.pdf")
fig.savefig(saving_dir + "/Syn_Original_correlation_" + str(dataset_number) + "_with_feature_names.png")
plt.close()

# variations
var_X = np.var(Cor)
print(var_X)

eig_values = np.linalg.eig(Cor)
print(np.round(eig_values[0], decimals=4))


n_cluster_values= [2,3,5,10, 15, 20]

for i in n_cluster_values:
    # n_clusters_chosen = 5
    model =SpectralCoclustering(n_clusters=i, random_state=0)
    model.fit(Cor_from_df)
    # score = consensus_score(model.biclusters_, row)
    fit_data = Cor_from_df.iloc[np.argsort(model.row_labels_)]
    fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

    sns_plot = sns.heatmap(fit_data, cmap="YlGnBu", annot=False, xticklabels=True, yticklabels=True)
    sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=8)
    sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=8)
    fig = sns_plot.get_figure()
    fig.savefig(saving_dir + "/Syn_" + str(dataset_number) + "_Block_correlation_" + str(i) + "_n_clusters_with_names.pdf", bbox='tight')
    fig.savefig(saving_dir + "/Syn_" + str(dataset_number) + "_Block_correlation_" + str(i) + "_n_clusters_with_names.png", bbox='tight')
    plt.close()

    del fit_data


