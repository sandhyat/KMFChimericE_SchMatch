"""

Exploration of real data



"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, linear_model, model_selection, metrics, ensemble
from sklearn.cluster import SpectralCoclustering
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
from sklearn.metrics import consensus_score


def categorical_encoding(data, variable_list):
    # data = data0.copy()
    import itertools
    encoded_variables = list()
    for variable in variable_list:
        one_hot_df = pd.get_dummies(data[variable], dummy_na=False, prefix=variable)
        data = pd.concat([data.drop(columns=variable),
                               one_hot_df], axis=1)
        encoded_variables.append([column for column in one_hot_df.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))
    return data, encoded_variables

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

random.seed(100)
np.random.seed(100)  # fixing the seed for reproducibility


# data details
outcome = 'Mortality_30d'

# reading data
filename_x = '/home/trips/Feature_confusion/data/preopsFull15012021.csv'
filename_y = '/home/trips/Feature_confusion/data/outcomes15012021.csv'
x = pd.read_csv(filename_x)
outcomes_Full = pd.read_csv(filename_y)

# snippet to convert categorical/object type variables to numerical values
some_cat_variables = x.select_dtypes(include=['object']).columns
print(some_cat_variables)
dict_for_categorical_var ={}
for i in some_cat_variables:
    if i != 'caseid':
        temp = np.sort(x[i].astype(str).unique())
        d = {v:j for j,v in enumerate(temp)}
        dict_for_categorical_var[i] = d

print("Dictionary for object to categorical variables \n")
print(dict_for_categorical_var)
x.replace(dict_for_categorical_var,inplace=True)

# replacing the nan in preop los
x.fillna(x.median(), inplace=True)


x.set_index('caseid', inplace=True)
outcomes_Full.set_index('caseid', inplace=True)

#finding the common indices and using only those for both x and the outcome
common_indices = list(set(x.index) & set(outcomes_Full.index))
x = x.loc[common_indices]
outcomes_Full = outcomes_Full.loc[common_indices]

full_Data = pd.concat([x, outcomes_Full[outcome]], axis=1)
full_Data.dropna(how='any', subset=[outcome], inplace=True)
# index_preop = full_Data.loc[full_Data['blank_preop'] == 0].index
index_preop = full_Data.loc[(full_Data['neval_valid'] > 0)][full_Data['blank_preop'] == 0].index
full_Data = full_Data.loc[index_preop]
full_Data.drop(columns=['Location', 'age_missing', 'year', 'case_duration', 'Intubation','blank_preop'], inplace=True)


continuos_features = ['SPL_THEMES', 'RPL_THEMES', 'LVEF', 'Neck', 'PreOp_Diastolic', 'PreOp_Systolic',
                            'PreOp.SpO2', 'PreOp.HR', 'Age',
                            'HEIGHT', 'WEIGHT', 'BMI', 'Ideal_Body_Weight', 'Albumin', 'ALT', 'Alkaline_Phosphate',
                            'Creatinine', 'Glucose', 'Hematocrit', 'Partial_Thromboplastin_Time', 'Potassium', 'Sodium',
                            'Urea_Nitrogen', 'White_Blood_Cells', 'preop_los',"neval", "neval_valid"]
features_frm_freetxt = ['txwv'+str(i) for i in range(1,51)]
features_frm_diagncode = ['dxwv'+str(i) for i in range(1,51)]

categorical_features = ["Anesthesia_Type", "VALVULAR_DISEASE",
                                   "CPAP.Usage", "ASA", "PAP_Type", "Surg_Type", "FUNCTIONAL_CAPACITY", "SEX",
                                   "RACE"]

bin_features = [c for c in full_Data.columns if c not in [
    outcome] + continuos_features + categorical_features + features_frm_diagncode + features_frm_freetxt]

# getting the number of unique values for the categorical variables
uniqu_ele_cat_values = np.zeros(len(categorical_features))
for i in range(len(categorical_features)):
    uniqu_ele_cat_values[i] = len(full_Data[categorical_features[i]].unique())

print(uniqu_ele_cat_values)
# one hot encoding the categorical variables
full_data_ohe, encoded_var = categorical_encoding(full_Data, categorical_features)

full_data_ohe['Dialysis_History'] = (full_data_ohe['Dialysis_History'] == 2).astype(
    int)  # this is the only variable that has 1 2 as binary

# reordering to making sure that mapped features are at the starting of the vector
feature_names = encoded_var + bin_features
full_data_ohe = full_data_ohe.reindex(columns=feature_names + [outcome])

print(full_data_ohe.columns)
print(len(list(full_data_ohe.columns)))

Feature_matrix = full_data_ohe.iloc[:,:-1]

Cor_from_df = Feature_matrix.corr()
Cor = np.corrcoef(Feature_matrix.values.T)
print(Cor_from_df)
# variation in the correlation matrix
sns_plot = sns.heatmap(Cor_from_df, cmap="YlGnBu", annot=False, xticklabels=True, yticklabels=True)
sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=3)
sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=3)
fig = sns_plot.get_figure()

fig.savefig("Original_correlation_with_feature_names.pdf")
# plt.matshow(Cor, cmap = plt.cm.Blues)
# plt.savefig("Original_correlation.png")
plt.close()


var_X = np.var(Cor)
print(var_X)
#
# u, s, v = np.linalg.svd(Cor)

eig_values = np.linalg.eig(Cor)

print(np.round(eig_values[0], decimals=4))
# exit()




# sns_plot = sns.heatmap(Cor, cmap="YlGnBu", annot=False, xticklabels=True, yticklabels=True)
# sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=3)
# sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=3)
# fig = sns_plot.get_figure()
# fig.savefig("Original_correlation.png")
# # plt.matshow(Cor, cmap = plt.cm.Blues)
# # plt.savefig("Original_correlation.png")
# plt.close()

# n_cluster_values= [5,10, 15, 20]
n_cluster_values= [10]  # finally chose 10

for i in n_cluster_values:
    # n_clusters_chosen = 5
    model =SpectralCoclustering(n_clusters=i, random_state=0)
    model.fit(Cor_from_df)
    # score = consensus_score(model.biclusters_, row)
    fit_data = Cor_from_df.iloc[np.argsort(model.row_labels_)]
    fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

    sns_plot = sns.heatmap(fit_data, cmap="YlGnBu", annot=False, xticklabels=True, yticklabels=True)
    sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=3)
    sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=3)
    fig = sns_plot.get_figure()
    fig.savefig("Real_Data_Block_correlation_" + str(i) + "_n_clusters_with_names.pdf", bbox='tight')
    plt.close()

    del fit_data





