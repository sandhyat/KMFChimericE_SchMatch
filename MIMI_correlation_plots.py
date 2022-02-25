"""
Correlation for MIMIC

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



# Reading data files
CV_full = pd.read_csv('/home/trips/MIMIC_feature_confusion/Final_MIMIC_lab_chart_CV.csv')
MV_full = pd.read_csv('/home/trips/MIMIC_feature_confusion/Final_MIMIC_lab_chart_MV.csv')
# Getting list of all items along with the source and label
item_id_dbsource = pd.read_csv('/home/trips/d_items_chartevents.csv')
itemid_labs = pd.read_csv('/home/trips/d_items_labevents.csv')

# reseting the index to hadm_id for both
CV_full.set_index('hadm_id', inplace=True)
MV_full.set_index('hadm_id', inplace=True)

CV_full.drop(columns='742', inplace=True) # Dropping because this variable only takes 1 as the value.

# getting the itemids to be used later
list_lab_ids = list(CV_full.columns)[:58]
onlychart_CV = list(CV_full.columns)[58:]
onlychart_MV = list(MV_full.columns)[58:]
onlychart_binary_MV = [i for i in onlychart_MV if len(MV_full[i].value_counts()) < 3] # since metavision has all binaries
onlychart_cont_MV = [i for i in onlychart_MV if i not in onlychart_binary_MV]
onlychart_cat_CV = [ i for i in onlychart_CV if CV_full[i].dtypes == 'object']
onlychart_cont_CV = [i for i in onlychart_CV if CV_full[i].dtypes == 'float64']

# creating label dictionary for later
itemid_label_lab = dict(zip(list(itemid_labs.itemid), list(itemid_labs.label)))
itemid_label_chart = dict(zip(list(item_id_dbsource.itemid), list(item_id_dbsource.label)))
itemid_label_dict = {**itemid_label_chart, **itemid_label_lab}  # merging two dictionaries here


""" preparing true match matrix """

# Itemids with match in order of matching (ultiple case as of now; will be removed in a while)
CV_itemids_with_match_t = [211, 8549, 5815, 51, 8368, 52, 5813, 8547, 113, 455, 8441, 456, 618, 779, 834, 814, 778,
                       646, 506, 813, 861, 1127, 1542, 770, 788, 1523, 791, 1525, 811, 821, 1532, 769, 1536,
                       8551, 5817, 678, 8554, 5820, 780, 1126, 470, 190, 50, 8553, 5819, 763, 683, 682, 684,
                       450, 619, 614, 615, 535, 543, 444, 578, 776, 773, 1162, 781, 786, 1522, 784, 796, 797,
                       798, 799, 800, 807, 816, 818, 1531, 827, 1534, 848, 1538, 777, 762, 837, 1529, 920, 1535,
                       785, 772, 828, 829, 1286, 824, 1533, 825, 1530, 815, 6206, 6207]
MV_itemids_with_match_t = [220045, 220046, 220047, 220050, 220051, 220052, 220056, 220058, 220074, 220179, 220180,
                       220181, 220210, 220224, 220227, 220228, 220235, 220277, 220339, 220545, 220546, 220546,
                       220546, 220587, 220602, 220602, 220615, 220615, 220621, 220635, 220635, 220644, 220645,
                       223751, 223752, 223761, 223769, 223770, 223830, 223830, 223834, 223835, 223876, 224161,
                       224162, 224639, 224684, 224685, 224686, 224687, 224688, 224689, 224690, 224695, 224696,
                       224697, 224701, 224828, 225612, 225624, 225624, 225625, 225625, 225634, 225639, 225640,
                       225641, 225642, 225643, 225664, 225667, 225668, 225668, 225677, 225677, 225690, 225690,
                       225698, 226512, 226534, 226537, 226707, 227442, 227445, 227456, 227457, 227464, 227465,
                       227465, 227466, 227466, 227467, 227467, 227565, 227566]

# converting the above integers into strings
CV_itemids_with_match_t = [str(i) for i in CV_itemids_with_match_t]
MV_itemids_with_match_t = [str(i) for i in MV_itemids_with_match_t]

match_df = pd.DataFrame(columns=['CV_itemids', 'CV_labels', 'MV_itemids', 'MV_labels'])
match_df['CV_itemids'] = CV_itemids_with_match_t
match_df['MV_itemids'] = MV_itemids_with_match_t
for i in range(len(match_df)):
    match_df.loc[i, "CV_labels"] = itemid_label_dict[int(match_df.loc[i, 'CV_itemids'])]
    match_df.loc[i, "MV_labels"] = itemid_label_dict[int(match_df.loc[i, 'MV_itemids'])]

# removing the rows that are beyond one to one matching
match_df.drop_duplicates(subset=['MV_itemids'], inplace=True)

CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

CV_itemids_to_drop = [i for i in CV_itemids_with_match_t if i not in CV_itemids_with_match]
# run the follwing routine once only since it is inplace
for i in CV_itemids_to_drop:
    onlychart_CV.remove(str(i))
    onlychart_cont_CV.remove(str(i))

""" Correlation computation """  # this is done because there are some repetititve lab features in chartevents
Cor_CV = CV_full[list_lab_ids + onlychart_cont_CV].corr()
Cor_MV = MV_full[list_lab_ids + onlychart_cont_MV].corr()

# computing the max in each row
high_cor_lab_chart_pairsCV = {}
for i in list_lab_ids:
    if Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1)[0] > 0.97:
        high_cor_lab_chart_pairsCV[i] = Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1).index[0]
    # print(itemid_label_dict_labs[int(i)], itemid_label_dict[int(Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1).index[0])], Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1)[0])

high_cor_lab_chart_pairsMV = {}
for i in list_lab_ids:
    if Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1)[0] > 0.97:
        high_cor_lab_chart_pairsMV[i] = Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1).index[0]
    # print(itemid_label_dict_labs[int(i)], itemid_label_dict[int(Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1).index[0])], Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1)[0])

match_df.drop(match_df[match_df['CV_itemids'].isin(high_cor_lab_chart_pairsCV.values())].index, inplace=True)
match_df.drop(match_df[match_df['MV_itemids'].isin(high_cor_lab_chart_pairsMV.values())].index, inplace=True)

CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

for i in high_cor_lab_chart_pairsCV.values():
    onlychart_CV.remove(str(i))
    onlychart_cont_CV.remove(str(i))

for i in high_cor_lab_chart_pairsMV.values():
    onlychart_MV.remove(str(i))
    onlychart_cont_MV.remove(str(i))

# final matching dict
match_dic = dict(zip(CV_itemids_with_match, MV_itemids_with_match))

# itemids with no match
CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]
MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]

print(" CV_itemids_with match ", len(CV_itemids_with_match))
print(" MV_itemids_with match ", len(MV_itemids_with_match))

print(" CV_itemids_with NO match ", len(CV_itemids_withnomatch))
print(" MV_itemids_with NO match ", len(MV_itemids_withnomatch))


CV_data = CV_full[list_lab_ids+onlychart_cont_CV]
MV_data = MV_full[list_lab_ids+onlychart_cont_MV]

Cor_from_df_CV = CV_data.corr()
Cor_from_df_MV = MV_data.corr()

"""  Getting the highest correlation lab chart pair """

print(" **********************************************")
high_cor_lab_chart_pairsCV_temp = {}
cor_lab_chart_CV = []
ctr=0
for i in list_lab_ids:
    high_cor_lab_chart_pairsCV_temp[i] = itemid_label_dict[
        int(Cor_from_df_CV[i][Cor_from_df_CV[i].index.isin(onlychart_cont_CV)].nlargest(1).index[0])]
    cor_lab_chart_CV.append(Cor_from_df_CV[i][Cor_from_df_CV[i].index.isin(onlychart_cont_CV)].nlargest(1)[0])
    print(itemid_label_dict[int(i)], itemid_label_dict[
        int(Cor_from_df_CV[i][Cor_from_df_CV[i].index.isin(onlychart_cont_CV)].nlargest(1).index[0])],
          Cor_from_df_CV[i][Cor_from_df_CV[i].index.isin(onlychart_cont_CV)].nlargest(1)[0], ctr)
    ctr = ctr+1

Highest_lab_chart_cor_arg_CV = np.argmax(np.array(cor_lab_chart_CV))
print(" index value", Highest_lab_chart_cor_arg_CV)
print(" *** Largest correlation lab chart pair CV ",
      itemid_label_dict[int(list(high_cor_lab_chart_pairsCV_temp.keys())[Highest_lab_chart_cor_arg_CV])],
      list(high_cor_lab_chart_pairsCV_temp.values())[Highest_lab_chart_cor_arg_CV], " with correlation value of ",
      cor_lab_chart_CV[Highest_lab_chart_cor_arg_CV])


high_cor_lab_chart_pairsMV_temp = {}
cor_lab_chart_MV = []
ctr=0
for i in list_lab_ids:
    high_cor_lab_chart_pairsMV_temp[i] = itemid_label_dict[
        int(Cor_from_df_MV[i][Cor_from_df_MV[i].index.isin(onlychart_cont_MV)].nlargest(1).index[0])]
    cor_lab_chart_MV.append(Cor_from_df_MV[i][Cor_from_df_MV[i].index.isin(onlychart_cont_MV)].nlargest(1)[0])
    print(itemid_label_dict[int(i)], itemid_label_dict[
        int(Cor_from_df_MV[i][Cor_from_df_MV[i].index.isin(onlychart_cont_MV)].nlargest(1).index[0])],
          Cor_from_df_MV[i][Cor_from_df_MV[i].index.isin(onlychart_cont_MV)].nlargest(1)[0], ctr)
    ctr = ctr+1

Highest_lab_chart_cor_arg_MV = np.argmax(np.array(cor_lab_chart_MV))
print(" index value", Highest_lab_chart_cor_arg_MV)
print(" *** Largest correlation lab chart pair MV ",
      itemid_label_dict[int(list(high_cor_lab_chart_pairsMV_temp.keys())[Highest_lab_chart_cor_arg_MV])],
      list(high_cor_lab_chart_pairsMV_temp.values())[Highest_lab_chart_cor_arg_MV], " with correlation value of ",
      cor_lab_chart_MV[Highest_lab_chart_cor_arg_MV])

print(" **********************************************")


simple_CV_cor_plot = sns.heatmap(Cor_from_df_CV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
simple_CV_cor_plot.set_title("MIMIC-III Carevue (CV) era feature correlations")
simple_CV_cor_plot.hlines([58],*simple_CV_cor_plot.get_xlim(), colors='black')
simple_CV_cor_plot.vlines([58],*simple_CV_cor_plot.get_ylim(), colors='black')
fig = simple_CV_cor_plot.get_figure()
fig.savefig("CV_MIMIC_Block_correlation.pdf", bbox='tight')
fig.savefig("CV_MIMIC_Block_correlation.png", bbox='tight')
plt.close()

simple_MV_cor_plot = sns.heatmap(Cor_from_df_MV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
simple_MV_cor_plot.set_title("MIMIC-III Metavision (MV) era feature correlations")
simple_MV_cor_plot.hlines([58],*simple_MV_cor_plot.get_xlim(), colors='black')
simple_MV_cor_plot.vlines([58],*simple_MV_cor_plot.get_ylim(), colors='black')
fig = simple_MV_cor_plot.get_figure()
fig.savefig("MV_MIMIC_Block_correlation.pdf", bbox='tight')
fig.savefig("MV_MIMIC_Block_correlation.png", bbox='tight')
plt.close()

n_cluster_values= [5]

for i in n_cluster_values:
    # n_clusters_chosen = 5
    model =SpectralCoclustering(n_clusters=i, random_state=0)
    model.fit(Cor_from_df_CV)
    # score = consensus_score(model.biclusters_, row)
    fit_data = Cor_from_df_CV.iloc[np.argsort(model.row_labels_)]
    fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

    sns_plot = sns.heatmap(fit_data, cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    sns_plot.set_title("MIMIC-III Carevue (CV) era feature correlations")
    fig = sns_plot.get_figure()
    fig.savefig("CV_MIMIC_Block_correlation_" + str(i) + "_n_clusters.pdf", bbox='tight')
    fig.savefig("CV_MIMIC_Block_correlation_" + str(i) + "_n_clusters.png", bbox='tight')
    plt.close()

    del fit_data

for i in n_cluster_values:
    # n_clusters_chosen = 5
    model =SpectralCoclustering(n_clusters=i, random_state=0)
    model.fit(Cor_from_df_MV)
    # score = consensus_score(model.biclusters_, row)
    fit_data = Cor_from_df_MV.iloc[np.argsort(model.row_labels_)]
    fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

    sns_plot = sns.heatmap(fit_data, cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(), fontsize=3)
    sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), fontsize=3)
    sns_plot.set_title("MIMIC-III Metavisioin (MV) era feature correlations")
    fig = sns_plot.get_figure()
    fig.savefig("MV_MIMIC_Block_correlation_" + str(i) + "_n_clusters.pdf", bbox='tight')
    fig.savefig("MV_MIMIC_Block_correlation_" + str(i) + "_n_clusters.png", bbox='tight')
    plt.close()

    del fit_data
