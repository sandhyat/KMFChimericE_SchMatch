"""
This code provides a pipeline to feature mapping problem. It compares the KMF (Sim correlation),  two stage procedure (KMF + Sim correlation), Kang and RadialGAN vs the sample size

The components for both the methods consist of :
1) Training KMF -> Chimeric AE  and  Simple Correlation on exactly same data
2) Using stable marriage algorithm on the correlation matrix from 1 and get the final mappings.
3) Also, for KMF -> Chimeric AE , record the cross correlation values of the final mappings.

INPUT:

Full dataset, the model details, number of permutations, number of partitioning of dataset, fraction of data to be permuted, number of mapped features

OUTPUT:

An array with rows as the number of mapped features and columns as the trial number and cell entry as the avg fraction of mismatches for the a fixed trial and the fixed number of mapped variables


This code has randomness over mapped features and unmapped features too


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
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise,mutual_info_score
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
import pingouin as pg
import datetime


def Stable_matching_algorithm(C_X1_train, C_X2_train, index_O_to_R, index_R_to_O,num_mapped_axis):
    # creating the preference dictionaries
    ####### ----------  X1 train ------------- ##########

    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}

    for i in range(C_X1_train.shape[0]):
        sorted_index = np.argsort(-C_X1_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X1_train.shape[1]):
        sorted_index = np.argsort(-C_X1_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X1_train)
    # print(cross_recon_features_pref_X1_train)

    game_X1_train = StableMarriage.create_from_dictionaries(true_features_pref_X1_train,
                                                            cross_recon_features_pref_X1_train)

    ####### ----------  X2 train ------------- ##########

    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}

    for i in range(C_X2_train.shape[0]):
        sorted_index = np.argsort(-C_X2_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):
        sorted_index = np.argsort(-C_X2_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X2_train)
    # print(cross_recon_features_pref_X2_train)

    game_X2_train = StableMarriage.create_from_dictionaries(true_features_pref_X2_train,
                                                            cross_recon_features_pref_X2_train)


    ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)


    # for comparison to the the initial index that were passed
    x1_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x2_train.values()])

    # getting the number of mismatches
    mismatched_x1_train = [i for i, j in zip(index_O_to_R, x1_train_y) if i != j]
    mismatched_x2_train = [i for i, j in zip(index_R_to_O, x2_train_y) if i != j]

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(C_X2_train.shape)

    for i in range(matching_x1_train_matrix.shape[0]):
        # print(i, x1_train_y[i]-1)
        matching_x1_train_matrix[i,x1_train_y[i]-1-num_mapped_axis]=1


    for i in range(matching_x2_train_matrix.shape[0]):
        # print(i, x2_train_y[i]-1)
        matching_x2_train_matrix[i,x2_train_y[i]-1-num_mapped_axis]=1

    print("Mistakes x1")
    print(mismatched_x1_train)
    print(" Mistakes x2 train")
    print(mismatched_x2_train)

    return mismatched_x1_train, mismatched_x2_train, matching_x1_train_matrix, matching_x2_train_matrix

def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()

    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std

    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v

    return data

# function to give initial random weights
# function to give initial random weights
def weights_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size, device="cuda") * xavier_stddev, requires_grad=True)

class TabularDataset(Dataset):
    def __init__(self, data, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
          The data frame object for the input data. It must
          contain all the continuous, categorical and the
          output columns to be used.

        cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding
          layers in the model. These columns must be
          label encoded beforehand.

        output_col: string
          The name of the output variable column in the data
          provided.
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        # self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in [output_col]]
        # print(self.cont_cols)

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        # if self.cat_cols:
        #     self.cat_X = data[cat_cols].astype(np.int64).values
        # else:
        #     self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx]]

class AE_2_hidden_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.no_of_cont = kwargs["input_shape"]
        self.batchnorm = kwargs['batchnorm']
        self.drop_out_rate = kwargs["drop_out_p"]

        print("input_dimension_total", self.no_of_cont)
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=80)
        self.bn1 = nn.BatchNorm1d(num_features=80)
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer2 = nn.Linear(in_features=80, out_features=40)
        self.bn2 = nn.BatchNorm1d(num_features=40)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=40, out_features=hidden_dim)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)
        self.decoder_hidden_layer1 = nn.Linear(in_features=hidden_dim, out_features=40)
        self.bn4 = nn.BatchNorm1d(num_features=40)
        self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer2 = nn.Linear(in_features=40, out_features=80)
        self.bn5 = nn.BatchNorm1d(num_features=80)
        self.drop_layer4 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_output_layer = nn.Linear(in_features=80, out_features=kwargs["input_shape"])

    def forward(self, cont_data, cross):
        if cross != 1:
            # print("inside the normal loop")
            activation = self.encoder_hidden_layer1(cont_data)
            if self.batchnorm == 1:
                activation = self.bn1(activation)
            activation = self.encoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn2(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer2(activation)
            code0 = self.encoder_output_layer(activation)
            if self.batchnorm == 1:
                code0 = self.bn3(code0)
            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn4(activation)

            activation = F.tanh(activation)
            activation = self.drop_layer3(activation)
            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            reconstructed = self.decoder_output_layer(activation)
        else:
            # print("inside the cross loop")
            code0 = cont_data
            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn4(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer3(activation)
            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            reconstructed = self.decoder_output_layer(activation)

        return code0, reconstructed

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    # print(c_xy)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# MI based method that first creates a graph and then looks for a permutation matrix that minimizes the distance between the adjcency matrices of the two graphs
def Kang_MI_HC_opt_with_Euclidean_dist(df_train_preproc, df_rename_preproc, true_perm,
                    reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1

    num_features = len(reordered_column_names_r) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    if datatype == 'b':
        MI_orig = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
        # MI computation
        for i in range(num_NonCat_features_orig):
            p1_i = sum(df_train_preproc.iloc[:, i]) / df_train_preproc.shape[0]
            p0_i = 1 - p1_i
            MI_orig[i, i] = -p0_i * np.log(p0_i) - p1_i * np.log(p1_i)
            for j in range(i + 1, num_NonCat_features_orig):
                p1_j = sum(df_train_preproc.iloc[:, j]) / df_train_preproc.shape[0]
                p0_j = 1 - p1_j
                p11 = sum(df_train_preproc.iloc[:, i] * df_train_preproc.iloc[:, j]) / df_train_preproc.shape[0]
                p00 = sum(np.where((df_train_preproc.iloc[:, i] == 0) & (df_train_preproc.iloc[:, j] == 0), 1, 0)) / \
                      df_train_preproc.shape[0]
                p01 = sum(np.where((df_train_preproc.iloc[:, i] == 0) & (df_train_preproc.iloc[:, j] == 1), 1, 0)) / \
                      df_train_preproc.shape[0]
                p10 = sum(np.where((df_train_preproc.iloc[:, i] == 1) & (df_train_preproc.iloc[:, j] == 0), 1, 0)) / \
                      df_train_preproc.shape[0]

                MI_orig[i, j] = p00 * np.log(p00 / (p0_i * p0_j)) + p01 * np.log(p01 / (p0_i * p1_j)) + p10 * np.log(
                    p10 / (p1_i * p0_j)) + p11 * np.log(p11 / (p1_i * p1_j))
                MI_orig[j, i] = MI_orig[i, j]

        MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
        # MI computation
        for i in range(num_NonCat_features_r):
            p1_i = sum(df_rename_preproc.iloc[:, i]) / df_rename_preproc.shape[0]
            p0_i = 1 - p1_i
            MI_r[i, i] = -p0_i * np.log(p0_i) - p1_i * np.log(p1_i)
            for j in range(i + 1, num_NonCat_features_r):
                p1_j = sum(df_rename_preproc.iloc[:, j]) / df_rename_preproc.shape[0]
                p0_j = 1 - p1_j
                p11 = sum(df_rename_preproc.iloc[:, i] * df_rename_preproc.iloc[:, j]) / df_rename_preproc.shape[0]
                p00 = sum(np.where((df_rename_preproc.iloc[:, i] == 0) & (df_rename_preproc.iloc[:, j] == 0), 1, 0)) / \
                      df_rename_preproc.shape[0]
                p01 = sum(np.where((df_rename_preproc.iloc[:, i] == 0) & (df_rename_preproc.iloc[:, j] == 1), 1, 0)) / \
                      df_rename_preproc.shape[0]
                p10 = sum(np.where((df_rename_preproc.iloc[:, i] == 1) & (df_rename_preproc.iloc[:, j] == 0), 1, 0)) / \
                      df_rename_preproc.shape[0]

                MI_r[i, j] = p00 * np.log(p00 / (p0_i * p0_j)) + p01 * np.log(p01 / (p0_i * p1_j)) + p10 * np.log(
                    p10 / (p1_i * p0_j)) + p11 * np.log(p11 / (p1_i * p1_j))
                MI_r[j, i] = MI_r[i, j]

    else:
        MI_orig = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
        # MI computation
        for i in range(num_NonCat_features_orig):
            for j in range(i + 1, num_NonCat_features_orig):
                MI_orig[i, j] = calc_MI(df_train_preproc.iloc[:, i], df_train_preproc.iloc[:, j], 20)
                MI_orig[j, i] = MI_orig[i, j]

        MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
        # MI computation
        for i in range(num_NonCat_features_r):
            for j in range(i + 1, num_NonCat_features_r):
                MI_r[i, j] = calc_MI(df_rename_preproc.iloc[:, i], df_rename_preproc.iloc[:, j], 20)
                MI_r[j, i] = MI_r[i, j]

    num_iter = 1000
    # initial random permutation and corresponding distance computation
    initial_perm = np.array(list(np.arange(mpfeatures)) + list(np.random.choice( np.arange(unmapped_features_r), unmapped_features_orig, replace =False)+mpfeatures))

    D_M_normal_ini = 0

    for i in range(len(MI_orig)):
        for j in range(len(MI_orig)):
            # temp = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) / (
            #             MI_orig[i, j] + MI_r[initial_perm[i], initial_perm[j]]))
            temp = (MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) * (MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]])
            if np.isnan(temp) != True :
                D_M_normal_ini = D_M_normal_ini + temp
                # print(temp)
    D_M_normal_ini = np.sqrt(D_M_normal_ini)
    print("Distance_initial_value", D_M_normal_ini)

    D_M_normal = D_M_normal_ini
    for iter in range(num_iter):
        temp_per = initial_perm.copy()
        idx_unmp_feature = np.random.choice(np.arange(unmapped_features_orig),1) + mpfeatures  # random index chosen for swapping the value
        # print(unm_feat)
        # idx_unmp_feature = np.where(temp_per == unm_feat)
        # print(idx_unmp_feature)
        unm_feat_match_random = np.random.choice(np.arange(unmapped_features_r),1)+mpfeatures
        # print(unm_feat_match_random)
        if unm_feat_match_random in temp_per:
            idx_for_already_existing_match = np.where(temp_per==unm_feat_match_random)
            temp_to_swap_value = temp_per[idx_unmp_feature]
            temp_per[idx_unmp_feature] =  unm_feat_match_random
            temp_per[idx_for_already_existing_match] = temp_to_swap_value
            # print(temp_per)
        else:
            temp_per[idx_unmp_feature] =  unm_feat_match_random
            # print(temp_per)

        # checking the cost
        temp_dist_normal = 0
        for i in range(len(MI_orig)):
            for j in range(len(MI_orig)):
                # temp0 = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]]) / (
                #         MI_orig[i, j] + MI_r[temp_per[i], temp_per[j]]))
                temp0 = (MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]])* (MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]])
                if np.isnan(temp0) != True:
                    temp_dist_normal = temp_dist_normal + temp0
                    # print(temp0)
        temp_dist_normal = np.sqrt(temp_dist_normal)
        # updating the cost and the permutation vector
        if temp_dist_normal < D_M_normal:
            # print(" Iteration number where it changed", iter +1)
            # print("initial cost ", D_M_normal)
            # print("initial permutation", initial_perm)

            D_M_normal = temp_dist_normal
            initial_perm = temp_per

            # print(" Updated cost ", D_M_normal)
            # print(" updated permutation ", temp_per)

            print("\n ---------------------- \n")
            print(" Matching after Iteration number ", iter +1)
            print(dict(zip(list(reordered_column_names_orig[mpfeatures:-1]),
                           [reordered_column_names_r[i] for i in initial_perm[mpfeatures:]])))
            print("\n ---------------------- \n")

    mistake_indices = []
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_perm[i]-1==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1
        else:
            mistake_indices.append(true_perm[i])


    print(" \n Mistakes by the KANG method on holdout data")

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in mistake_indices]

    print(" KANG method  X1 mistakes", MisF_X1_te)

    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    print(" -------- KANG  methods  ends ------------- \n \n  ")

    del df_rename_preproc
    # exit()
    return MisF_X1_te

# MI based method that first creates a graph and then looks for a permutation matrix that minimizes the distance between the adjcency matrices of the two graphs
def Kang_MI_HC_opt(df_train_preproc, df_rename_preproc, true_perm
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1

    num_features = len(reordered_column_names_r) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    MI_orig = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
    # MI computation
    for i in range(num_NonCat_features_orig):
        for j in range(i+1, num_NonCat_features_orig):
            MI_orig[i,j] = calc_MI(df_train_preproc.iloc[:,i], df_train_preproc.iloc[:,j], 20)
            MI_orig[j,i] = MI_orig[i,j]

    MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
    # MI computation
    for i in range(num_NonCat_features_r):
        for j in range(i+1, num_NonCat_features_r):
            MI_r[i,j] = calc_MI(df_rename_preproc.iloc[:,i], df_rename_preproc.iloc[:,j], 20)
            MI_r[j,i] = MI_r[i,j]

    num_iter = 5
    # initial random permutation and corresponding distance computation
    initial_perm = np.array(list(np.arange(mpfeatures)) + list(np.random.choice( np.arange(unmapped_features_r), unmapped_features_orig, replace =False)+mpfeatures))

    D_M_normal_ini = 0

    for i in range(len(MI_orig)):
        for j in range(len(MI_orig)):
            temp = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) / (
                        MI_orig[i, j] + MI_r[initial_perm[i], initial_perm[j]]))
            if np.isnan(temp) != True :
                D_M_normal_ini = D_M_normal_ini + temp
                # print(temp)
    print("Distance_initial_value", D_M_normal_ini)

    D_M_normal = D_M_normal_ini
    for iter in range(num_iter):
        temp_per = initial_perm.copy()
        idx_unmp_feature = np.random.choice(np.arange(unmapped_features_orig),1) + mpfeatures  # random index chosen for swapping the value
        unm_feat_match_random = np.random.choice(np.arange(unmapped_features_r),1)+mpfeatures
        # print(unm_feat_match_random)
        if unm_feat_match_random in temp_per:
            idx_for_already_existing_match = np.where(temp_per==unm_feat_match_random)
            temp_to_swap_value = temp_per[idx_unmp_feature]
            temp_per[idx_unmp_feature] =  unm_feat_match_random
            temp_per[idx_for_already_existing_match] = temp_to_swap_value
        else:
            temp_per[idx_unmp_feature] =  unm_feat_match_random

        # checking the cost
        temp_dist_normal = 0
        for i in range(len(MI_orig)):
            for j in range(len(MI_orig)):
                temp0 = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]]) / (
                        MI_orig[i, j] + MI_r[temp_per[i], temp_per[j]]))
                if np.isnan(temp0) != True:
                    temp_dist_normal = temp_dist_normal + temp0
                    # print(temp0)

        # updating the quantity to optimize and the permutation vector
        if temp_dist_normal > D_M_normal:
            D_M_normal = temp_dist_normal
            initial_perm = temp_per


    # true_permutation = list(np.arange(mpfeatures)) +  [np.where(P_x1[a,:]==1)[0] + mpfeatures for a in range(len(P_x1))]
    mistake_indices = []
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_perm[i]-1==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1
        else:
            mistake_indices.append(true_perm[i])


    print(" \n Mistakes by the KANG method on holdout data")

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in mistake_indices]

    print(" KANG method  X1 mistakes", MisF_X1_te)


    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    print(" -------- KANG  methods  ends ------------- \n \n  ")

    del DF_holdout_r

    return MisF_X1_te

def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc,index_for_mapping_orig_to_rename,
                   index_for_mapping_rename_to_orig
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

    device = torch.device('cpu')
    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_unmap_mapped = np.zeros((unmapped_features_orig, mpfeatures))
    CorMatrix_X2_unmap_mapped = np.zeros((unmapped_features_r, mpfeatures))
    CorMatrix_X1_unmap_mapped_P_value = np.zeros((unmapped_features_orig, mpfeatures))
    CorMatrix_X2_unmap_mapped_P_value = np.zeros((unmapped_features_r, mpfeatures))

    for i in range(unmapped_features_orig):
        for j in range(mpfeatures):
            temp = stats.pearsonr(df_train_preproc.values[:, mpfeatures + i], df_train_preproc.values[:, j])
            CorMatrix_X1_unmap_mapped[i, j] = temp[0]
            CorMatrix_X1_unmap_mapped_P_value[i,j] = temp[1]

    for i in range(unmapped_features_r):
        for j in range(mpfeatures):
            temp = stats.pearsonr(df_rename_preproc.values[:, mpfeatures + i], df_rename_preproc.values[:, j])
            CorMatrix_X2_unmap_mapped[i, j] = temp[0]
            CorMatrix_X2_unmap_mapped_P_value[i,j] = temp[1]



    # similarity between the correlation matrices
    sim_cor_norm_X1_to_X2 = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,
                                                           dense_output=True)
    sim_cor_norm_X2_to_X1 = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped, CorMatrix_X1_unmap_mapped,
                                                           dense_output=True)

    """ Calling the stable marriage algorithm for mappings  """

    Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,
                                                               index_for_mapping_orig_to_rename[
                                                               len(mapped_features):],
                                                               index_for_mapping_rename_to_orig[
                                                               len(mapped_features):],
                                                               len(mapped_features))


    test_statistic_num_fromX1 = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                     j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    test_statistic_num_fromX2 = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                     j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                         i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small


    # Bootstrap samples to obtain the standard deviation methods to be later used in p value computation
    num_of_bts = 50
    bts_for_allthe_accepted_matches_fromX1 = np.zeros((unmapped_features_orig,num_of_bts))
    bts_for_allthe_accepted_matches_fromX2 = np.zeros((unmapped_features_orig,num_of_bts))

    for bts in range(num_of_bts):
        Df_holdout_orig_bts = Df_holdout_orig.sample(n=len(Df_holdout_orig), replace=True, random_state=bts, axis=0)
        DF_holdout_r_bts = DF_holdout_r.sample(n=len(DF_holdout_r), replace=True, random_state=bts, axis=0)
        CorMatrix_X1_unmap_mapped_bts = np.zeros((unmapped_features_orig, mpfeatures))
        CorMatrix_X2_unmap_mapped_bts = np.zeros((unmapped_features_r, mpfeatures))

        for i in range(unmapped_features_orig):
            for j in range(mpfeatures):
                temp = stats.pearsonr(Df_holdout_orig_bts.values[:, mpfeatures + i], Df_holdout_orig_bts.values[:, j])
                CorMatrix_X1_unmap_mapped_bts[i, j] = temp[0]

        for i in range(unmapped_features_r):
            for j in range(mpfeatures):
                temp = stats.pearsonr(DF_holdout_r_bts.values[:, mpfeatures + i], DF_holdout_r_bts.values[:, j])
                CorMatrix_X2_unmap_mapped_bts[i, j] = temp[0]

        # similarity between the correlation matrices
        # sim_cor_X1_to_X2_bts = np.matmul(CorMatrix_X1_unmap_mapped_bts, np.transpose(CorMatrix_X2_unmap_mapped_bts))
        # sim_cor_X2_to_X1_bts = np.matmul(CorMatrix_X2_unmap_mapped_bts, np.transpose(CorMatrix_X1_unmap_mapped_bts))

        if np.any(np.isnan(CorMatrix_X1_unmap_mapped_bts))==True or np.any(np.isnan(CorMatrix_X2_unmap_mapped_bts))==True:
            CorMatrix_X1_unmap_mapped_bts = np.nan_to_num(CorMatrix_X1_unmap_mapped_bts)
            CorMatrix_X2_unmap_mapped_bts = np.nan_to_num(CorMatrix_X2_unmap_mapped_bts)
            print("Here here")

        sim_cor_norm_X1_to_X2_bts = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped_bts, CorMatrix_X2_unmap_mapped_bts, dense_output=True)
        sim_cor_norm_X2_to_X1_bts = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped_bts, CorMatrix_X1_unmap_mapped_bts, dense_output=True)
        """ Calling the stable marriage algorithm for mappings  """

        # _ ,_ , x1_match_matrix_test_bts, x2_match_matrix_test_bts = Matching_via_HRM(sim_cor_norm_X1_to_X2_bts, sim_cor_norm_X2_to_X1_bts, P_x1, len(mapped_features))


        # we will use the matched found on the whole dataset and use the bootstraps only to get the dot product estimates
        bts_for_allthe_accepted_matches_fromX1[:,bts] = [sim_cor_norm_X1_to_X2_bts[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                     j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        bts_for_allthe_accepted_matches_fromX2[:,bts] = [sim_cor_norm_X2_to_X1_bts[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                     j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                         i, j] == 1]


    test_statistic_den_fromX1 = [np.std(bts_for_allthe_accepted_matches_fromX1[i,:]) for i in range(x1_match_matrix_test.shape[0])]
    test_statistic_den_fromX2 = [np.std(bts_for_allthe_accepted_matches_fromX2[i, :]) for i in
                                 range(x1_match_matrix_test.shape[0])]


    temp_inf_x1 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])


    # getting the p values that needs to be tested for significance

    test_statistic_for_cor_sig_fromX1 = np.array(test_statistic_num_fromX1)/np.array(test_statistic_den_fromX1)
    test_statistic_for_cor_sig_fromX2 = np.array(test_statistic_num_fromX2)/np.array(test_statistic_den_fromX2)

    temp_inf_x1.corr_p_value = [stats.norm.sf(abs(x))*2 for x in test_statistic_for_cor_sig_fromX1]
    temp_inf_x2.corr_p_value =[stats.norm.sf(abs(x))*2 for x in test_statistic_for_cor_sig_fromX2]

    temp_inf_x1.estimated_cross_corr = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                     j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    temp_inf_x2.estimated_cross_corr = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                     j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                         i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small

    # # correlation values of the accepted matches
    # print(" correlation values of the accepted matches from CC x1 (Holdout_sample)")
    # print(CC_values_for_testing_from_x1_test)
    # print(" correlation values of the accepted matches from CC x2 (Holdout_sample)")
    # print(CC_values_for_testing_from_x2_test)

    # testing whether some of the proposed matches are significant or not;
    # False in the reject list below can be interpreted as the case where the  testing procedure says this match is not significant
    temp_inf_x1.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x1.corr_p_value), method='fdr_by', alpha=0.05)
    temp_inf_x2.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x2.corr_p_value), method='fdr_by', alpha=0.05)

    # print("reject from x1 (Holdout_sample)")
    # print(reject_x1_test)
    # print("reject from x2 (Holdout_sample)")
    # print(reject_x2_test)

    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]


    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]


    # dropping non significant matched feature pairs
    num_insign_dropp_x1 = 0
    num_insign_dropp_x2 = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x1 ", temp_inf_x1.ump_feature_in_X1[i], temp_inf_x1.match_byGS[i], temp_inf_x1.estimated_cross_corr[i])
            temp_inf_x1.drop([i], inplace=True)
            num_insign_dropp_x1 = num_insign_dropp_x1 +1

        if temp_inf_x2.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x2 ", temp_inf_x2.ump_feature_in_X1[i], temp_inf_x2.match_byGS[i], temp_inf_x2.estimated_cross_corr[i])
            temp_inf_x2.drop([i], inplace=True)
            num_insign_dropp_x2 = num_insign_dropp_x2 + 1

    print(" Number of insignificant feature pair drops from x1 ", num_insign_dropp_x1)
    print(" Number of insignificant feature pair drops from x2 ", num_insign_dropp_x2)

    # ordering the
    temp_inf_x1 = temp_inf_x1.sort_values(by='estimated_cross_corr', ascending=False)
    temp_inf_x2 = temp_inf_x2.sort_values(by='estimated_cross_corr', ascending=False)

    num_additional_mapped_for_next_stage_x1 = int(len(temp_inf_x1)/2)
    num_additional_mapped_for_next_stage_x2 = int(len(temp_inf_x2)/2)

    # taking the intersection of the additional mapped features
    temp_x1_x1 = [temp_inf_x1.ump_feature_in_X1[i] for i in list(temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x1_match = [temp_inf_x1.match_byGS[i] for i in list(temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x2_x1 = [temp_inf_x2.ump_feature_in_X1[i] for i in list(temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]
    temp_x2_match = [temp_inf_x2.match_byGS[i] for i in list(temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]



    final_additional_mapped = list(set(temp_x1_x1).intersection(temp_x2_x1))
    final_additional_mapped_corr_match =[]
    for i in final_additional_mapped:
        final_additional_mapped_corr_match.append(temp_x1_match[temp_x1_x1.index(i)])
    # idx_final = temp_inf_x1.index(final_additional_mapped)
    # final_additional_mapped_corr_match = temp_inf_x2[idx_final]

    print(" \n Mistakes by the simple correlation method on holdout data")

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

    print(" KMF  X1 mistakes", MisF_X1_te)
    print(" KMF  X2 mistakes", MisF_X2_te)

    print(" Sim_Correlation  X1_train mistakes number", len(Mistakes_X1_te), "out of ", unmapped_features_orig)
    print(" Sim_Correlation  X2_train mistakes number", len(Mistakes_X2_te), "out of ", unmapped_features_r)


    print(" -------- Sim_Correlation  methods  ends ------------- \n \n  ")

    del df_rename_preproc

    return Mistakes_X1_te, Mistakes_X2_te, temp_inf_x1, temp_inf_x2, final_additional_mapped, final_additional_mapped_corr_match

def Train_cross_AE(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features,mapped_features_updated_orig, mapped_features_updated_r, Cor_from_df,  Df_holdout_orig, DF_holdout_r,filename_for_saving_tran_quality,normalizing_values_orig,normalizing_values_r):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_r) - 1


    num_features = len(reordered_column_names_r) -1
    num_NonCat_features_r = len(reordered_column_names_r) - 1


    print(" -------- Chimeric AE training starts with perm -------------  ")

    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batch_size, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batch_size, shuffle=True, num_workers=1)

    if num_of_hidden_layers == 2:
        model_orig = AE_2_hidden_layer(input_shape=num_NonCat_features_orig, batchnorm=batchnorm, drop_out_p=dropout_rate).to(
            device)
        model_r = AE_2_hidden_layer(input_shape=num_NonCat_features_r, batchnorm=batchnorm, drop_out_p=dropout_rate).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_r = optim.Adam(model_r.parameters(), lr=learning_rate, weight_decay=1e-5)

    # lr scheduler
    scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, patience=2, verbose=True)
    scheduler_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_orig, patience=2, verbose=True)

    # initializing the loss function
    criterion = nn.MSELoss()

    total_loss = []
    ae_orig_error_list = []
    ae_r_error_list = []
    ae_orig_on_r_cross_list = []
    ae_r_on_orig_cross_list = []
    ortho_for_epoch_orig_list = []
    ortho_for_epoch_r_list = []
    cycle_for_epoch_orig_list = []
    cycle_for_epoch_r_list = []
    for epoch in range(epochs):
        loss = 0
        ae_orig_error = 0
        ae_r_error = 0
        ae_orig_on_r_cross = 0
        ae_r_on_orig_cross = 0

        ortho_for_epoch_orig = 0
        ortho_for_epoch_r = 0
        cycle_for_epoch_orig = 0
        cycle_for_epoch_r = 0

        for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
            if len(data[0][1]) == len(data[1][1]):
                x_o = data[0][1].to(device)
                x_r = data[1][1].to(device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer_orig.zero_grad()
                optimizer_r.zero_grad()

                # compute reconstructions
                L_orig, outputs_orig = model_orig(x_o, 0)
                L_r, outputs_r = model_r(x_r, 0)

                # compute cross reconstruction
                _, output_cross_r = model_orig(L_r, 1)
                _, output_cross_orig = model_r(L_orig, 1)

                # compute training reconstruction loss
                train_loss_orig = criterion(outputs_orig, x_o)
                train_loss_r = criterion(outputs_r, x_r)

                # compute loss on cross reconstruction
                train_loss_cross_r = criterion(output_cross_r[:, :len(mapped_features_updated_orig)],
                                               x_r[:, :len(mapped_features_updated_orig)])
                train_loss_cross_orig = criterion(output_cross_orig[:, :len(mapped_features_updated_orig)],
                                                  x_o[:, :len(mapped_features_updated_orig)])

                # compute cycle consistency loss
                L_after_first_cross_r, _ = model_r(output_cross_orig,0)
                _, double_cross_orig = model_orig(L_after_first_cross_r,1)

                L_after_first_cross_orig, _ = model_orig(output_cross_r,0)
                _, double_cross_r = model_r(L_after_first_cross_orig,1)

                train_loss_cycle_orig = criterion(double_cross_orig,x_o)
                train_loss_cycle_r = criterion(double_cross_r,x_r)

                # compute orthogonality loss
                if orthogonalization_type == 1:
                    # print(L_orig.get_device())
                    # print(L_r.get_device())
                    # print(torch.eye(L_orig.shape[1]).to(device).get_device())
                    # print(torch.eye(L_r.shape[1]).get_device())
                    orth_weighting = 0.01
                    orthog_loss_orig = torch.norm(
                        torch.matmul(torch.transpose(L_orig, 0, 1), L_orig) - torch.eye(L_orig.shape[1]).to(device))
                    orthog_loss_r = torch.norm(
                        torch.matmul(torch.transpose(L_r, 0, 1), L_r) - torch.eye(L_r.shape[1]).to(device))

                elif orthogonalization_type == 3:  # this snippet tries to match the joint distribution between the two latent spaces
                    orth_weighting = 1

                    # minimizing the norm of the difference between cov of two latent spaces,  div by 2 to be consistent with the notations
                    orthog_loss_orig = torch.norm(
                        torch.matmul(torch.transpose(L_orig, 0, 1), L_orig) - torch.matmul(
                            torch.transpose(L_r, 0, 1), L_r)) / 2
                    orthog_loss_r = torch.norm(
                        torch.matmul(torch.transpose(L_orig, 0, 1), L_orig) - torch.matmul(
                            torch.transpose(L_r, 0, 1), L_r)) / 2

                else:
                    orth_weighting = 0
                    orthog_loss_orig = torch.zeros(1)
                    orthog_loss_r = torch.zeros(1)

                train_loss = weight_direct * ((1 - frac_renamed) * train_loss_orig + frac_renamed * train_loss_r) + orth_weighting * (
                                     orthog_loss_orig + orthog_loss_r) + weight_cross * (
                                     train_loss_cross_orig + train_loss_cross_r) +weight_cycle * (train_loss_cycle_orig+train_loss_cycle_orig)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer_orig.step()
                optimizer_r.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

                ae_orig_error += train_loss_orig.item()
                ae_r_error += train_loss_r.item()
                ae_orig_on_r_cross += train_loss_cross_r.item()
                ae_r_on_orig_cross += train_loss_cross_orig.item()
                ortho_for_epoch_orig += orthog_loss_orig.item()
                ortho_for_epoch_r += orthog_loss_r.item()
                cycle_for_epoch_orig += train_loss_cycle_orig.item()
                cycle_for_epoch_r += train_loss_cycle_r.item()

                # print("Different individual (not cumulative) losses in successive iterations")
                # print("direct ae orig ", train_loss_orig.item())
                # print("direct ae r", train_loss_r.item())
                # print("cross r to orig",train_loss_cross_r.item() )
                # print("cross orig to r", train_loss_cross_orig.item())
                # print("cycle for ae orig ", train_loss_cycle_orig.item())
                # print(" cycle for ae r", train_loss_cycle_r.item())
                # print("------------------------------")

        # compute the epoch training loss
        loss = loss / (len(train_loader_orig) + len(train_loader_r))
        ae_orig_error = ae_orig_error / len(train_loader_orig)
        ae_r_error = ae_r_error / len(train_loader_r)
        ae_orig_on_r_cross = ae_orig_on_r_cross / len(train_loader_r)
        ae_r_on_orig_cross = ae_r_on_orig_cross / len(train_loader_orig)
        ortho_for_epoch_orig = ortho_for_epoch_orig / len(train_loader_orig)
        ortho_for_epoch_r = ortho_for_epoch_r / len(train_loader_r)
        cycle_for_epoch_orig = cycle_for_epoch_orig / len(train_loader_orig)
        cycle_for_epoch_r = cycle_for_epoch_r / len(train_loader_r)

        scheduler_r.step(ae_r_error)
        scheduler_orig.step(ae_orig_error)

        # display the epoch training loss
        # print("epoch : {}/{}, total loss = {:.8f}".format(epoch + 1, epochs, loss))
        # print("epoch : {}/{}, recon loss ae orig= {:.8f}".format(epoch + 1, epochs, ae_orig_error))
        # print("epoch : {}/{}, recon loss ae r= {:.8f}".format(epoch + 1, epochs, ae_r_error))
        # print("epoch : {}/{}, cross recon loss  on ae orig when data is renamed = {:.8f}".format(epoch + 1,
        #                                                                                          epochs,
        #                                                                                          ae_orig_on_r_cross))
        # print("epoch : {}/{}, cross recon loss on ae r when data is orig = {:.8f}".format(epoch + 1, epochs,
        #                                                                                   ae_r_on_orig_cross))
        # print("epoch : {}/{}, cycle loss ae orig= {:.8f}".format(epoch + 1, epochs, cycle_for_epoch_orig))
        # print("epoch : {}/{}, cycle loss ae r= {:.8f}".format(epoch + 1, epochs, cycle_for_epoch_r))
        # print("epoch : {}/{}, ortho loss ae orig= {:.8f}".format(epoch + 1, epochs, ortho_for_epoch_orig))
        # print("epoch : {}/{}, ortho loss ae r= {:.8f}".format(epoch + 1, epochs, ortho_for_epoch_r))

        total_loss.append(loss)
        ae_orig_error_list.append(ae_orig_error)
        ae_r_error_list.append(ae_r_error)
        ae_orig_on_r_cross_list.append(ae_orig_on_r_cross)
        ae_r_on_orig_cross_list.append(ae_r_on_orig_cross)
        cycle_for_epoch_orig_list.append(cycle_for_epoch_orig)
        cycle_for_epoch_r_list.append(cycle_for_epoch_r)
        ortho_for_epoch_orig_list.append(ortho_for_epoch_orig)
        ortho_for_epoch_r_list.append(ortho_for_epoch_r)

    print("Combined AE loss is ", loss)

    """ AE part preprocessing  ends   """

    # switching to eval mode so that drop out is off when evaluating
    model_orig.eval()
    model_r.eval()

    # comparing actual reconstruction and cross recontruction on original data
    # latent_code_Orig_fullTrain_orig, recons_orig_train_from_orig = model_orig(
    #     torch.Tensor(df_train_preproc.iloc[:, :-1].values), 0)
    # _, recons_orig_train_frommodelR = model_r(latent_code_Orig_fullTrain_orig, 1)
    #
    # # comparing actual reconstruction and cross recontruction on renamed data
    # latent_code_renamed, recons_rename_train_frommodelR = model_r(
    #     torch.Tensor(df_rename_preproc.iloc[:, :-1].values), 0)
    # _, recons_rename_train_frommodelOrig = model_orig(latent_code_renamed, 1)
    #
    #
    # features_reconst_from_crossR = recons_orig_train_frommodelR.detach().numpy()
    # features_true_orig = df_train_preproc.iloc[:, :-1].values
    #
    # features_reconst_from_crossO = recons_rename_train_frommodelOrig.detach().numpy()
    # features_true_renamed = df_rename_preproc.iloc[:, :-1].values
    #
    #
    # # computing the correlation matrix between original feature values and cross reconstruction
    # CorMatrix_X1_X1_hat_cross = np.zeros((num_features, num_features))
    # CorMatrix_X2_X2_hat_cross = np.zeros((num_features, num_features))
    #
    # for i in range(num_features):
    #     for j in range(num_features):
    #         CorMatrix_X1_X1_hat_cross[i, j] = \
    #             stats.pearsonr(features_true_orig[:, i], features_reconst_from_crossR[:, j])[0]
    #         CorMatrix_X2_X2_hat_cross[i, j] = \
    #             stats.pearsonr(features_true_renamed[:, i], features_reconst_from_crossO[:, j])[0]
    #
    # # selecting the correlation only for unmapped variables
    # short_CorMatrix_X1_X1_hat_cross = CorMatrix_X1_X1_hat_cross[len(mapped_features):,
    #                                   len(mapped_features):]
    # short_CorMatrix_X2_X2_hat_cross = CorMatrix_X2_X2_hat_cross[len(mapped_features):,
    #                                   len(mapped_features):]


    print(" \n **********************************************************************")
    print(" -------------------  Holdout sample observations -------------------")
    print("********************************************************************** \n")
    # comparing actual reconstruction and cross recontruction on original data
    latent_code_Orig_fullTest_orig, recons_orig_Test_from_orig = model_orig(
        torch.Tensor(Df_holdout_orig.iloc[:, :-1].values).to(device), 0)
    _, recons_orig_Test_frommodelR = model_r(latent_code_Orig_fullTest_orig, 1)

    # comparing actual reconstruction and cross recontruction on renamed data
    latent_code_renamed_test, recons_rename_Test_frommodelR = model_r(
        torch.Tensor(DF_holdout_r.iloc[:, :-1].values).to(device), 0)
    _, recons_rename_Test_frommodelOrig = model_orig(latent_code_renamed_test, 1)

    features_reconst_from_crossR_test = recons_orig_Test_frommodelR.cpu().detach().numpy()
    features_true_orig_test = Df_holdout_orig.iloc[:, :-1].values

    features_reconst_from_crossO_test = recons_rename_Test_frommodelOrig.cpu().detach().numpy()
    features_true_renamed_test = DF_holdout_r.iloc[:, :-1].values

    CorMatrix_X1_X1_hat_dir_test = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
    for i in range(num_NonCat_features_orig):
        for j in range(num_NonCat_features_orig):
            temp = stats.pearsonr(features_true_orig_test[:, i],
                                  recons_orig_Test_from_orig.cpu().detach().numpy()[:, j])
            CorMatrix_X1_X1_hat_dir_test[i, j] = temp[0]

    CorMatrix_X2_X2_hat_dir_test = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
    for i in range(num_NonCat_features_r):
        for j in range(num_NonCat_features_r):
            temp = stats.pearsonr(features_true_renamed_test[:, i],
                                  recons_rename_Test_frommodelR.cpu().detach().numpy()[:, j])
            CorMatrix_X2_X2_hat_dir_test[i, j] = temp[0]

    np.savetxt("Cor_dir_X1.csv", np.round(CorMatrix_X1_X1_hat_dir_test, decimals=2), delimiter=",")
    np.savetxt("Cor_dir_X2.csv", np.round(CorMatrix_X2_X2_hat_dir_test, decimals=2), delimiter=",")



    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_X1_hat_cross_test = np.zeros((num_NonCat_features_orig, num_NonCat_features_r))
    CorMatrix_X2_X2_hat_cross_test = np.zeros((num_NonCat_features_r, num_NonCat_features_orig))
    CorMatrix_X1_X1_hat_cross_P_value_test = np.zeros((num_NonCat_features_orig, num_NonCat_features_r))
    CorMatrix_X2_X2_hat_cross_P_value_test = np.zeros((num_NonCat_features_r, num_NonCat_features_orig))

    for i in range(num_NonCat_features_orig):
        for j in range(num_NonCat_features_r):
            temp = stats.pearsonr(features_true_orig_test[:, i], features_reconst_from_crossR_test[:, j])
            CorMatrix_X1_X1_hat_cross_test[i, j] = temp[0]
            CorMatrix_X1_X1_hat_cross_P_value_test[i, j] = temp[1]

    for i in range(num_NonCat_features_r):
        for j in range(num_NonCat_features_orig):
            temp0 = stats.pearsonr(features_true_renamed_test[:, i], features_reconst_from_crossO_test[:, j])
            CorMatrix_X2_X2_hat_cross_test[i, j] = temp0[0]
            CorMatrix_X2_X2_hat_cross_P_value_test[i, j] = temp0[1]

    # selecting the correlation only for unmapped variables
    short_CorMatrix_X1_X1_hat_cross_test = CorMatrix_X1_X1_hat_cross_test[len(mapped_features):,
                                           len(mapped_features):]
    short_CorMatrix_X2_X2_hat_cross_test = CorMatrix_X2_X2_hat_cross_test[len(mapped_features):,
                                           len(mapped_features):]


    """ Calling the stable marriage algorithm for mappings  """

    Mistakes_X1_tr, Mistakes_X2_tr, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(short_CorMatrix_X1_X1_hat_cross_test,
                                                                               short_CorMatrix_X2_X2_hat_cross_test,
                                                                               index_for_mapping_orig_to_rename[
                                                                               len(mapped_features):],
                                                                               index_for_mapping_rename_to_orig[
                                                                               len(mapped_features):],
                                                                               len(mapped_features))

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_tr]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_tr]

    print(" Chimeric  X1_train mistakes", MisF_X1_te)
    print(" Chimeric  X2_train mistakes", MisF_X2_te)

    print(" Chimeric  X1_train mistakes number on holdout set", len(MisF_X1_te), "out of ", num_NonCat_features_orig - len(mapped_features))
    print(" Chimeric  X2_train mistakes number on holdout set", len(MisF_X2_te), "out of ", num_NonCat_features_orig - len(mapped_features))

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr'])

    # Estimated correlation of the transformed data
    temp_inf_x1.estimated_cross_corr = [short_CorMatrix_X1_X1_hat_cross_test[i, j] for i in
                                        range(x1_match_matrix_test.shape[0]) for
                                        j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]

    temp_inf_x2.estimated_cross_corr = [short_CorMatrix_X2_X2_hat_cross_test[i, j] for i in
                                        range(x2_match_matrix_test.shape[0]) for
                                        j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]

    for i in range(len(temp_inf_x1)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]

    for i in range(len(temp_inf_x2)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_r[len(mapped_features) + i]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_orig[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_r[len(mapped_features) + i], reordered_column_names_orig[
                len(mapped_features) + matched_index[0]]]


    print(" -------- Chimeric AE method training ends ------------- \n \n  ")


    # plots for transformation evaluation
    # row_index_no_orig = np.random.choice(len(features_true_orig_test), 500, replace=False)
    # row_index_no_rename = np.random.choice(len(features_true_renamed_test), 500, replace=False)
    # if sq_transf_features != []:
    #     for feature_name in sq_transf_features:
    #         print('\n ---------- feature name --------- \n ', feature_name)
    #         col_idxno_orig = reordered_column_names_orig.index(feature_name)
    #         col_idxno_renamed = reordered_column_names_r.index(feature_name)
    #         # denormalizing_orig = features_true_orig_test[row_index_no_orig, col_idxno_orig]*normalizing_values_orig['std'].loc[feature_name] + normalizing_values_orig['mean'].loc[feature_name]
    #         # sq_orig_feature_value = np.square(denormalizing_orig)
    #         # x_axis = (sq_orig_feature_value-normalizing_values_r['mean'].loc[feature_name])/normalizing_values_r['std'].loc[feature_name]
    #         x_axis = features_true_orig_test[row_index_no_orig, col_idxno_orig]
    #         y_axis = features_reconst_from_crossR_test[row_index_no_orig, col_idxno_renamed]
    #
    #         # x_axis = features_true_orig_test[row_index_no_orig, col_idxno_orig] * \
    #         #                      normalizing_values_orig['std'].loc[feature_name] + normalizing_values_orig['mean'].loc[
    #         #                          feature_name]
    #         # y_axis = features_reconst_from_crossR_test[row_index_no_orig, col_idxno_renamed] * \
    #         #          normalizing_values_r['std'].loc[feature_name] + normalizing_values_r['mean'].loc[feature_name]
    #
    #         plt.scatter(x_axis, y_axis, color='blue')
    #         plt.xlabel("true squared feature value")
    #         plt.ylabel("reconstructed feature value ")
    #         temp = stats.pearsonr(x_axis, y_axis)[0]
    #         plt.figtext(0.6, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
    #         # plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #         # plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #         plt.title(" number of mapped feature  " + str(
    #             mpfeatures) + " & " + str(feature_name) + " squared in renamed data", fontsize=8)
    #         if feature_name in mapped_features_updated_orig:
    #             plt.savefig(
    #                 filename_for_saving_tran_quality + "_GotMappedafter_KMF_sqfeat_" + str(feature_name) + ".pdf", bbox='tight')
    #             plt.savefig(
    #                 filename_for_saving_tran_quality + "_GotMappedafter_KMF_sqfeat_" + str(feature_name) + ".png", bbox='tight')
    #         else:
    #             plt.savefig(
    #                 filename_for_saving_tran_quality + "_sqfeat_"+str(feature_name) + ".pdf", bbox='tight')
    #             plt.savefig(
    #                 filename_for_saving_tran_quality + "_sqfeat_"+str(feature_name) + ".png", bbox='tight')
    #         plt.close()
    #
    # for feature_name in reordered_column_names_orig[:-1]:
    #     col_idxno_orig = list(Df_holdout_orig.columns).index(feature_name)
    #     x_axis = features_true_orig_test[row_index_no_orig,col_idxno_orig]
    #     y_axis = recons_orig_Test_from_orig.cpu().detach().numpy()[row_index_no_orig,col_idxno_orig]
    #     plt.scatter(x_axis, y_axis, color='blue')
    #     plt.xlabel("true feature value (x1) ")
    #     plt.ylabel("direct reconstructed feature value (x1)")
    #     temp = stats.pearsonr(x_axis,y_axis)[0]
    #     plt.figtext(0.6,0.8,"Cor_value = "+str(np.round(temp, decimals=3)))
    #     plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     plt.title(" number of mapped feature  " + str(
    #         mpfeatures)+ " & feature name is " + str(feature_name) , fontsize=8)
    #     if feature_name in mapped_features_updated_orig:
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_Mappedafter_KMF" + "_Direct_recons_X1_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_Mappedafter_KMF" + "_Direct_recons_X1_" + str(feature_name) + ".png", bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_NOTMappedafter_KMF_Direct_recons_X1_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_NOTMappedafter_KMF_Direct_recons_X1_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    #
    # for feature_name in reordered_column_names_r[:-1]:
    #     col_idxno_r = list(DF_holdout_r.columns).index(feature_name)
    #     x_axis = features_true_renamed_test[row_index_no_orig,col_idxno_r]
    #     y_axis = recons_rename_Test_frommodelR.cpu().detach().numpy()[row_index_no_orig,col_idxno_r]
    #     plt.scatter(x_axis, y_axis, color='blue')
    #     plt.xlabel("true feature value (x2) ")
    #     plt.ylabel("direct reconstructed feature value (x2)")
    #     temp = stats.pearsonr(x_axis,y_axis)[0]
    #     plt.figtext(0.6,0.8,"Cor_value = "+str(np.round(temp, decimals=3)))
    #     plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     plt.title(" number of mapped feature  " + str(
    #         mpfeatures)+ " & feature name is " + str(feature_name) , fontsize=8)
    #     if feature_name in mapped_features_updated_r:
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_Mappedafter_KMF" + "_Direct_recons_X2_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_Mappedafter_KMF" + "_Direct_recons_X2_" + str(feature_name) + ".png", bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_NOTMappedafter_KMF_Direct_recons_X2_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_tran_quality + "_NOTMappedafter_KMF_Direct_recons_X2_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    del df_rename_preproc
    # exit()
    return MisF_X1_te, MisF_X2_te, temp_inf_x1, temp_inf_x2


def RadialGAN(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features, partition_no, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r):
    device = torch.device('cuda')
    num_features = len(reordered_column_names_r) -1

    print(" -------- RadialGAN method training starts ------------- \n \n  ")


    """ ==================== GRADIENT PENALTY ======================== """

    def calc_gradient_penalty(real_data, fake_data, D_num):
        alpha0 = torch.rand(real_data.size()[0], 1).to(device)
        alpha0 = alpha0.expand(real_data.size())

        interpolates = alpha0 * real_data + ((1 - alpha0) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        if D_num == 1:
            disc_interpolates = D1(interpolates)
        if D_num == 2:
            disc_interpolates = D2(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device = 'cuda'),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * beta

        # print(gradient_penalty)
        return gradient_penalty

    """ ==================== Setting up the disciminators ======================== """

    Wxh_o = weights_init(size=[num_features, num_features])
    bxh_o = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3y_o = weights_init(size=[num_features, 1])
    bh3y_o = Variable(torch.zeros(1, device = "cuda"), requires_grad=True)

    Wxh_r = weights_init(size=[num_features, num_features])
    bxh_r = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3y_r = weights_init(size=[num_features, 1])
    bh3y_r = Variable(torch.zeros(1, device = "cuda"), requires_grad=True)

    def D1(X):
        h = torch.tanh(X.mm(Wxh_o) + bxh_o.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o) + bhh2_o.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o) + bh2h3_o.repeat(h_2.size(0), 1))
        y = torch.sigmoid(h_3.mm(Wh3y_o) + bh3y_o.repeat(h_3.size(0), 1))
        # y = F.linear(h_3, Wh3y_o.t(), bh3y_o.repeat(h_3.size(0), 1))
        return y  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def D2(X):
        h = torch.tanh(X.mm(Wxh_r) + bxh_r.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_r) + bhh2_r.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_r) + bh2h3_r.repeat(h_2.size(0), 1))
        y = torch.sigmoid(h_3.mm(Wh3y_r) + bh3y_r.repeat(h_3.size(0), 1))
        # y = F.linear(h_3, Wh3y_r.t(), bh3y_r.repeat(h_3.size(0), 1))
        return y  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    D1_params = [Wxh_o, bxh_o, Whh2_o, bhh2_o, Wh2h3_o, bh2h3_o, Wh3y_o, bh3y_o]
    D2_params = [Wxh_r, bxh_r, Whh2_r, bhh2_r, Wh2h3_r, bh2h3_r, Wh3y_r, bh3y_r]

    """ ==================== Setting up the encoders F ======================== """

    Wxh_o_F = weights_init(size=[num_features, num_features])
    bxh_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o_F = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o_F = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o_F = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3z_o_F = weights_init(size=[num_features, num_features])
    bh3z_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wxh_r_F = weights_init(size=[num_features, num_features])
    bxh_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r_F = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r_F = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r_F = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3z_r_F = weights_init(size=[num_features, num_features])
    bh3z_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    def F1(X):
        h = torch.tanh(X.mm(Wxh_o_F) + bxh_o_F.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o_F) + bhh2_o_F.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o_F) + bh2h3_o_F.repeat(h_2.size(0), 1))
        z = torch.sigmoid(h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1))
        # z = h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1)
        return z  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def F2(X):
        h = torch.tanh(X.mm(Wxh_o_F) + bxh_o_F.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o_F) + bhh2_o_F.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o_F) + bh2h3_o_F.repeat(h_2.size(0), 1))
        z = torch.sigmoid(h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1))
        # z = h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1)
        return z  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    F1_params = [Wxh_o_F, bxh_o_F, Whh2_o_F, bhh2_o_F, Wh2h3_o_F, bh2h3_o_F, Wh3z_o_F, bh3z_o_F]
    F2_params = [Wxh_r_F, bxh_r_F, Whh2_r_F, bhh2_r_F, Wh2h3_r_F, bh2h3_r_F, Wh3z_r_F, bh3z_r_F]

    """ ==================== Setting up the decoders G ======================== """

    Wzh_o_G = weights_init(size=[num_features, num_features])
    bzh_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o_G = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o_G = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o_G = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3x_o_G = weights_init(size=[num_features, num_features])
    bh3x_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wzh_r_G = weights_init(size=[num_features, num_features])
    bzh_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r_G = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r_G = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r_G = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3x_r_G = weights_init(size=[num_features, num_features])
    bh3x_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    def G1(Z):
        h = F.tanh(Z.mm(Wzh_o_G) + bzh_o_G.repeat(Z.size(0), 1))
        h_2 = F.tanh(h.mm(Whh2_o_G) + bhh2_o_G.repeat(h.size(0), 1))
        h_3 = F.tanh(h_2.mm(Wh2h3_o_G) + bh2h3_o_G.repeat(h_2.size(0), 1))
        # x = F.sigmoid(h_3.mm(Wh3x_o_G) + bh3x_o_G.repeat(h_3.size(0), 1))
        x = h_3.mm(Wh3x_o_G) + bh3x_o_G.repeat(h_3.size(0), 1)
        return x  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def G2(Z):
        h = F.tanh(Z.mm(Wzh_r_G) + bzh_r_G.repeat(Z.size(0), 1))
        h_2 = F.tanh(h.mm(Whh2_r_G) + bhh2_r_G.repeat(h.size(0), 1))
        h_3 = F.tanh(h_2.mm(Wh2h3_r_G) + bh2h3_r_G.repeat(h_2.size(0), 1))
        # x = F.sigmoid(h_3.mm(Wh3x_r_G) + bh3x_r_G.repeat(h_3.size(0), 1))
        x = h_3.mm(Wh3x_r_G) + bh3x_r_G.repeat(h_3.size(0), 1)
        return x  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    G1_params = [Wzh_o_G, bzh_o_G, Whh2_o_G, bhh2_o_G, Wh2h3_o_G, bh2h3_o_G, Wh3x_o_G, bh3x_o_G]
    G2_params = [Wzh_r_G, bzh_r_G, Whh2_r_G, bhh2_r_G, Wh2h3_r_G, bh2h3_r_G, Wh3x_r_G, bh3x_r_G]

    params = D1_params + D2_params + F1_params + F2_params + G1_params + G2_params

    """ ===================== TRAINING STARTS ======================== """

    def reset_grad():
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()).to(device)

    # some data formatting for getting the minibatches
    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batch_size, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batch_size, shuffle=True, num_workers=1)

    # create optimizer objects
    F1_solver = optim.Adam(F1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    G1_solver = optim.Adam(G1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    D1_solver = optim.Adam(D1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)

    F2_solver = optim.Adam(F2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    G2_solver = optim.Adam(G2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    D2_solver = optim.Adam(D2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)

    # lr scheduler
    scheduler_F1 = torch.optim.lr_scheduler.ReduceLROnPlateau(F1_solver, patience=2, verbose=True)
    scheduler_G1 = torch.optim.lr_scheduler.ReduceLROnPlateau(G1_solver, patience=2, verbose=True)
    scheduler_D1 = torch.optim.lr_scheduler.ReduceLROnPlateau(D1_solver, patience=2, verbose=True)

    scheduler_F2 = torch.optim.lr_scheduler.ReduceLROnPlateau(F2_solver, patience=2, verbose=True)
    scheduler_G2 = torch.optim.lr_scheduler.ReduceLROnPlateau(G2_solver, patience=2, verbose=True)
    scheduler_D2 = torch.optim.lr_scheduler.ReduceLROnPlateau(D2_solver, patience=2, verbose=True)

    total_loss = []
    dis_o_loss = []
    dis_r_loss = []
    G_o_loss = []
    G_r_loss = []
    CYC_o_loss_dir = []
    CYC_r_loss_dir = []
    CYC_o_loss_cross = []
    CYC_r_loss_cross = []

    for epoch in range(epochs_RadialGAN):
        loss = 0
        D_o_loss = 0
        D_r_loss = 0
        gen_o_loss = 0
        gen_r_loss = 0
        cyc_o_loss_dir = 0
        cyc_r_loss_dir = 0
        cyc_o_loss_cross = 0
        cyc_r_loss_cross = 0

        D1_within_epoch_loss = []
        D2_within_epoch_loss = []

        gen1_within_epoch_loss = []
        gen2_within_epoch_loss = []

        cyc1_within_epoch_loss = []
        cyc2_within_epoch_loss = []

        # cyc1_within_epoch_loss_cross = []
        # cyc2_within_epoch_loss_cross = []

        counter_num_mb_for_dis = 0
        for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
            if len(data[0][1]) == len(data[1][1]):
                x_o = data[0][1].to(device)
                x_r = data[1][1].to(device)

                if i == 0 or i % 5 != 0:
                    """  training D1  """
                    D1_real = D1(x_o)
                    D1_fake = D1(G1(F2(x_r)))

                    # train with gradient penalty
                    gradient_penalty1 = calc_gradient_penalty(x_o, G1(F2(x_r)), 1)
                    gradient_penalty1.backward()

                    D1_loss0 = -(torch.mean(D1_real) - torch.mean(D1_fake))
                    D1_loss0.backward()

                    D1_loss = D1_loss0 + gradient_penalty1
                    D1_solver.step()

                    D_o_loss = D_o_loss + D1_loss.item()

                    """  training D2  """
                    D2_real = D2(x_r)
                    D2_fake = D2(G2(F1(x_o)))

                    # train with gradient penalty
                    gradient_penalty2 = calc_gradient_penalty(x_r, G2(F1(x_o)), 2)
                    gradient_penalty2.backward()

                    D2_loss0 = -(torch.mean(D2_real) - torch.mean(D2_fake))
                    D2_loss0.backward()

                    D2_loss = D2_loss0 + gradient_penalty2
                    D2_solver.step()

                    D_r_loss = D_r_loss + D2_loss.item()

                    counter_num_mb_for_dis += 1
                    D1_within_epoch_loss.append(D1_loss.item())
                    D2_within_epoch_loss.append(D2_loss.item())
                    gen1_within_epoch_loss.append(0)
                    gen2_within_epoch_loss.append(0)
                    cyc1_within_epoch_loss.append(0)
                    cyc2_within_epoch_loss.append(0)

                    # Housekeeping - reset gradient
                    reset_grad()

                if i != 0 and i % 5 == 0:
                    D1_fake = D1(G1(F2(x_r)))
                    D2_fake = D2(G2(F1(x_o)))

                    gen1_loss = - torch.mean(D1_fake)
                    gen2_loss = - torch.mean(D2_fake)

                    # cycle consistency loss
                    cyc_loss1 = torch.norm(x_o - G1(F1(x_o))) + torch.norm(F2(x_r) - F1(G1(F2(x_r))))
                    cyc_loss2 = torch.norm(x_r - G2(F2(x_r))) + torch.norm(F1(x_o) - F2(G2(F1(x_o))))

                    # tr_loss = gen1_loss + gen2_loss + Lambda* (cyc_loss1 + cyc_loss2)
                    tr_loss = gen1_loss + gen2_loss + Lambda_dir * (
                                torch.norm(x_o - G1(F1(x_o))) + torch.norm(x_r - G2(F2(x_r)))) + Lambda_on_z * (
                                          torch.norm(F2(x_r) - F1(G1(F2(x_r)))) + torch.norm(F2(x_r) - F1(G1(F2(x_r)))))

                    tr_loss.backward()

                    # updating the parameters
                    G1_solver.step()
                    G2_solver.step()
                    F1_solver.step()
                    F2_solver.step()

                    loss = loss + tr_loss.item()
                    gen_o_loss = gen_o_loss + gen1_loss.item()
                    gen_r_loss = gen_r_loss + gen2_loss.item()
                    cyc_o_loss_dir = cyc_o_loss_dir + torch.norm(x_o - G1(F1(x_o))).item()
                    cyc_r_loss_dir = cyc_r_loss_dir + torch.norm(x_r - G2(F2(x_r))).item()
                    cyc_o_loss_cross = cyc_o_loss_cross + torch.norm(F2(x_r) - F1(G1(F2(x_r)))).item()
                    cyc_r_loss_cross = cyc_r_loss_cross + torch.norm(F1(x_o) - F2(G2(F1(x_o)))).item()

                    D1_within_epoch_loss.append(0)
                    D2_within_epoch_loss.append(0)
                    gen1_within_epoch_loss.append(gen1_loss.item())
                    gen2_within_epoch_loss.append(gen2_loss.item())
                    cyc1_within_epoch_loss.append(cyc_loss1.item())
                    cyc2_within_epoch_loss.append(cyc_loss2.item())

                    # Housekeeping - reset gradient
                    reset_grad()

        # compute the epoch training loss
        loss = loss / (5 * (len(train_loader_orig) + len(train_loader_r)))
        D_o_loss = D_o_loss / (5 * len(train_loader_orig))
        D_r_loss = D_r_loss / (5 * len(train_loader_r))
        gen_o_loss = gen_o_loss / (5 * len(train_loader_orig))
        gen_r_loss = gen_r_loss / (5 * len(train_loader_r))
        cyc_o_loss_dir = cyc_o_loss_dir / (5 * len(train_loader_orig))
        cyc_r_loss_dir = cyc_r_loss_dir / (5 * len(train_loader_r))
        cyc_o_loss_cross = cyc_o_loss_cross / (5 * len(train_loader_orig))
        cyc_r_loss_cross = cyc_r_loss_cross / (5 * len(train_loader_r))

        scheduler_D1.step(D_o_loss)
        scheduler_D2.step(D_r_loss)

        # display the epoch training loss
        print("epoch : {}/{}, total loss = {:.8f}".format(epoch + 1, epochs, loss))
        print("epoch : {}/{}, cyc orig= {:.8f}".format(epoch + 1, epochs, cyc_o_loss_dir + cyc_o_loss_cross))
        print("epoch : {}/{}, cyc r= {:.8f}".format(epoch + 1, epochs, cyc_r_loss_cross + cyc_r_loss_dir))
        print("epoch : {}/{}, gen orig = {:.8f}".format(epoch + 1, epochs, gen_o_loss))
        print("epoch : {}/{}, gen r = {:.8f}".format(epoch + 1, epochs, gen_r_loss))
        print("epoch : {}/{}, critic loss ae orig= {:.8f}".format(epoch + 1, epochs, D_o_loss))
        print("epoch : {}/{}, critic loss ae r= {:.8f}".format(epoch + 1, epochs, D_r_loss))

        total_loss.append(loss)
        G_o_loss.append(gen_o_loss)
        G_r_loss.append(gen_r_loss)
        CYC_o_loss_dir.append(cyc_o_loss_dir)
        CYC_r_loss_dir.append(cyc_r_loss_dir)
        CYC_o_loss_cross.append(cyc_o_loss_cross)
        CYC_r_loss_cross.append(cyc_r_loss_cross)
        dis_o_loss.append(D_o_loss)
        dis_r_loss.append(D_r_loss)

    # saving_dir = './AE_reconstruction_stuff/Permutation_Datafrom_2021-04-16/Avg_all_random/Frac_Shuffled_' + str(
    #     frac_renamed) + '/RadialGAN_#ofhidden_layers_' + str(
    #     num_of_hidden_layers) + '/Syn_' + str(dataset_number)
    #
    # if not os.path.exists(saving_dir):
    #     os.makedirs(saving_dir)
    #
    # plt.plot(np.array(dis_o_loss), '-o', color='black', label='Dis_orig')
    # plt.plot(np.array(dis_r_loss), '-o', color='red', label='Dis_r')
    # plt.plot(np.array(G_o_loss), '-o', color='blue', label='Gen_orig')
    # plt.plot(np.array(G_r_loss), '-o', color='green', label='Gen_r')
    # plt.plot(np.array(CYC_o_loss_dir), '-o', color='magenta', label='Cyc_orig dir')
    # plt.plot(np.array(CYC_r_loss_dir), '-o', color='yellow', label='Cyc_r_dir')
    # plt.plot(np.array(CYC_o_loss_cross), '-o', color='brown', label='Cyc_orig_on_z')
    # plt.plot(np.array(CYC_r_loss_cross), '-o', color='orange', label='Cyc_r_on_z')
    # plt.title("Training loss over the epochs " + " #epochs " + str(epochs))
    # plt.xlabel("epochs")
    # plt.ylabel("loss value")
    # plt.legend()
    # plt.savefig(saving_dir + "/Training_loss_Across_" + str(epochs) + "epochs_perm#_" + str(partition_no) + ".png",
    #             bbbox='tight')
    # plt.close()
    print("Combined AE loss for partition ", partition_no, " is ", loss)

    """ ===================== TRAINING ENDS ======================== """

    # preparing to get the generated/reconstructed values

    features_true_orig = Df_holdout_orig.iloc[:, :-1].values
    features_true_renamed = DF_holdout_r.iloc[:, :-1].values

    gen_features_orig = G1(F2(torch.Tensor(DF_holdout_r.iloc[:, :-1].values).to(device))).cpu().detach().numpy()
    gen_features_r = G2(F1(torch.Tensor(Df_holdout_orig.iloc[:, :-1].values).to(device))).cpu().detach().numpy()

    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_X1_hat_cross = np.zeros((num_features, num_features))
    CorMatrix_X2_X2_hat_cross = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            CorMatrix_X1_X1_hat_cross[i, j] = \
                stats.pearsonr(features_true_orig[:, i], gen_features_r[:, j])[0]
            CorMatrix_X2_X2_hat_cross[i, j] = \
                stats.pearsonr(features_true_renamed[:, i], gen_features_orig[:, j])[0]

    # selecting the correlation only for unmapped variables
    short_CorMatrix_X1_X1_hat_cross = CorMatrix_X1_X1_hat_cross[len(mapped_features):,
                                      len(mapped_features):]
    short_CorMatrix_X2_X2_hat_cross = CorMatrix_X2_X2_hat_cross[len(mapped_features):,
                                      len(mapped_features):]


    """ Calling the stable marriage algorithm for mappings  """

    Mistakes_X1_te, Mistakes_X2_te, _, _ = Stable_matching_algorithm(short_CorMatrix_X1_X1_hat_cross,
                                                                               short_CorMatrix_X2_X2_hat_cross,
                                                                               index_for_mapping_orig_to_rename[
                                                                               len(mapped_features):],
                                                                               index_for_mapping_rename_to_orig[
                                                                               len(mapped_features):],
                                                                               len(mapped_features))
    mpfeatures = len(mapped_features)
    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

    print(" RadialGAN  X1_train mistakes", MisF_X1_te)
    print(" RadialGAN  X2_train mistakes", MisF_X2_te)

    print(" RadialGAN  X1_train mistakes number on holdout set", len(MisF_X1_te), "out of ", num_features - len(mapped_features))
    print(" RadialGAN  X2_train mistakes number on holdout set", len(MisF_X2_te), "out of ", num_features - len(mapped_features))

    print(" -------- RadialGAN method training ends ------------- \n \n  ")

    del df_rename_preproc

    return MisF_X1_te, MisF_X2_te


def main(dataset_no_sample):

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(6)
    print("Number of threads being used are ", torch.get_num_threads())

    random.seed(100)
    np.random.seed(100)  # fixing the seed for reproducibility

    # reading the data
    filename = "./Syn_data/SD_" + str(dataset_number) + "/2021-05-18Syn_Data_" + str(
        dataset_number) + "_Sample_no_" + str(
        dataset_no_sample) + "_size_20_10000_for_AE_balanced.csv"  # for dataset 1 and 2
    # filename = "./Syn_data/SD_" + str(dataset_number) + "/2021-05-20Syn_Data_" + str(dataset_number) +  "_Sample_no_" + str(dataset_no_sample) +"_size_20_10000_for_AE_balanced.csv"
    full_Data0 = pd.read_csv(filename)  # for dataset 4
    full_data_points = full_Data0.shape[0]
    num_features = full_Data0.shape[1] - 1

    # full data initial correlation
    Feature_matrix = full_Data0.iloc[:, :-1]
    Cor_from_df = Feature_matrix.corr()

    # output arrays

    AVG_MISMATCHES_X1_tr = np.zeros((len(list_of_number_samples), n_t))
    AVG_MISMATCHES_X2_tr = np.zeros((len(list_of_number_samples), n_t))

    Frac_mismatches_across_trial_perm_X1_tr = np.zeros((len(list_of_number_samples), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr = np.zeros((len(list_of_number_samples), n_t * n_p))

    AVG_MISMATCHES_X1_tr_sim_Cor = np.zeros((len(list_of_number_samples), n_t))
    AVG_MISMATCHES_X2_tr_sim_Cor = np.zeros((len(list_of_number_samples), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_number_samples), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_number_samples), n_t * n_p))

    AVG_MISMATCHES_X1_tr_RG = np.zeros((len(list_of_number_samples), n_t))
    AVG_MISMATCHES_X2_tr_RG = np.zeros((len(list_of_number_samples), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_RG = np.zeros((len(list_of_number_samples), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_RG = np.zeros((len(list_of_number_samples), n_t * n_p))

    AVG_MISMATCHES_X1_tr_Kang = np.zeros((len(list_of_number_samples), n_t))
    AVG_MISMATCHES_X2_tr_Kang = np.zeros((len(list_of_number_samples), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_number_samples), n_t * n_p))
    
    # the set of mapped features is selected apriori adn that same set is used across different sample sizes to avoid variation due to the mapped features
    mapped_random = []
    for i in range(n_t):
        mapped_random.append(np.random.choice(num_features, mpfeatures, replace=False))

    mapped_random = np.array(mapped_random)

    sample_index_list = []  # list that keeps track of the indices that have been already included in the previous case and needs to be avoided
    m = 0  # variables to keep track of the iterations over number of mapped features
    for num_samples in list_of_number_samples:
        run_num = 0  # variable to keep track of the run number out of n_t*n_p

        print("\n ********************************************************")
        print("Run when there are ", num_samples, " samples starts")
        print(" ******************************************************** \n")

        # getting the indexes
        indices = np.random.choice(np.array(list(set(np.arange(full_data_points)) - set(sample_index_list))),
                                   num_samples - len(sample_index_list), replace=False)
        print("Number of common indices between the new samples and existing ones ",
              len(set(indices).intersection(set(sample_index_list))))
        sample_index_list = np.append(sample_index_list, indices)
        print("number of indices sampled in this round ", len(indices))
        print("number of indices sampled till now", len(sample_index_list))

        full_Data = full_Data0.iloc[sample_index_list].copy()

        for trial in range(n_t):
            
            # array for saving the frac of mistakes
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG = np.zeros(n_p)
            
            # the copies are being made because of the multiple trials
            df_train = full_Data.copy()

            print("\n ********************************************************")
            print("Trial number ", trial + 1, "   starts when there are ", num_samples, " samples")
            print(" ******************************************************** \n")

            # shuffling the mapped and unmapped
            unmapped_random = list(set(np.arange(num_features)) - set(mapped_random[trial]))

            mapped_features = ["Col" + str(i + 1) for i in mapped_random[trial]]
            possible_feat_to_shuffle = ["Col" + str(i + 1) for i in unmapped_random]

            print("List of mapped features for trial number", trial + 1, "is ", mapped_features)

            # reordering to making sure that mapped features are at the starts of the vector
            feature_names = mapped_features + possible_feat_to_shuffle
            df_train = df_train.reindex(columns=feature_names + [outcome])

            # keeping a holdout sample aside
            Df_for_training, Df_holdout = model_selection.train_test_split(df_train, test_size=0.1,
                                                                           random_state=42 * trial * 10,
                                                                           stratify=df_train[outcome])

            # splitting the holdout df into two for using in the two databases
            Df_holdout_orig0, DF_holdout_r0 = model_selection.train_test_split(Df_holdout, test_size=0.5,
                                                                               random_state=42,
                                                                               stratify=Df_holdout[outcome])

            df_train1, df_train2 = model_selection.train_test_split(Df_for_training, test_size=frac_renamed,
                                                                    random_state=42 * trial * 10,
                                                                    stratify=Df_for_training[outcome])

            print(" trial data details \n")
            print("size of total train", len(df_train))
            print("size of train original", len(df_train1))
            print("size of train renamed", len(df_train2))

            device = torch.device('cpu')

            """ ORIGINAL DATA PREP """
            # data pre-processing normalization
            normalizing_values_orig = {}
            normalizing_values_orig['mean'] = df_train1[feature_names].mean(axis=0)
            normalizing_values_orig['std'] = df_train1[feature_names].std(axis=0)
            normalizing_values_orig['min'] = df_train1[feature_names].min(axis=0)
            normalizing_values_orig['max'] = df_train1[feature_names].max(axis=0)

            df_train_preproc0 = normalization(df_train1, 'mean_std', normalizing_values_orig, feature_names)
            Df_holdout_orig0 = normalization(Df_holdout_orig0, 'mean_std', normalizing_values_orig, feature_names)
            reordered_column_names_orig = mapped_features + [col for col in df_train_preproc0.columns if
                                                             col not in mapped_features + [outcome]] + [outcome]
            df_train_preproc0 = df_train_preproc0.reindex(columns=reordered_column_names_orig)
            Df_holdout_orig0 = Df_holdout_orig0.reindex(columns=reordered_column_names_orig)

            """ SHUFFLED FEATURES DATA PREP """
            feature_names_r = feature_names

            # data preprocessing
            normalizing_values_r = {}
            normalizing_values_r['mean'] = df_train2[feature_names_r].mean(axis=0)
            normalizing_values_r['std'] = df_train2[feature_names_r].std(axis=0)
            normalizing_values_r['min'] = df_train2[feature_names_r].min(axis=0)
            normalizing_values_r['max'] = df_train2[feature_names_r].max(axis=0)

            df_rename_preproc0 = normalization(df_train2, 'mean_std', normalizing_values_r, feature_names_r)
            DF_holdout_r0 = normalization(DF_holdout_r0, 'mean_std', normalizing_values_r, feature_names_r)

            # maximum possible mistakes for this trial
            max_mistakes = len(feature_names) - len(mapped_features)

            for partition in range(n_p):
                df_train_preproc = df_train_preproc0.copy()
                df_rename_preproc = df_rename_preproc0.copy()  # a copy to keep the original ordering as a baseline when matching
                DF_holdout_r = DF_holdout_r0.copy()
                Df_holdout_orig = Df_holdout_orig0.copy()
                
                print("\n ********************************************************")
                print(" Partition number ", partition + 1, "   starts for trail number ", trial + 1,
                      " when there are ",
                      num_samples, " samples")
                print(" ******************************************************** \n")

                # reordering the features (PERMUTATION)
                reorder_feat = possible_feat_to_shuffle.copy()
                random.shuffle(reorder_feat)
                index_for_mapping_orig_to_rename = [reorder_feat.index(num) + len(mapped_features) + 1 for num
                                                    in
                                                    [col for col in df_train_preproc.columns if
                                                     col not in mapped_features + [outcome]]]
                index_for_mapping_rename_to_orig = [[col for col in df_train_preproc.columns if
                                                     col not in mapped_features + [outcome]].index(num) + len(
                    mapped_features) + 1 for num in reorder_feat]

                # adding index variables for the mapped variables at the start of the list
                index_for_mapping_orig_to_rename = list(
                    np.arange(1, mpfeatures + 1)) + index_for_mapping_orig_to_rename
                index_for_mapping_rename_to_orig = list(
                    np.arange(1, mpfeatures + 1)) + index_for_mapping_rename_to_orig
                print(" Index for mapping orig to rename ", index_for_mapping_orig_to_rename)
                print(" Index for mapping rename to original ", index_for_mapping_rename_to_orig)

                reordered_column_names_r = mapped_features + reorder_feat + [outcome]
                df_rename_preproc = df_rename_preproc.reindex(columns=reordered_column_names_r)
                DF_holdout_r = DF_holdout_r.reindex(columns=reordered_column_names_r)

                print("\n \n ------  Ordering of variables when # of mapped features is ", mpfeatures,
                      " trial number is ",
                      trial + 1, " partition number is ", partition + 1, "\n ")
                print(" Original dataset \n ", reordered_column_names_orig)
                print(" Permuted features dataset \n", reordered_column_names_r)

                """ AE part preprocessing  starts   """

                Mistakes_X1_te_KANG = Kang_MI_HC_opt_with_Euclidean_dist(
                    df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename,
                    df_train_preproc.columns, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r)

                Mistakes_X1_tr_sim_Cor, Mistakes_X2_tr_sim_Cor, match_details_x1_sim_cor, match_details_x2_sim_cor, mapp_fea_to_add, mapp_fea_to_add_match = Simple_maximum_sim_viaCorrelation(
                    df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename,
                    index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r,
                    mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r)

                Mistakes_X1_tr_RG, Mistakes_X2_tr_RG = RadialGAN(df_train_preproc.copy(), df_rename_preproc.copy(),
                                                                 index_for_mapping_orig_to_rename,
                                                                 index_for_mapping_rename_to_orig,
                                                                 reordered_column_names_orig, reordered_column_names_r,
                                                                 mapped_features, partition, Cor_from_df,
                                                                 Df_holdout_orig,
                                                                 DF_holdout_r)

                # prep for second stage

                mapped_features_updated_orig = mapped_features + mapp_fea_to_add
                mapped_features_updated_r = mapped_features + mapp_fea_to_add_match

                remaining_unmapped_feature_orig = [col for col in df_train_preproc.columns if
                                                   col not in mapped_features_updated_orig + [outcome]]
                remaining_unmapped_feature_r = [col for col in df_rename_preproc.columns if
                                                col not in mapped_features_updated_r + [outcome]]

                reordered_column_names_orig_updated = mapped_features_updated_orig + remaining_unmapped_feature_orig + [
                    outcome]
                reordered_column_names_r_updated = mapped_features_updated_r + remaining_unmapped_feature_r + [outcome]

                # getting the updated indices

                index_for_mapping_orig_to_rename_updated = [reordered_column_names_r_updated.index(num) + 1 for num
                                                            in
                                                            [col for col in reordered_column_names_orig_updated if
                                                             col not in [outcome]]]
                index_for_mapping_rename_to_orig_updated = [reordered_column_names_orig_updated.index(num) + 1 for num
                                                            in
                                                            [col for col in reordered_column_names_r_updated if
                                                             col not in [outcome]]]

                print(" Index for mapping orig to rename second phase ", index_for_mapping_orig_to_rename_updated)
                print(" Index for mapping rename to original second phase ", index_for_mapping_rename_to_orig_updated)

                df_train_preproc = df_train_preproc.reindex(columns=reordered_column_names_orig_updated)
                Df_holdout_orig = Df_holdout_orig.reindex(columns=reordered_column_names_orig_updated)

                df_rename_preproc = df_rename_preproc.reindex(columns=reordered_column_names_r_updated)
                DF_holdout_r = DF_holdout_r.reindex(columns=reordered_column_names_r_updated)

                filename_for_saving_tran_quality = saving_dir + "/Syn" + str(
                    dataset_number) + "_Transf_quality_sq_part#_" + str(partition) + "_trial#_" + str(
                    trial) + "_#mfeat_" + str(mpfeatures) + "_L_dim_" + str(hidden_dim)
                Mistakes_X1_tr, Mistakes_X2_tr, match_details_x1, match_details_x2 = Train_cross_AE(
                    df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename_updated,
                    index_for_mapping_rename_to_orig_updated, reordered_column_names_orig_updated,
                    reordered_column_names_r_updated, mapped_features, mapped_features_updated_orig,
                    mapped_features_updated_r, Cor_from_df, Df_holdout_orig, DF_holdout_r,
                    filename_for_saving_tran_quality, normalizing_values_orig, normalizing_values_r)

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = len(Mistakes_X1_tr) / max_mistakes
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = len(Mistakes_X2_tr) / max_mistakes

                Frac_mismatches_across_trial_perm_X1_tr[m, run_num] = len(Mistakes_X1_tr) / max_mistakes
                Frac_mismatches_across_trial_perm_X2_tr[m, run_num] = len(Mistakes_X2_tr) / max_mistakes

                # no_match_inference_df_from_x1 = pd.concat([no_match_inference_df_from_x1, match_details_x1])
                # no_match_inference_df_from_x2 = pd.concat([no_match_inference_df_from_x2, match_details_x2])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = len(
                    Mistakes_X1_tr_sim_Cor) / max_mistakes
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = len(
                    Mistakes_X2_tr_sim_Cor) / max_mistakes

                Frac_mismatches_across_trial_perm_X1_tr_sim_Cor[m, run_num] = len(Mistakes_X1_tr_sim_Cor) / max_mistakes
                Frac_mismatches_across_trial_perm_X2_tr_sim_Cor[m, run_num] = len(Mistakes_X2_tr_sim_Cor) / max_mistakes

                # no_match_inference_df_from_x1_Sim_cor = pd.concat(
                #     [no_match_inference_df_from_x1_Sim_cor, match_details_x1_sim_cor])
                # no_match_inference_df_from_x2_Sim_cor = pd.concat(
                #     [no_match_inference_df_from_x2_Sim_cor, match_details_x2_sim_cor])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = len(Mistakes_X1_te_KANG) / (
                    max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_Kang[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG[partition] = len(Mistakes_X1_tr_RG) / max_mistakes
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG[partition] = len(Mistakes_X2_tr_RG) / max_mistakes

                Frac_mismatches_across_trial_perm_X1_tr_RG[m, run_num] = len(Mistakes_X1_tr_RG) / max_mistakes
                Frac_mismatches_across_trial_perm_X2_tr_RG[m, run_num] = len(Mistakes_X2_tr_RG) / max_mistakes

                run_num = run_num + 1

                # Deleting the reshuffled as we have already made a copy earlier
                del df_rename_preproc, df_train_preproc, DF_holdout_r, Df_holdout_orig

                # storing the averaged mismatches across all paritition for a fixed trial and fixed number of mapped features
                
                
            print(" Chimeric AE when the total sample size is ", num_samples)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr != []:
                AVG_MISMATCHES_X1_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr != []:
                AVG_MISMATCHES_X2_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            print(" Simple_correlation the total sample size is ", num_samples)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor != []:
                AVG_MISMATCHES_X1_tr_sim_Cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor != []:
                AVG_MISMATCHES_X2_tr_sim_Cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            print(" Kang et al's MI and HC based method the total sample size is ", num_samples)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang != []:
                AVG_MISMATCHES_X1_tr_Kang[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)

            print(" Radial GAN the total sample size is ", num_samples)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG != []:
                AVG_MISMATCHES_X1_tr_RG[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG != []:
                AVG_MISMATCHES_X2_tr_RG[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG)

            print("------")

        m = m + 1
        del full_Data

    print(" ----  Two stage AE ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr)

    print(" ----  Simple_correlation ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_sim_Cor)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_sim_Cor)

    print(" ----  Kang et al's method ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_Kang)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_Kang)

    print(" ----  RadialGAN ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_RG)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_RG)

    # no_match_inference_df_from_x1.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
    #     hidden_dim) + "L_dim_from_hold_out_sample_two_stage.csv", index=False)
    # no_match_inference_df_from_x2.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
    #     hidden_dim) + "L_dim_from_hold_out_sample_two_stage.csv", index=False)
    #
    # no_match_inference_df_from_x1_Sim_cor.to_csv(
    #     saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
    #         hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)
    # no_match_inference_df_from_x2_Sim_cor.to_csv(
    #     saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
    #         hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)

    return AVG_MISMATCHES_X1_tr, AVG_MISMATCHES_X2_tr, np.average(AVG_MISMATCHES_X1_tr, axis=1), np.average(
        AVG_MISMATCHES_X2_tr,
        axis=1), Frac_mismatches_across_trial_perm_X1_tr, Frac_mismatches_across_trial_perm_X2_tr, \
           AVG_MISMATCHES_X1_tr_sim_Cor, AVG_MISMATCHES_X2_tr_sim_Cor, np.average(AVG_MISMATCHES_X1_tr_sim_Cor,
                                                                                  axis=1), np.average(
        AVG_MISMATCHES_X2_tr_sim_Cor,
        axis=1), Frac_mismatches_across_trial_perm_X1_tr_sim_Cor, Frac_mismatches_across_trial_perm_X2_tr_sim_Cor, \
           AVG_MISMATCHES_X1_tr_Kang, np.average(AVG_MISMATCHES_X1_tr_Kang,
                                                 axis=1), Frac_mismatches_across_trial_perm_X1_tr_Kang, AVG_MISMATCHES_X1_tr_RG, AVG_MISMATCHES_X2_tr_RG, np.average(
        AVG_MISMATCHES_X1_tr_RG, axis=1), np.average(AVG_MISMATCHES_X2_tr_RG,
                                                     axis=1), Frac_mismatches_across_trial_perm_X1_tr_RG, Frac_mismatches_across_trial_perm_X2_tr_RG



n_p = 3  # number of permutations
n_t = 4  # number of data partitioning trials
# list_of_number_samples = [500,1000]
list_of_number_samples = [500,1000,2000,5000,10000]

# data details
outcome = "Y"
dataset_number = 2 # 10 is for complex data
frac_renamed = 0.5
ordering_type = 15  # number denotes the number of variables swapped, 41 denotes symmetrically swapped and 42 denotes asymmetric swapping, 15 is randomly shuffling all of them
num_of_dataset_samples = 5
mpfeatures = 4  # number of mapped features
datatype = 'c'  # b denotes when the data needs to be binarized

# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization
weight_direct = 0.8
weight_cross = 1.1  # 0 denotes no cross loss, 1 denotes cross loss
weight_cycle = 0.8

alpha = 1.2  # used in KANG method

# model architecture and parameter details
hidden_dim = 5
num_of_hidden_layers = 2
batch_size = 64
epochs = 40
learning_rate = 1e-2
dropout_rate = 0.6


epochs_RadialGAN = 100
beta = 10  # grad penalty parameter
Lambda_dir = 10  # cycle consistency weight on dir
Lambda_on_z = 1  # cycle consistency weight on z

# file saving logistics
saving_dir = './ChimericAE_Final_'+str(datetime.date.today()) +  "/#ofhidden_layers_" + str(
    num_of_hidden_layers) + '/L_dim_' + str(hidden_dim) + '/Syn_' + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + "_Varying_sample_size_"+str(dropout_rate)+str(datetime.datetime.now())

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


AVG_over_Dataset_samples_X1_tr = np.zeros((len(list_of_number_samples), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr = np.zeros((len(list_of_number_samples), num_of_dataset_samples))

file_name = saving_dir + "/" + "Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name)

AVG_over_Dataset_samples_X1_tr_sim_Cor = np.zeros((len(list_of_number_samples), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_sim_Cor = np.zeros((len(list_of_number_samples), num_of_dataset_samples))

file_name_sim_Cor = saving_dir + "/" + "Simple_correlation_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_sim_Cor):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_sim_Cor)

AVG_over_Dataset_samples_X1_tr_KANG= np.zeros((len(list_of_number_samples), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_KANG = np.zeros((len(list_of_number_samples), num_of_dataset_samples))

file_name_KANG = saving_dir + "/" + "KANG_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_KANG):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_KANG)

AVG_over_Dataset_samples_X1_tr_RG = np.zeros((len(list_of_number_samples), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_RG = np.zeros((len(list_of_number_samples), num_of_dataset_samples))

file_name_RG = saving_dir + "/" + "RG_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_RG):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_RG)


f = open(file_name, 'w')
f.write("\n \n ***Two stage Chimeric AE Present file settings ***")
f.write("\n \n Dataset number {0}\t".format(dataset_number))
f.write("\n Orthogonalization status {0}\t ".format(orthogonalization_type))
f.write("\n Batch norm {0}\t ".format(batchnorm))
f.write("\n Size of L {0}\t".format(hidden_dim))
f.write("\n Weight for direct AE loss {0}\t ".format(weight_direct))
f.write("\n Weight for cross AE loss {0}\t ".format(weight_cross))
f.write("\n Weight for cycle AE loss {0}\t ".format(weight_cycle))
f.write("\n Number of epochs {0}\t".format(epochs))
f.write("\n Starting learning rate {0}\t ".format(learning_rate))
f.write("\n Batch size {0}\t".format(batch_size))
f.write("\n")
f.close()

f = open(file_name_RG, 'w')
f.write("\n \n *** RadialGAN Present file settings ***")
f.write("\n \n Dataset number {0}\t".format(dataset_number))
f.write("\n Grad penalty parameter {0}\t".format(beta))
f.write("\n Cycle consistency weight on dir {0}\t ".format(Lambda_dir))
f.write("\n Cycle consistency weight on z {0}\t ".format(Lambda_on_z))
f.write("\n Number of epochs {0}\t".format(epochs_RadialGAN))
f.write("\n Starting learning rate {0}\t ".format(learning_rate))
f.write("\n Batch size {0}\t".format(batch_size))
f.write("\n")
f.close()

f = open(file_name_sim_Cor, 'w')
f.write("\n \n *** Simple_correlation Present file settings ***")
f.write("\n \n Dataset number {0}\t".format(dataset_number))
f.close()

f = open(file_name_KANG, 'w')
f.write("\n \n *** KANG Present file settings ***")
f.write("\n \n Dataset number {0}\t".format(dataset_number))
f.write("\n alpha parameter {0}\t".format(alpha))
f.close()

print("\n ********************************************************")
print("Number of mapped features for this experiment are ", mpfeatures)
print(" ******************************************************** \n")

for sample_no in range(1,num_of_dataset_samples+1):
    print("\n ********************************************************")
    print(" \n Run STARTS for sample no ", sample_no, "  of dataset ", dataset_number, "\n")
    print(" ******************************************************** \n")

    AVG_MISMATCHES_X1_tr, AVG_MISMATCHES_X2_tr, m_x1, m_x2_tr, Frac_X1_tr, Frac_X2_tr, AVG_MISMATCHES_X1_tr_sim_Cor, AVG_MISMATCHES_X2_tr_sim_Cor, m_x1_sim_Cor, m_x2_tr_sim_Cor, Frac_X1_tr_sim_Cor, Frac_X2_tr_sim_Cor, AVG_MISMATCHES_X1_tr_KANG, m_X1_tr_KANG, Frac_X1_tr_KANG, AVG_MISMATCHES_X1_tr_RG, AVG_MISMATCHES_X2_tr_RG, m_x1_RG, m_x2_tr_RG, Frac_X1_tr_RG, Frac_X2_tr_RG  = main(sample_no)

    # for Chimeric AE

    AVG_over_Dataset_samples_X1_tr[:,sample_no-1] = m_x1
    AVG_over_Dataset_samples_X2_tr[:,sample_no-1] = m_x2_tr

    f = open(file_name,'a')
    f.write("\n \n Frac of Mismatches for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X1_tr))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X2_tr))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        Frac_x1_tr_list = Frac_X1_tr
        Frac_x2_tr_list = Frac_X2_tr
    else:
        Frac_x1_tr_list = np.hstack((Frac_x1_tr_list, Frac_X1_tr))
        Frac_x2_tr_list = np.hstack((Frac_x2_tr_list, Frac_X2_tr))

    # for Simple_correlation (Simple_correlation)

    AVG_over_Dataset_samples_X1_tr_sim_Cor[:,sample_no-1] = m_x1_sim_Cor
    AVG_over_Dataset_samples_X2_tr_sim_Cor[:,sample_no-1] = m_x2_tr_sim_Cor

    f = open(file_name_sim_Cor,'a')
    f.write("\n \n Frac of Mismatches for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X1_tr_sim_Cor))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X2_tr_sim_Cor))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        Frac_x1_tr_list_sim_Cor = Frac_X1_tr_sim_Cor
        Frac_x2_tr_list_sim_Cor = Frac_X2_tr_sim_Cor
    else:
        Frac_x1_tr_list_sim_Cor = np.hstack((Frac_x1_tr_list_sim_Cor, Frac_X1_tr_sim_Cor))
        Frac_x2_tr_list_sim_Cor = np.hstack((Frac_x2_tr_list_sim_Cor, Frac_X2_tr_sim_Cor))

    # KANG method

    AVG_over_Dataset_samples_X1_tr_KANG[:, sample_no - 1] = m_X1_tr_KANG
    # AVG_over_Dataset_samples_X2_tr_KANG[:, sample_no - 1] = m_x2_tr_KANG

    f = open(file_name_KANG, 'a')
    f.write("\n \n Frac of Mismatches for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X1_tr_KANG))
    # f.write("\n X2_train \n")
    # f.write("{0}".format(AVG_MISMATCHES_X2_tr_KANG))
    f.write("\n \n ")
    f.close()

    if sample_no == 1:
        Frac_x1_tr_list_KANG = Frac_X1_tr_KANG
        # Frac_x2_tr_list_KANG = Frac_X2_tr_KANG
    else:
        Frac_x1_tr_list_KANG = np.hstack((Frac_x1_tr_list_KANG, Frac_X1_tr_KANG))
        # Frac_x2_tr_list_KANG = np.hstack((Frac_x2_tr_list_KANG, Frac_X2_tr_KANG))


    # for RadialGAN (RG)

    AVG_over_Dataset_samples_X1_tr_RG[:,sample_no-1] = m_x1_RG
    AVG_over_Dataset_samples_X2_tr_RG[:,sample_no-1] = m_x2_tr_RG

    f = open(file_name_RG,'a')
    f.write("\n \n Frac of Mismatches for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X1_tr_RG))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_MISMATCHES_X2_tr_RG))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        Frac_x1_tr_list_RG = Frac_X1_tr_RG
        Frac_x2_tr_list_RG = Frac_X2_tr_RG
    else:
        Frac_x1_tr_list_RG = np.hstack((Frac_x1_tr_list_RG, Frac_X1_tr_RG))
        Frac_x2_tr_list_RG = np.hstack((Frac_x2_tr_list_RG, Frac_X2_tr_RG))

    print("\n ********************************************************")
    print(" \n Run ENDS for sample no ", sample_no, "  of dataset ", dataset_number, "\n ")
    print(" ******************************************************** \n")



file_name_violin = saving_dir + "/" + "For_violin_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_violin):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_violin)

f = open(file_name_violin,'a')
f.write("\n \n List of mapped features \n ")
f.write("{0}".format(list_of_number_samples))
f.write("\n \n Frac for across trials and perm ")
f.write("\n X1_train \n")
f.write("{0}".format(Frac_x1_tr_list))
f.write("\n X2_train \n")
f.write("{0}".format(Frac_x2_tr_list))
f.write("\n X1_train_sim_Cor \n")
f.write("{0}".format(Frac_x1_tr_list_sim_Cor))
f.write("\n X2_train_sim_Cor \n")
f.write("{0}".format(Frac_x2_tr_list_sim_Cor))
f.write("\n X1_train_KANG \n")
f.write("{0}".format(Frac_x1_tr_list_KANG))
f.write("\n X1_train_RG \n")
f.write("{0}".format(Frac_x1_tr_list_RG))
f.write("\n X2_train_RG \n")
f.write("{0}".format(Frac_x2_tr_list_RG))
f.write("\n \n ")
f.close()

# Computing the average over the datset samples
Mean_over_trials_mismatches_X1_tr = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X1_tr = np.zeros(len(list_of_number_samples))
Mean_over_trials_mismatches_X2_tr = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X2_tr = np.zeros(len(list_of_number_samples))

Mean_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_number_samples))
Mean_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_number_samples))

Mean_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_number_samples))

Mean_over_trials_mismatches_X1_tr_RG = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X1_tr_RG = np.zeros(len(list_of_number_samples))
Mean_over_trials_mismatches_X2_tr_RG = np.zeros(len(list_of_number_samples))
SD_over_trials_mismatches_X2_tr_RG = np.zeros(len(list_of_number_samples))

x_axis = np.arange(len(list_of_number_samples))
x_axis1 = x_axis + 0.05
x_axis2 = x_axis + 0.1

for i in range(len(list_of_number_samples)):
    Mean_over_trials_mismatches_X1_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr[i] = np.round(np.std(Frac_x1_tr_list[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr[i] = np.round(np.std(Frac_x2_tr_list[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_sim_Cor[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_sim_Cor[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.std(Frac_x1_tr_list_sim_Cor[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.std(Frac_x2_tr_list_sim_Cor[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_KANG[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.std(Frac_x1_tr_list_KANG[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_RG[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_RG[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_RG[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_RG[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_RG[i] = np.round(np.std(Frac_x1_tr_list_RG[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_RG[i] = np.round(np.std(Frac_x2_tr_list_RG[i, :]), decimals=4)


plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr, color='blue', label=" 2stage Chimeric AE ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr, yerr=SD_over_trials_mismatches_X1_tr, linestyle="solid",
             color='blue')
plt.scatter(x_axis1, Mean_over_trials_mismatches_X1_tr_sim_Cor, color='red', label=" Simple_correlation ", linestyle='None')
plt.errorbar(x_axis1, Mean_over_trials_mismatches_X1_tr_sim_Cor, yerr=SD_over_trials_mismatches_X1_tr_sim_Cor, linestyle="solid",
             color='red')
plt.scatter(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, color='brown', label=" KANG ", linestyle='None')
plt.errorbar(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, yerr=SD_over_trials_mismatches_X1_tr_KANG, linestyle="solid",
             color='brown')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_RG, color='green', label=" RadialGAN ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_RG, yerr=SD_over_trials_mismatches_X1_tr_RG, linestyle="solid",
             color='green')
plt.xticks(x_axis, np.array(list_of_number_samples))
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Total sample size")
plt.ylabel("Fraction of mistakes across different dataset samples")
plt.title("Dataset no. " + str(dataset_number) + " Results from X1 cross correlation matrix ")
plt.legend()
plt.savefig(saving_dir + "/Comp_ChimVsothers_X1_tr_Syn_" + str(dataset_number) + "_varying_sample_size_" + str(
    len(list_of_number_samples)) + ".pdf", bbox='tight')
plt.savefig(saving_dir + "/Comp_ChimVsothers_X1_tr_Syn_" + str(dataset_number) + "_varying_sample_size_" + str(
    len(list_of_number_samples)) + ".png", bbox='tight')
plt.close()


plt.scatter(x_axis, Mean_over_trials_mismatches_X2_tr, color='blue', label="2stage Chimeric AE ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X2_tr, yerr=SD_over_trials_mismatches_X2_tr, linestyle="solid",
             color='blue')
plt.scatter(x_axis1, Mean_over_trials_mismatches_X2_tr_sim_Cor, color='red', label=" Simple_correlation ", linestyle='None')
plt.errorbar(x_axis1, Mean_over_trials_mismatches_X2_tr_sim_Cor, yerr=SD_over_trials_mismatches_X2_tr_sim_Cor, linestyle="solid",
             color='red')
plt.scatter(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, color='brown', label=" KANG ", linestyle='None')
plt.errorbar(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, yerr=SD_over_trials_mismatches_X1_tr_KANG, linestyle="solid",
             color='brown')
plt.scatter(x_axis, Mean_over_trials_mismatches_X2_tr_RG, color='green', label=" RadialGAN ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X2_tr_RG, yerr=SD_over_trials_mismatches_X2_tr_RG, linestyle="solid",
             color='green')
plt.xticks(x_axis, np.array(list_of_number_samples))
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Total sample size")
plt.ylabel("Fraction of  mistakes across different dataset samples")
plt.title("Dataset no. " + str(dataset_number) + " Results from X2 cross correlation matrix ")
plt.legend()
plt.savefig(saving_dir + "/Comp_ChimVsothers_X2_tr_Syn_" + str(dataset_number) + "_varying_sample_size_" + str(
    len(list_of_number_samples)) + ".pdf", bbox='tight')
plt.savefig(saving_dir + "/Comp_ChimVsothers_X2_tr_Syn_" + str(dataset_number) + "_varying_sample_size_" + str(
    len(list_of_number_samples)) + ".png", bbox='tight')
plt.close()
