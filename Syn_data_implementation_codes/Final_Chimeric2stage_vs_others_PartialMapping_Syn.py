"""
this file provides a pipeline of feature mapping problem. The components are:
1) Train an Autoencoder and compute the correlation matrix between the true feature values and the reconstructed features from cross AEs.
The model being used has no batch normalization and has simple orthogonalization ( in individual AEs).
2) Using hospital resident matching algorithm (with capacity 1) on the correlation matrix from 1 and get the final mappings.

The two methods that are being compared are two stage Chimeric AE approach and Simple correlation approach.


INPUT:

Full dataset, the model details, number of permutations, number of partitioning of dataset, fraction of data to be permuted, number of mapped features

OUTPUT:

An array with rows as the number of mapped features and columns as the trial number and cell entry as the avg mismatches for the a fixed trial and the fixed number of mapped variables


This code has randomness over mapped features and unmapped features too
In this code, at the step where mapped features are being sampled, we also decide the features that need to be dropped from the one of the datasets (dataset 2) so the number of features in two datasets are different.
This is with the motivation that a lot of features are redundant if the underlying factors are very small.

This code has multiple (10) sampled datasets from the same distribution

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
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise, mutual_info_score
from sklearn.decomposition import PCA
from scipy import linalg, stats
import pickle
import datetime
import os.path
import math
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from collections import Counter
import random
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from matching.games import HospitalResident, StableMarriage
import statsmodels
import pingouin as pg

def Matching_via_HRM(C_X1_train, C_X2_train, P_x1_O_to_R, num_mapped_axis):  # in this case here the small feature sized database is X1, so we need to treat it as hospital and there will be capacities on it.
    # creating the preference dictionaries
    ####### ----------  X1 train ------------- ##########

    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}
    capacities_X1_train = {}

    for i in range(C_X1_train.shape[0]):
        sorted_index = np.argsort(-C_X1_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index
        capacities_X1_train["R" + str(i + 1)] = 1

    for j in range(C_X1_train.shape[1]):
        sorted_index = np.argsort(-C_X1_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X1_train)
    # print(cross_recon_features_pref_X1_train)
    # print(capacities_X1_train)
    game_X1_train = HospitalResident.create_from_dictionaries(cross_recon_features_pref_X1_train,true_features_pref_X1_train,
                                                              capacities_X1_train)

    ####### ----------  X2 train ------------- ##########

    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}
    capacities_X2_train = {}

    for i in range(C_X2_train.shape[0]):
        sorted_index = np.argsort(-C_X2_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):
        sorted_index = np.argsort(-C_X2_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index
        capacities_X2_train["C" + str(j + 1)] = 1


    # print(true_features_pref_X2_train)
    # print(cross_recon_features_pref_X2_train)
    # print(capacities_X2_train)

    game_X2_train = HospitalResident.create_from_dictionaries(true_features_pref_X2_train,cross_recon_features_pref_X2_train,
                                                               capacities_X2_train)


    ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)


    x1_train_y = np.array([int(str(v[0]).split("C")[1])  for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v[0]).split("R")[1])  for v in matching_x2_train.values()])


    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(np.transpose(C_X2_train).shape)

    for i in range(matching_x1_train_matrix.shape[0]):
        # print(i, x1_train_y[i]-1)
        matching_x1_train_matrix[i,x1_train_y[i]-1]=1


    for i in range(matching_x2_train_matrix.shape[0]):
        # print(i, x2_train_y[i]-1)
        matching_x2_train_matrix[i,x2_train_y[i]-1]=1

    # getting the number of correct matches that had a match in other database
    num_correct_from_x1 = 0
    num_correct_from_x2 = 0
    for i in range(P_x1_O_to_R.shape[0]):
        if np.all(P_x1_O_to_R[i] == matching_x1_train_matrix[i]):
            num_correct_from_x1 = num_correct_from_x1 + 1
        if np.all(P_x1_O_to_R[i] == matching_x2_train_matrix[i]):
            num_correct_from_x2 = num_correct_from_x2 + 1

    return num_correct_from_x1, num_correct_from_x2, matching_x1_train_matrix, matching_x2_train_matrix


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
            activation = torch.tanh(activation)
            activation = self.drop_layer2(activation)
            code0 = self.encoder_output_layer(activation)
            if self.batchnorm == 1:
                code0 = self.bn3(code0)
            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn4(activation)
            activation = torch.tanh(activation)
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
            activation = torch.tanh(activation)
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
def Kang_MI_HC_opt(df_train_preproc, df_rename_preproc, P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
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
            for j in range(i+1, num_NonCat_features_orig):
                MI_orig[i,j] = calc_MI(df_train_preproc.iloc[:,i], df_train_preproc.iloc[:,j], 20)
                MI_orig[j,i] = MI_orig[i,j]

        MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
        # MI computation
        for i in range(num_NonCat_features_r):
            for j in range(i+1, num_NonCat_features_r):
                MI_r[i,j] = calc_MI(df_rename_preproc.iloc[:,i], df_rename_preproc.iloc[:,j], 20)
                MI_r[j,i] = MI_r[i,j]

    num_iter = 1000
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
                temp0 = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]]) / (
                        MI_orig[i, j] + MI_r[temp_per[i], temp_per[j]]))
                if np.isnan(temp0) != True:
                    temp_dist_normal = temp_dist_normal + temp0
                    # print(temp0)

        # updating the cost and the permutation vector
        if temp_dist_normal > D_M_normal:
            # print(" Iteration nnumber where it changed", iter )
            # print("initial cost ", D_M_normal)
            # print("initial permutation", initial_perm)

            D_M_normal = temp_dist_normal
            initial_perm = temp_per

    #         print(" Updated cost ", D_M_normal)
    #         print(" updated permutation ", temp_per)
    # print("Blah")


    true_permutation = [np.where(P_x1[a,:]==1)[0]for a in range(len(P_x1))]
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_permutation[i]==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1

    print(" \n Mistakes by the KANG method on training data")

    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    print(" -------- KANG  methods  ends ------------- \n \n  ")

    del DF_holdout_r

    x1_match_matrix_test = np.zeros(P_x1.shape)
    for i in range(x1_match_matrix_test.shape[0]):
        # print(i, x1_train_y[i]-1)
        x1_match_matrix_test[i, initial_perm[i]] = 1

    P_x1_updated = P_x1[mpfeatures:,mpfeatures:]
    x1_match_matrix_test_updated = x1_match_matrix_test[mpfeatures:,mpfeatures:]
    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1_updated.shape[0]):
        for j in range(P_x1_updated.shape[1]):
            if (P_x1_updated[i, j] == 1) & (x1_match_matrix_test_updated[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1_updated[i, j] == 1) & (x1_match_matrix_test_updated[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1_updated[i, j] == 0) & (x1_match_matrix_test_updated[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1_updated[i, j] == 0) & (x1_match_matrix_test_updated[i, j] == 1):
                FP_x1 = FP_x1 + 1
    #
    # TP_x2 = 0
    # FP_x2 = 0
    # TN_x2 = 0
    # FN_x2 = 0
    # for i in range(P_x1.shape[0]):
    #     for j in range(P_x1.shape[1]):
    #         if (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
    #             TP_x2 = TP_x2 + 1
    #         elif (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
    #             FN_x2 = FN_x2 + 1
    #         elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
    #             TN_x2 = TN_x2 + 1
    #         elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
    #             FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    # F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    return correct_total_fromKANG, F1_fromx1


# MI based method that first creates a graph and then looks for a permutation matrix that minimizes the distance between the adjcency matrices of the two graphs
# def Kang_MI_HC_opt(df_train_preproc, df_rename_preproc, true_perm
#                    , reordered_column_names_orig, reordered_column_names_r,
#                    mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
#     mpfeatures = len(mapped_features)
#     unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
#     unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1
#     device = torch.device('cuda')
#     num_NonCat_features_orig = len(reordered_column_names_orig) - 1
#
#     num_features = len(reordered_column_names_r) - 1
#     num_NonCat_features_r = len(reordered_column_names_r) - 1
#
#
#     if datatype == 'b':
#         MI_orig = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
#         # MI computation
#         for i in range(num_NonCat_features_orig):
#             p1_i = sum(df_train_preproc.iloc[:, i]) / df_train_preproc.shape[0]
#             p0_i = 1 - p1_i
#             MI_orig[i, i] = -p0_i * np.log(p0_i) - p1_i * np.log(p1_i)
#             for j in range(i + 1, num_NonCat_features_orig):
#                 p1_j = sum(df_train_preproc.iloc[:, j]) / df_train_preproc.shape[0]
#                 p0_j = 1 - p1_j
#                 p11 = sum(df_train_preproc.iloc[:, i] * df_train_preproc.iloc[:, j]) / df_train_preproc.shape[0]
#                 p00 = sum(np.where((df_train_preproc.iloc[:, i] == 0) & (df_train_preproc.iloc[:, j] == 0), 1, 0)) / \
#                       df_train_preproc.shape[0]
#                 p01 = sum(np.where((df_train_preproc.iloc[:, i] == 0) & (df_train_preproc.iloc[:, j] == 1), 1, 0)) / \
#                       df_train_preproc.shape[0]
#                 p10 = sum(np.where((df_train_preproc.iloc[:, i] == 1) & (df_train_preproc.iloc[:, j] == 0), 1, 0)) / \
#                       df_train_preproc.shape[0]
#
#                 MI_orig[i, j] = p00 * np.log(p00 / (p0_i * p0_j)) + p01 * np.log(p01 / (p0_i * p1_j)) + p10 * np.log(
#                     p10 / (p1_i * p0_j)) + p11 * np.log(p11 / (p1_i * p1_j))
#                 MI_orig[j, i] = MI_orig[i, j]
#
#         MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
#         # MI computation
#         for i in range(num_NonCat_features_r):
#             p1_i = sum(df_rename_preproc.iloc[:, i]) / df_rename_preproc.shape[0]
#             p0_i = 1 - p1_i
#             MI_r[i, i] = -p0_i * np.log(p0_i) - p1_i * np.log(p1_i)
#             for j in range(i + 1, num_NonCat_features_r):
#                 p1_j = sum(df_rename_preproc.iloc[:, j]) / df_rename_preproc.shape[0]
#                 p0_j = 1 - p1_j
#                 p11 = sum(df_rename_preproc.iloc[:, i] * df_rename_preproc.iloc[:, j]) / df_rename_preproc.shape[0]
#                 p00 = sum(np.where((df_rename_preproc.iloc[:, i] == 0) & (df_rename_preproc.iloc[:, j] == 0), 1, 0)) / \
#                       df_rename_preproc.shape[0]
#                 p01 = sum(np.where((df_rename_preproc.iloc[:, i] == 0) & (df_rename_preproc.iloc[:, j] == 1), 1, 0)) / \
#                       df_rename_preproc.shape[0]
#                 p10 = sum(np.where((df_rename_preproc.iloc[:, i] == 1) & (df_rename_preproc.iloc[:, j] == 0), 1, 0)) / \
#                       df_rename_preproc.shape[0]
#
#                 MI_r[i, j] = p00 * np.log(p00 / (p0_i * p0_j)) + p01 * np.log(p01 / (p0_i * p1_j)) + p10 * np.log(
#                     p10 / (p1_i * p0_j)) + p11 * np.log(p11 / (p1_i * p1_j))
#                 MI_r[j, i] = MI_r[i, j]
#
#     else:
#         MI_orig = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
#         # MI computation
#         for i in range(num_NonCat_features_orig):
#             for j in range(i+1, num_NonCat_features_orig):
#                 MI_orig[i,j] = calc_MI(df_train_preproc.iloc[:,i], df_train_preproc.iloc[:,j], 20)
#                 MI_orig[j,i] = MI_orig[i,j]
#
#         MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
#         # MI computation
#         for i in range(num_NonCat_features_r):
#             for j in range(i+1, num_NonCat_features_r):
#                 MI_r[i,j] = calc_MI(df_rename_preproc.iloc[:,i], df_rename_preproc.iloc[:,j], 20)
#                 MI_r[j,i] = MI_r[i,j]
#
#     num_iter = 1000
#     # initial random permutation and corresponding distance computation
#     initial_perm = np.array(list(np.arange(mpfeatures)) + list(np.random.choice( np.arange(unmapped_features_r), unmapped_features_orig, replace =False)+mpfeatures))
#
#     D_M_normal_ini = 0
#
#     for i in range(len(MI_orig)):
#         for j in range(len(MI_orig)):
#             temp = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) / (
#                         MI_orig[i, j] + MI_r[initial_perm[i], initial_perm[j]]))
#             if np.isnan(temp) != True :
#                 D_M_normal_ini = D_M_normal_ini + temp
#                 # print(temp)
#     print("Distance_initial_value", D_M_normal_ini)
#
#     D_M_normal = D_M_normal_ini
#     for iter in range(num_iter):
#         temp_per = initial_perm.copy()
#         idx_unmp_feature = np.random.choice(np.arange(unmapped_features_orig),1) + mpfeatures  # random index chosen for swapping the value
#         unm_feat_match_random = np.random.choice(np.arange(unmapped_features_r),1)+mpfeatures
#         # print(unm_feat_match_random)
#         if unm_feat_match_random in temp_per:
#             idx_for_already_existing_match = np.where(temp_per==unm_feat_match_random)
#             temp_to_swap_value = temp_per[idx_unmp_feature]
#             temp_per[idx_unmp_feature] =  unm_feat_match_random
#             temp_per[idx_for_already_existing_match] = temp_to_swap_value
#         else:
#             temp_per[idx_unmp_feature] =  unm_feat_match_random
#
#         # checking the cost
#         temp_dist_normal = 0
#         for i in range(len(MI_orig)):
#             for j in range(len(MI_orig)):
#                 temp0 = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]]) / (
#                         MI_orig[i, j] + MI_r[temp_per[i], temp_per[j]]))
#                 if np.isnan(temp0) != True:
#                     temp_dist_normal = temp_dist_normal + temp0
#                     # print(temp0)
#
#         # updating the quantity to optimize and the permutation vector
#         if temp_dist_normal > D_M_normal:
#             D_M_normal = temp_dist_normal
#             initial_perm = temp_per
#
#
#     # true_permutation = list(np.arange(mpfeatures)) +  [np.where(P_x1[a,:]==1)[0] + mpfeatures for a in range(len(P_x1))]
#     mistake_indices = []
#     correct_total_fromKANG = 0
#     for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
#         # print(i)
#         if true_perm[i]-1==initial_perm[i]:
#             correct_total_fromKANG = correct_total_fromKANG + 1
#         else:
#             mistake_indices.append(true_perm[i])
#
#
#     print(" \n Mistakes by the KANG method on holdout data")
#
#     print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")
#
#     MisF_X1_te = [reordered_column_names_r[i - 1] for i in mistake_indices]
#
#     print(" KANG method  X1 mistakes", MisF_X1_te)
#
#     # print(" KANG method mistakes on sq transformation X1 ", list(set(sq_transf_features).intersection(MisF_X1_te)))
#
#     print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)
#
#     print(" -------- KANG  methods  ends ------------- \n \n  ")
#
#     del DF_holdout_r
#
#     return MisF_X1_te


def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc, P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

    device = torch.device('cuda')
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

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,
                                                      P_x1, len(mapped_features))


    test_statistic_num_fromX1 = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                     j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    test_statistic_num_fromX2 = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                     j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                         i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small





    # Bootstrap samples to obtain the standard deviation methods to be later used in p value computation
    num_of_bts = 10
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
        sim_cor_X1_to_X2_bts = np.matmul(CorMatrix_X1_unmap_mapped_bts, np.transpose(CorMatrix_X2_unmap_mapped_bts))
        sim_cor_X2_to_X1_bts = np.matmul(CorMatrix_X2_unmap_mapped_bts, np.transpose(CorMatrix_X1_unmap_mapped_bts))

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

    # testing whether some of the proposed matches are such that there exist no match in reality but GS assigned one;
    # False in the reject list below can be interpreted as the case where the  testing procedure says there wasn't any match originally
    temp_inf_x1.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x1.corr_p_value), method='fdr_by', alpha=0.05)
    temp_inf_x2.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x2.corr_p_value), method='fdr_by', alpha=0.05)


    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]
        if np.all(P_x1[i] == 0):
            temp_inf_x1.loc[i,"no_match_or_not"] = 1
        else:
            temp_inf_x1.loc[i,"no_match_or_not"] = 0


    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]
        if np.all(P_x1[i] == 0):
            temp_inf_x2.loc[i,"no_match_or_not"] = 1
        else:
            temp_inf_x2.loc[i,"no_match_or_not"] = 0

    correct_with_no_match_from_CCx1_test = 0
    correct_with_no_match_from_CCx2_test = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False and np.all(P_x1[i] == 0):
            correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
        if temp_inf_x2.SD_rejects_H0[i] == False and np.all(P_x1[i] == 0):
            correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the simple correlation method on holdout data")
    print(" Sim_Correlation  X1_train mistakes number", unmapped_features_orig-correct_with_match_from_x1_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number", unmapped_features_orig-correct_with_match_from_x2_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    print("Mistakes by the significance testing algorithm on holdout data")
    print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)

    # preparing for the second stage

    Copy_temp_inf_x1 = temp_inf_x1.copy()
    Copy_temp_inf_x2 = temp_inf_x2.copy()

    # dropping non significant matched feature pairs
    num_insign_dropp_x1 = 0
    num_insign_dropp_x2 = 0
    for i in range(len(Copy_temp_inf_x1.SD_rejects_H0)):
        if Copy_temp_inf_x1.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x1 ", Copy_temp_inf_x1.ump_feature_in_X1[i], Copy_temp_inf_x1.match_byGS[i], Copy_temp_inf_x1.estimated_cross_corr[i])
            Copy_temp_inf_x1.drop([i], inplace=True)
            num_insign_dropp_x1 = num_insign_dropp_x1 +1

        if Copy_temp_inf_x2.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x2 ", Copy_temp_inf_x2.ump_feature_in_X1[i], Copy_temp_inf_x2.match_byGS[i], Copy_temp_inf_x2.estimated_cross_corr[i])
            Copy_temp_inf_x2.drop([i], inplace=True)
            num_insign_dropp_x2 = num_insign_dropp_x2 + 1

    print(" Number of insignificant feature pair drops from x1 ", num_insign_dropp_x1)
    print(" Number of insignificant feature pair drops from x2 ", num_insign_dropp_x2)

    # ordering the
    Copy_temp_inf_x1 = Copy_temp_inf_x1.sort_values(by='estimated_cross_corr', ascending=False)
    Copy_temp_inf_x2 = Copy_temp_inf_x2.sort_values(by='estimated_cross_corr', ascending=False)

    num_additional_mapped_for_next_stage_x1 = int(len(Copy_temp_inf_x1)/2)
    num_additional_mapped_for_next_stage_x2 = int(len(Copy_temp_inf_x2)/2)

    # taking the intersection of the additional mapped features
    temp_x1_x1 = [Copy_temp_inf_x1.ump_feature_in_X1[i] for i in list(Copy_temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x1_match = [Copy_temp_inf_x1.match_byGS[i] for i in list(Copy_temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x2_x1 = [Copy_temp_inf_x2.ump_feature_in_X1[i] for i in list(Copy_temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]
    temp_x2_match = [Copy_temp_inf_x2.match_byGS[i] for i in list(Copy_temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]



    final_additional_mapped = list(set(temp_x1_x1).intersection(temp_x2_x1))
    final_additional_mapped_corr_match =[]
    for i in final_additional_mapped:
        final_additional_mapped_corr_match.append(temp_x1_match[temp_x1_x1.index(i)])

    # print(" \n Mistakes by the simple correlation method on holdout data")
    #
    # print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")
    #
    # MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    # MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]
    #
    # print(" KMF  X1 mistakes", MisF_X1_te)
    # print(" KMF  X2 mistakes", MisF_X2_te)
    #
    # print(" KMF mistakes on sq transformation X1 ", list(set(sq_transf_features).intersection(MisF_X1_te)))
    # print(" KMF mistakes on sq transformation X2 ", list(set(sq_transf_features).intersection(MisF_X2_te)))
    #
    # print(" Sim_Correlation  X1_train mistakes number", len(Mistakes_X1_te), "out of ", unmapped_features_orig)
    # print(" Sim_Correlation  X2_train mistakes number", len(Mistakes_X2_te), "out of ", unmapped_features_r)

    print(" -------- Sim_Correlation  methods  ends ------------- \n \n  ")

    del df_rename_preproc

    # Computation of F1 scores

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
                FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    return correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, final_additional_mapped, final_additional_mapped_corr_match, F1_fromx1, F1_fromx2

def Train_cross_AE(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, mapped_features_updated_orig, mapped_features_updated_r, Cor_from_df,Df_holdout_orig, DF_holdout_r,unmapped_features_extra_orig,filename_for_saving_PM_quality,normalizing_values_orig,normalizing_values_r, DF_holdout_orig0_not_includedwhiletraining):
    mpfeatures = len(mapped_features)
    device = torch.device('cpu')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1


    num_features = len(reordered_column_names_r) -1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

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



    ####################  Whole of above evaluation and analysis on holdout samples *********************************
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

    short_CorMatrix_X1_X1_hat_cross_P_value_test = CorMatrix_X1_X1_hat_cross_P_value_test[len(mapped_features):,
                                                   len(mapped_features):]
    short_CorMatrix_X2_X2_hat_cross_P_value_test = CorMatrix_X2_X2_hat_cross_P_value_test[len(mapped_features):,
                                                   len(mapped_features):]

    """ Calling the stable marriage algorithm for mappings  """

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
        short_CorMatrix_X1_X1_hat_cross_test,
        short_CorMatrix_X2_X2_hat_cross_test,
        P_x1, len(mapped_features))

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value',
                 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value',
                 'SD_rejects_H0', 'no_match_or_not'])

    # getting the p values that needs to be tested for significance

    temp_inf_x1.corr_p_value = [short_CorMatrix_X1_X1_hat_cross_P_value_test[i, j] for i in
                                range(x1_match_matrix_test.shape[0]) for j in range(x1_match_matrix_test.shape[1]) if
                                x1_match_matrix_test[i, j] == 1]
    temp_inf_x2.corr_p_value = [short_CorMatrix_X2_X2_hat_cross_P_value_test[j, i] for i in
                                range(x2_match_matrix_test.shape[0]) for j in range(x2_match_matrix_test.shape[1]) if
                                x2_match_matrix_test[
                                    i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small

    temp_inf_x1.estimated_cross_corr = [short_CorMatrix_X1_X1_hat_cross_test[i, j] for i in
                                        range(x1_match_matrix_test.shape[0]) for
                                        j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    temp_inf_x2.estimated_cross_corr = [short_CorMatrix_X2_X2_hat_cross_test[j, i] for i in
                                        range(x2_match_matrix_test.shape[0]) for
                                        j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                            i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small


    # testing whether some of the proposed matches are such that there exist no match in reality but GS assigned one;
    # False in the reject list below can be interpreted as the case where the  testing procedure says there wasn't any match originally
    temp_inf_x1.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x1.corr_p_value), method='fdr_by', alpha=0.05)
    temp_inf_x2.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x2.corr_p_value), method='fdr_by', alpha=0.05)

    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.all(P_x1[i] == 0):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0

    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.all(P_x1[i] == 0):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0

    correct_with_no_match_from_CCx1_test = 0
    correct_with_no_match_from_CCx2_test = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False and np.all(P_x1[i] == 0):
            correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
        if temp_inf_x2.SD_rejects_H0[i] == False and np.all(P_x1[i] == 0):
            correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the 2stage Chimeric method on holdout data")
    print(" Chimeric  X1_train mistakes number on holdout set", unmapped_features_orig-correct_with_match_from_x1_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" Chimeric  X2_train mistakes number on holdout set", unmapped_features_orig-correct_with_match_from_x2_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    print("Mistakes by the significance testing algorithm on holdout data")
    print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)


    print(" -------- Chimeric AE method training ends ------------- \n \n  ")

    print("Mapped feature_set after KMF ")
    print(" X1 ", mapped_features_updated_orig)
    print(" X2 ", mapped_features_updated_r)

    # plot for partial mapping evaluation
    features_true_orig_test_not_inorig = DF_holdout_orig0_not_includedwhiletraining.values
    row_index_no_orig = np.random.choice(len(features_true_orig_test_not_inorig), 500, replace=False)
    for feature_name in unmapped_features_extra_orig:
        print('\n ---------- feature name --------- \n ', feature_name)
        col_idxno_orig = list(DF_holdout_orig0_not_includedwhiletraining.columns).index(feature_name)
        col_idxno_renamed = reordered_column_names_r.index(feature_name)
        denormalizing_orig = features_true_orig_test_not_inorig[row_index_no_orig, col_idxno_orig] * \
                             normalizing_values_orig['std'].loc[feature_name] + normalizing_values_orig['mean'].loc[
                                 feature_name]
        x_axis = (denormalizing_orig - normalizing_values_r['mean'].loc[feature_name]) / \
                 normalizing_values_r['std'].loc[feature_name]
        y_axis = features_reconst_from_crossR_test[row_index_no_orig, col_idxno_renamed]

        plt.scatter(x_axis, y_axis, color='blue')
        plt.xlabel("true feature value (not present in x1 while training)")
        plt.ylabel("reconstructed feature value from x2")
        temp = stats.pearsonr(x_axis,y_axis)[0]
        plt.figtext(0.6,0.8,"Cor_value = "+str(np.round(temp, decimals=3)))
        plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
        plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
        plt.title("Dataset no. " + str(dataset_number) + " when  number of mapped features are  " + str(
            mpfeatures) + "\n & " + str(feature_name) + " not in original data", fontsize=8)
        if feature_name in mapped_features_updated_orig:
            plt.savefig(
                filename_for_saving_PM_quality + "_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
            plt.savefig(
                filename_for_saving_PM_quality + "_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
        else:
            plt.savefig(
                filename_for_saving_PM_quality + "_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
            plt.savefig(
                filename_for_saving_PM_quality + "_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
        plt.close()

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
    #             filename_for_saving_PM_quality + "_Mappedafter_KMF" + "_Direct_recons_X1_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_Mappedafter_KMF" + "_Direct_recons_X1_" + str(feature_name) + ".png", bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NOTMappedafter_KMF_Direct_recons_X1_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NOTMappedafter_KMF_Direct_recons_X1_" + str(feature_name) + ".png", bbox='tight')
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
    #             filename_for_saving_PM_quality + "_Mappedafter_KMF" + "_Direct_recons_X2_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_Mappedafter_KMF" + "_Direct_recons_X2_" + str(feature_name) + ".png", bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NOTMappedafter_KMF_Direct_recons_X2_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NOTMappedafter_KMF_Direct_recons_X2_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    del df_rename_preproc

    # F1 sore computation

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
                FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    return correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2



def main(dataset_no_sample):

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(100)
    np.random.seed(100)  # fixing the seed for reproducibility

    # reading the data
    # reading the data
    # filename = "./Syn_data/Complex_synData_dim11_size_10000_for_AE_balanced.csv"  # complex data
    filename = "/project/tantra/Sandhya/DeployModel/Syn_data/SD_" + str(dataset_number) + "/2021-05-18Syn_Data_" + str(
        dataset_number) + "_Sample_no_" + str(
        dataset_no_sample) + "_size_20_10000_for_AE_balanced.csv"  # for dataset 1 and 2
    # filename = "./Syn_data/SD_" + str(dataset_number) + "/2021-05-21Syn_Data_" + str(dataset_number) +  "_Sample_no_" + str(dataset_no_sample) +"_size_50_10000_for_AE_balanced.csv"  # for dataset 5
    # filename = "./Syn_data/SD_" + str(dataset_number) + "/2021-05-24Syn_Data_" + str(dataset_number) +  "_Sample_no_" + str(dataset_no_sample) +"_size_30_10000_for_AE_balanced.csv"  # for dataset 6
    full_Data = pd.read_csv(filename)
    num_sample = full_Data.shape[0]
    num_features = full_Data.shape[1] - 1

    # full data initial correlation
    Feature_matrix = full_Data.iloc[:, :-1]
    Cor_from_df = Feature_matrix.corr()

    # output arrays

    AVG_MISMATCHES_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), n_t))
    AVG_MISMATCHES_X2_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))

    AVG_MISMATCHES_X1_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t))
    AVG_MISMATCHES_X2_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))

    AVG_F1_X1_tr = np.zeros((len(list_of_total_Features_in_large_database), n_t))
    AVG_F1_X2_tr = np.zeros((len(list_of_total_Features_in_large_database), n_t))

    F1_across_trial_perm_X1_tr = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))
    F1_across_trial_perm_X2_tr = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))

    AVG_F1_X1_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), n_t))
    AVG_F1_X2_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), n_t))

    F1_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))
    F1_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))

    AVG_F1_X1_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t))
    AVG_F1_X2_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t))

    F1_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_total_Features_in_large_database), n_t * n_p))


    # the set of mapped features is selected apriori and that same set is used across different sample sizes to avoid variation due to the mapped features
    mapped_random = []
    for i in range(n_t):
        mapped_random.append(np.random.choice(num_features, mpfeatures, replace=False))

    mapped_random = np.array(mapped_random)

    selected_feature_indices = []
    extra_features_in_small_list =[]
    features_to_drop_from_small_list =[]
    for trial in range(n_t):
        feature_index_list = []
        a = np.random.choice(np.array(list(set(np.arange(num_features)) - set(mapped_random[trial]) - set(feature_index_list))), num_xtra_feat_inX1, replace = False)
        extra_features_in_small_list.append(a)
        print(extra_features_in_small_list)
        for i  in range(len(list_of_total_feat_in_D2_minus_mapped)):
            if i == 0:
                feature_indices = np.random.choice(
                    np.array(list(set(np.arange(num_features)) - set(mapped_random[trial]) - set(feature_index_list) - set(a))),
                    list_of_total_feat_in_D2_minus_mapped[i],
                    replace=False)
                temp_drop = np.random.choice(feature_indices, num_xtra_feat_inX1, replace = False)
                features_to_drop_from_small_list.append(temp_drop)
            else:
                feature_indices = np.random.choice(np.array(list(set(np.arange(num_features)) - set(mapped_random[trial])- set(feature_index_list)- set(a))), list_of_total_feat_in_D2_minus_mapped[i] - list_of_total_feat_in_D2_minus_mapped[i-1] , replace = False)
            print(len(set(feature_indices).intersection(set(feature_index_list))))
            feature_index_list = np.append(feature_index_list, feature_indices)
            feature_index_list = [int(i) for i in feature_index_list]
            print("number selected", list_of_total_feat_in_D2_minus_mapped[i], " feature_index selected", feature_index_list)
        selected_feature_indices.append(feature_index_list)

    selected_feature_indices = np.array(selected_feature_indices)
    extra_features_in_small_list = np.array(extra_features_in_small_list)
    features_to_drop_from_small_list = np.array(features_to_drop_from_small_list)

    no_match_inference_df_from_x1 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    m = 0  # variables to keep track of the iterations over number of mapped features
    for num_feat in list_of_total_feat_in_D2_minus_mapped:
        run_num = 0  # variable to keep track of the run number out of n_t*n_p
        print("\n ********************************************************")
        print("Run when there are ", num_feat + mpfeatures, " features in large database starts")
        print(" ******************************************************** \n")

        for trial in range(n_t):

            # array for saving the frac of mistakes
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Kang = np.zeros(n_p)

            F1_for_fixed_trial_fixed_num_mapped_X1_tr = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang = np.zeros(n_p)

            # the copies are being made because of the multiple trials
            df_train = full_Data.copy()

            print("\n ********************************************************")
            print("Trial number ", trial + 1, "   starts when there are ", num_feat+ mpfeatures, " features in large database starts")
            print(" ******************************************************** \n")


            # keeping a holdout sample aside
            Df_for_training, Df_holdout = model_selection.train_test_split(df_train, test_size=0.1,
                                                                    random_state=42 * trial * 10,
                                                                    stratify=df_train[outcome])


            # splitting the holdout df into two for using in the two databases
            Df_holdout_orig0, DF_holdout_r0 = model_selection.train_test_split(Df_holdout, test_size=0.5, random_state=42,stratify=Df_holdout[outcome])


            df_train1, df_train2 = model_selection.train_test_split(Df_for_training, test_size=frac_renamed,
                                                                    random_state=42 * trial * 10,
                                                                    stratify=Df_for_training[outcome])

            # shuffling the mapped and unmapped
            unmapped_random_orig = selected_feature_indices[trial,:list_of_total_feat_in_D2_minus_mapped[0]]
            unmapped_random_extra_in_orig = extra_features_in_small_list[trial]
            unmapped_random_to_drop_from_orig = features_to_drop_from_small_list[trial]
            unmapped_random_reshuffle = selected_feature_indices[trial,:num_feat]

            mapped_features = ["Col" + str(i + 1) for i in mapped_random[trial]]
            unmapped_features_orig = ["Col" + str(i + 1) for i in unmapped_random_orig]
            unmapped_features_extra_orig = ["Col" + str(i + 1) for i in unmapped_random_extra_in_orig]
            unmapped_features_to_drop_from_orig = ["Col" + str(i + 1) for i in unmapped_random_to_drop_from_orig]
            unmapped_features_reshuffle = ["Col" + str(i + 1) for i in unmapped_random_reshuffle]

            print("mapped features", mapped_features)
            print("unmapped orig/small", unmapped_features_orig)
            print("unmapped extra/without_match  in orig/small", unmapped_features_extra_orig)
            print(" unmapped features that were replaced by extra in orig/match", unmapped_features_to_drop_from_orig)
            print("unmapped rename/large", unmapped_features_reshuffle)

            print("List of mapped features for trial number", trial + 1, "is ", mapped_features)

            # reordering to making sure that mapped features are at the starts of the vector
            feature_names_orig = mapped_features + unmapped_features_orig +unmapped_features_extra_orig

            df_train1 = df_train1.reindex(columns=feature_names_orig + [outcome])



            print(" trial data details \n")
            print("size of total train", len(df_train))
            print("size of train original", len(df_train1))
            print("size of train renamed", len(df_train2))

            device = torch.device('cpu')

            """ ORIGINAL DATA PREP """
            # data pre-processing normalization
            normalizing_values_orig = {}
            normalizing_values_orig['mean'] = df_train1[feature_names_orig].mean(axis=0)
            normalizing_values_orig['std'] = df_train1[feature_names_orig].std(axis=0)
            normalizing_values_orig['min'] = df_train1[feature_names_orig].min(axis=0)
            normalizing_values_orig['max'] = df_train1[feature_names_orig].max(axis=0)

            df_train_preproc0 = normalization(df_train1, 'mean_std', normalizing_values_orig, feature_names_orig)
            Df_holdout_orig0 = normalization(Df_holdout_orig0, 'mean_std', normalizing_values_orig, feature_names_orig)


            DF_holdout_orig0_not_includedwhiletraining = Df_holdout_orig0.reindex(columns = unmapped_features_to_drop_from_orig)
            feature_names_orig = [i for i  in feature_names_orig if i not in unmapped_features_to_drop_from_orig]
            df_train1 = df_train1.reindex(columns=feature_names_orig + [outcome])
            df_train_preproc0 = df_train_preproc0.reindex(columns = feature_names_orig+[outcome])


            reordered_column_names_orig = mapped_features + [col for col in df_train_preproc0.columns if
                                                             col not in mapped_features + [outcome]] + [outcome]
            df_train_preproc0 = df_train_preproc0.reindex(columns=reordered_column_names_orig)
            Df_holdout_orig0 = Df_holdout_orig0.reindex(columns=reordered_column_names_orig)

            """ SHUFFLED FEATURES DATA PREP """

            # reordering to making sure that mapped features are at the starts of the vector

            feature_names_r = mapped_features + unmapped_features_reshuffle
            df_train2 = df_train2.reindex(columns = feature_names_r + [outcome] )
            DF_holdout_r0 = DF_holdout_r0.reindex(columns = feature_names_r + [outcome] )

            # square transformation on *num_feat_sq_trans* randomly selected unmapped variables
            if num_feat_sq_trans > 0:
                feat_index_for_trans = np.random.choice(np.arange(mpfeatures, len(feature_names_r)), num_feat_sq_trans,
                                                        replace=False)

                for j in feat_index_for_trans:
                    df_train2.iloc[:, j] = df_train2.iloc[:, j] * df_train2.iloc[:, j]
                    DF_holdout_r0.iloc[:, j] = DF_holdout_r0.iloc[:, j] * DF_holdout_r0.iloc[:, j]
                print("\n  Type of transformation is non 1-1 like square")
                print("Transformed feature names for dataset rename is ",
                      [list(df_train2.columns)[i] for i in feat_index_for_trans])

            # data preprocessing
            normalizing_values_r = {}
            normalizing_values_r['mean'] = df_train2[feature_names_r].mean(axis=0)
            normalizing_values_r['std'] = df_train2[feature_names_r].std(axis=0)
            normalizing_values_r['min'] = df_train2[feature_names_r].min(axis=0)
            normalizing_values_r['max'] = df_train2[feature_names_r].max(axis=0)

            df_rename_preproc0 = normalization(df_train2, 'mean_std', normalizing_values_r, feature_names_r)
            DF_holdout_r0 = normalization(DF_holdout_r0, 'mean_std', normalizing_values_r, feature_names_r)

            # maximum possible mistakes for this trial
            max_mistakes = len(feature_names_orig) - len(mapped_features)  # this is taken to be orig (small number of feature case) because in both cases of correlation matrix we are treating orig to be hospital that have capacity

            for partition in range(n_p):
                df_train_preproc = df_train_preproc0.copy()
                df_rename_preproc = df_rename_preproc0.copy()  # a copy to keep the original ordering as a baseline when matching
                DF_holdout_r = DF_holdout_r0.copy()
                Df_holdout_orig = Df_holdout_orig0.copy()

                print("\n ********************************************************")
                print(" Partition number ", partition + 1, "   starts for trail number ", trial + 1,
                      " when there are ",
                      mpfeatures, " mapped features")
                print(" ******************************************************** \n")

                if ordering_type == 15:  # all unmapped features are shuffled
                    reorder_feat = unmapped_features_reshuffle.copy()
                    random.shuffle(reorder_feat)

                reordered_column_names_r = mapped_features + reorder_feat + [outcome]
                df_rename_preproc = df_rename_preproc.reindex(columns=reordered_column_names_r)
                DF_holdout_r = DF_holdout_r.reindex(columns=reordered_column_names_r)

                print("\n \n ------  Ordering of variables when # of mapped features is ", mpfeatures,
                      " trial number is ",
                      trial + 1, " partition number is ", partition + 1, "\n ")
                print(" Original dataset \n ", reordered_column_names_orig)
                print(" Permuted features dataset \n", reordered_column_names_r)

                # printing the permutation matrix
                P_x1 = np.zeros((len(reordered_column_names_orig), len(reordered_column_names_r)))

                for i in range(len(reordered_column_names_orig)):
                    for j in range(len(reordered_column_names_r)):
                        if reordered_column_names_orig[i] == reordered_column_names_r[j]:
                            P_x1[i,j]=1

                correct_with_match_from_x1_test_Kang,F1_x1_kang = Kang_MI_HC_opt(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[:-1,:-1],
                    df_train_preproc.columns, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r)

                correct_with_match_from_x1_test_sim_cor, correct_with_match_from_x2_test_sim_cor, correct_with_no_match_from_CCx1_test_sim_cor, correct_with_no_match_from_CCx2_test_sim_cor, temp_infer_from_x1_sim_cor, temp_infer_from_x2_sim_cor, mapp_fea_to_add, mapp_fea_to_add_match, F1_x1_sim_cor, F1_x2_sim_cor = Simple_maximum_sim_viaCorrelation(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):-1,
                                                                       len(mapped_features):-1],
                    df_train_preproc.columns, reordered_column_names_r, mapped_features, Cor_from_df,Df_holdout_orig,
                    DF_holdout_r)

                # prep for second stage

                mapped_features_updated_orig = mapped_features + mapp_fea_to_add
                mapped_features_updated_r = mapped_features + mapp_fea_to_add_match

                remaining_unmapped_feature_orig = [col for col in df_train_preproc.columns if col not in mapped_features_updated_orig + [outcome] ]
                remaining_unmapped_feature_r = [col for col in df_rename_preproc.columns if col not in mapped_features_updated_r + [outcome] ]

                reordered_column_names_orig_updated = mapped_features_updated_orig + remaining_unmapped_feature_orig + [outcome]
                reordered_column_names_r_updated = mapped_features_updated_r + remaining_unmapped_feature_r + [outcome]


                df_train_preproc = df_train_preproc.reindex(columns=reordered_column_names_orig_updated)
                Df_holdout_orig = Df_holdout_orig.reindex(columns = reordered_column_names_orig_updated)

                df_rename_preproc = df_rename_preproc.reindex(columns=reordered_column_names_r_updated)
                DF_holdout_r = DF_holdout_r.reindex(columns = reordered_column_names_r_updated)

                # updated permutation matrix
                P_x1_updated = np.zeros((len(reordered_column_names_orig_updated), len(reordered_column_names_r_updated)))

                for i in range(len(reordered_column_names_orig_updated)):
                    for j in range(len(reordered_column_names_r_updated)):
                        if reordered_column_names_orig_updated[i] == reordered_column_names_r_updated[j]:
                            P_x1_updated[i,j]=1

                filename_for_saving_PM_quality = saving_dir + "/Syn" + str(dataset_number)+"_PM_quality_part#_" +str(partition)+"_trial#_"+str(trial)+"_#mfeat_"+str(mpfeatures)+"_L_dim_"+str(hidden_dim)+"feat_inLargerDB_"+str(num_feat+len(mapped_features))
                correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_infer_from_x1, temp_infer_from_x2, F1_x1, F1_x2 = Train_cross_AE(df_train_preproc.copy(), df_rename_preproc.copy(), P_x1_updated[len(mapped_features):-1,
                                      len(mapped_features):-1], reordered_column_names_orig_updated, reordered_column_names_r_updated, mapped_features,mapped_features_updated_orig, mapped_features_updated_r, Cor_from_df, Df_holdout_orig, DF_holdout_r, unmapped_features_to_drop_from_orig,filename_for_saving_PM_quality,normalizing_values_orig,normalizing_values_r, DF_holdout_orig0_not_includedwhiletraining)

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = 1 - (
                            correct_with_match_from_x1_test + correct_with_no_match_from_CCx1_test) / (max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = 1 - (
                            correct_with_match_from_x2_test + correct_with_no_match_from_CCx2_test) / (max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[
                    partition]
                Frac_mismatches_across_trial_perm_X2_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[
                    partition]

                no_match_inference_df_from_x1 = pd.concat([no_match_inference_df_from_x1, temp_infer_from_x1])
                no_match_inference_df_from_x2 = pd.concat([no_match_inference_df_from_x2, temp_infer_from_x2])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor[partition] = 1 - (
                            correct_with_match_from_x1_test_sim_cor + correct_with_no_match_from_CCx1_test_sim_cor) / (
                                                                                         max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor[partition] = 1 - (
                            correct_with_match_from_x2_test_sim_cor + correct_with_no_match_from_CCx2_test_sim_cor) / (
                                                                                         max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr_Sim_cor[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor[partition]
                Frac_mismatches_across_trial_perm_X2_tr_Sim_cor[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor[partition]

                no_match_inference_df_from_x1_Sim_cor = pd.concat(
                    [no_match_inference_df_from_x1_Sim_cor, temp_infer_from_x1_sim_cor])
                no_match_inference_df_from_x2_Sim_cor = pd.concat(
                    [no_match_inference_df_from_x2_Sim_cor, temp_infer_from_x2_sim_cor])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = 1-(correct_with_match_from_x1_test_Kang)/(max_mistakes)
                # Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Kang[partition] = 1-(correct_with_match_from_x2_test_Kang + correct_with_no_match_from_CCx2_test_Kang)/(max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr_Kang[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]
                # Frac_mismatches_across_trial_perm_X2_tr_Kang[m, run_num] = (1-correct_with_match_from_x2_test_Kang)/(max_mistakes -num_xtra_feat_inX1)

                # no_match_inference_df_from_x1_Kang = pd.concat([no_match_inference_df_from_x1_Kang, temp_infer_from_x1_Kang])
                # no_match_inference_df_from_x2_Kang = pd.concat([no_match_inference_df_from_x2_Kang, temp_infer_from_x2_Kang])

                F1_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = F1_x1
                F1_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = F1_x2

                F1_across_trial_perm_X1_tr[m, run_num] = F1_x1
                F1_across_trial_perm_X2_tr[m, run_num] = F1_x2

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = F1_x1_sim_cor
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = F1_x2_sim_cor

                F1_across_trial_perm_X1_tr_sim_Cor[m, run_num] = F1_x1_sim_cor
                F1_across_trial_perm_X2_tr_sim_Cor[m, run_num] = F1_x2_sim_cor


                F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = F1_x1_kang

                F1_across_trial_perm_X1_tr_Kang[m, run_num] = F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]


                run_num = run_num + 1

                # Deleting the reshuffled as we have already made a copy earlier
                del df_rename_preproc, df_train_preproc, DF_holdout_r, Df_holdout_orig
                # storing the averaged mismatches across all paritition for a fixed trial and fixed number of mapped features

            print(" Sim correlation method when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor != []:
                AVG_MISMATCHES_X1_tr_Sim_cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Sim_cor)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor != []:
                AVG_MISMATCHES_X2_tr_Sim_cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Sim_cor)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor != []:
                AVG_F1_X1_tr_sim_Cor[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor != []:
                AVG_F1_X2_tr_sim_Cor[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            print(" Kang et al's MI and HC based method when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Kang)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang != []:
                AVG_MISMATCHES_X1_tr_Kang[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Kang != []:
                AVG_MISMATCHES_X2_tr_Kang[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_Kang)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang != []:
                AVG_F1_X1_tr_Kang[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)

            print(" Chimeric AE when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr != []:
                AVG_MISMATCHES_X1_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr != []:
                AVG_MISMATCHES_X2_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr != []:
                AVG_F1_X1_tr[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr != []:
                AVG_F1_X2_tr[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr)

        m = m + 1
        del df_train

    print(" ----  Simple correlation ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_Sim_cor)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_Sim_cor)

    print(" ----  Kang et al's method ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_Kang)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_Kang)

    print(" ---- Two stage  Chimeric AE ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr)

    no_match_inference_df_from_x1.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_two_stage.csv", index=False)
    no_match_inference_df_from_x2.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_two_stage.csv", index=False)

    no_match_inference_df_from_x1_Sim_cor.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)
    no_match_inference_df_from_x2_Sim_cor.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)

    return AVG_F1_X1_tr, AVG_F1_X2_tr, np.average(AVG_F1_X1_tr, axis=1), np.average(AVG_F1_X2_tr, axis = 1), F1_across_trial_perm_X1_tr, F1_across_trial_perm_X2_tr, \
           AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, np.average(AVG_F1_X1_tr_sim_Cor, axis=1), np.average(AVG_F1_X2_tr_sim_Cor, axis = 1), F1_across_trial_perm_X1_tr_sim_Cor, F1_across_trial_perm_X2_tr_sim_Cor,\
           AVG_F1_X1_tr_Kang, np.average(AVG_F1_X1_tr_Kang,axis=1), F1_across_trial_perm_X1_tr_Kang


n_p = 3  # number of permutations
n_t = 4  # number of data partitioning trials
# list_of_total_Features_in_large_database = [7, 8, 9, 11] # fo complex data
list_of_total_Features_in_large_database = [12,14, 16,18, 20] # for dataset 2
# list_of_total_Features_in_large_database = [15,20,25,30,35,40,45,50]  # dataset 3
num_feat_sq_trans = 0

# data details
outcome = "Y"
dataset_number =  2  #  5 for complex data dependencies
frac_renamed = 0.5
ordering_type = 15  # number denotes the number of variables swapped, 41 denotes symmetrically swapped and 42 denotes asymmetric swapping, 15 is randomly shuffling all of them
num_of_dataset_samples = 5
mpfeatures = 2  # number of mapped features
num_xtra_feat_inX1 = 0  # 0 in onto case, 5 in ppartial mapping case
datatype = 'c'  # b for the case whent he data is binarized


alpha = 1.2  # used in KANG method


list_of_total_feat_in_D2_minus_mapped = [i-mpfeatures for i in list_of_total_Features_in_large_database]

# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization
weight_direct = 0.6
weight_cross = 1.2  # 0 denotes no cross loss, 1 denotes cross loss
weight_cycle = 0.9
dropout_rate = 0.6


# model architecture and parameter details
hidden_dim = 5
num_of_hidden_layers = 2
batch_size = 64
epochs = 40
learning_rate = 1e-2

# file saving logistics
saving_dir = './ChimericAE_Final_'+str(datetime.date.today()) +  "/#ofhidden_layers_" + str(
    num_of_hidden_layers) + '/L_dim_' + str(hidden_dim) + '/Syn_' + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + "_Asymmetric_NOT_all_matches_TWO-stage_vs_KMF_KANG_Alpha_" + str(alpha)+"_mapped_features_"+str(mpfeatures)+"_dropout_"+str(dropout_rate)+str(datetime.datetime.now())+"_with_nomatch_"+str(num_xtra_feat_inX1)

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

AVG_over_Dataset_samples_X1_tr = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))

file_name = saving_dir + "/" + "Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name)

AVG_over_Dataset_samples_X1_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_Sim_cor = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))

file_name_Sim_cor = saving_dir + "/" + "Sim_cor_Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(
    orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(
        file_name_Sim_cor):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_Sim_cor)

AVG_over_Dataset_samples_X1_tr_KANG = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_KANG = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))

file_name_KANG = saving_dir + "/" + "KANG_Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(
    orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(
        file_name_KANG):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_KANG)

# AVG_over_Dataset_samples_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), num_of_dataset_samples))
# AVG_over_Dataset_samples_X2_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), num_of_dataset_samples))
#
# file_name_RG = saving_dir + "/" + "RG_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
#     dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"
#
# if os.path.exists(file_name_RG):  # removing the existing file so that it doesn't append the file from previous experiments
#     os.remove(file_name_RG)


f = open(file_name, 'w')
f.write("\n \n *** Chimeric AE Present file settings ***")
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

# f = open(file_name_RG, 'w')
# f.write("\n \n *** RadialGAN Present file settings ***")
# f.write("\n \n Dataset number {0}\t".format(dataset_number))
# f.write("\n Grad penalty parameter {0}\t".format(beta))
# f.write("\n Cycle consistency weight on dir {0}\t ".format(Lambda_dir))
# f.write("\n Cycle consistency weight on z {0}\t ".format(Lambda_on_z))
# f.write("\n Number of epochs {0}\t".format(epochs_RadialGAN))
# f.write("\n Starting learning rate {0}\t ".format(learning_rate))
# f.write("\n Batch size {0}\t".format(batch_size))
# f.write("\n")
# f.close()


f = open(file_name_Sim_cor, 'w')
f.write("\n \n *** Simple_correlation Present file settings ***")
f.close()

f = open(file_name_KANG, 'w')
f.write("\n \n *** KANG Present file settings ***")
f.close()

for sample_no in range(1,num_of_dataset_samples+1):
    print("\n ********************************************************")
    print(" \n Run STARTS for sample no ", sample_no, "  of dataset ", dataset_number, "\n")
    print(" ******************************************************** \n")

    AVG_F1_X1_tr, AVG_F1_X2_tr, m_x1, m_x2_tr, F1_elongated_X1_tr, F1_elongated_X2_tr, AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor,\
    m_x1_sim_Cor, m_x2_tr_sim_Cor, F1_elongated_X1_tr_sim_Cor, F1_elongated_X2_tr_sim_Cor, AVG_F1_X1_tr_KANG, m_X1_tr_KANG, F1_elongated_X1_tr_KANG = main(
        sample_no)

    # for Chimeric AE

    AVG_over_Dataset_samples_X1_tr[:,sample_no-1] = m_x1
    AVG_over_Dataset_samples_X2_tr[:,sample_no-1] = m_x2_tr

    f = open(file_name,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        F1_elongated_x1_tr_list = F1_elongated_X1_tr
        F1_elongated_x2_tr_list = F1_elongated_X2_tr
    else:
        F1_elongated_x1_tr_list = np.hstack((F1_elongated_x1_tr_list, F1_elongated_X1_tr))
        F1_elongated_x2_tr_list = np.hstack((F1_elongated_x2_tr_list, F1_elongated_X2_tr))

    # for Simple_correlation (Simple_correlation)

    AVG_over_Dataset_samples_X1_tr_Sim_cor[:,sample_no-1] = m_x1_sim_Cor
    AVG_over_Dataset_samples_X2_tr_Sim_cor[:,sample_no-1] = m_x2_tr_sim_Cor

    f = open(file_name_Sim_cor,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_sim_Cor))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr_sim_Cor))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        F1_elongated_x1_tr_list_sim_Cor = F1_elongated_X1_tr_sim_Cor
        F1_elongated_x2_tr_list_sim_Cor = F1_elongated_X2_tr_sim_Cor
    else:
        F1_elongated_x1_tr_list_sim_Cor = np.hstack((F1_elongated_x1_tr_list_sim_Cor, F1_elongated_X1_tr_sim_Cor))
        F1_elongated_x2_tr_list_sim_Cor = np.hstack((F1_elongated_x2_tr_list_sim_Cor, F1_elongated_X2_tr_sim_Cor))

    # KANG method

    AVG_over_Dataset_samples_X1_tr_KANG[:, sample_no - 1] = m_X1_tr_KANG
    # AVG_over_Dataset_samples_X2_tr_KANG[:, sample_no - 1] = m_x2_tr_KANG

    f = open(file_name_KANG, 'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_KANG))
    # f.write("\n X2_train \n")
    # f.write("{0}".format(AVG_F1_X2_tr_KANG))
    f.write("\n \n ")
    f.close()

    if sample_no == 1:
        F1_elongated_x1_tr_list_KANG = F1_elongated_X1_tr_KANG
        # F1_elongated_x2_tr_list_KANG = F1_elongated_X2_tr_KANG
    else:
        F1_elongated_x1_tr_list_KANG = np.hstack((F1_elongated_x1_tr_list_KANG, F1_elongated_X1_tr_KANG))
        # F1_elongated_x2_tr_list_KANG = np.hstack((F1_elongated_x2_tr_list_KANG, F1_elongated_X2_tr_KANG))

    print("\n ********************************************************")
    print(" \n Run ENDS for sample no ", sample_no, "\n ")
    print(" ******************************************************** \n")

file_name_violin = saving_dir + "/" + "For_violin_Mismatch_metric_L_" + str(hidden_dim) + "_Syn_" + str(
    dataset_number) + "_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_violin):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_violin)

f = open(file_name_violin,'a')
f.write("\n \n List of total features in large database when small database has 10 features \n ")
f.write("{0}".format(list_of_total_Features_in_large_database))
f.write("\n \n F1 for across trials and perm ")
f.write("\n X1_train \n")
f.write("{0}".format(F1_elongated_x1_tr_list))
f.write("\n X2_train \n")
f.write("{0}".format(F1_elongated_x2_tr_list))
f.write("\n X1_train_sim_Cor \n")
f.write("{0}".format(F1_elongated_x1_tr_list_sim_Cor))
f.write("\n X2_train_sim_Cor \n")
f.write("{0}".format(F1_elongated_x2_tr_list_sim_Cor))
f.write("\n X1_train_KANG \n")
f.write("{0}".format(F1_elongated_x1_tr_list_KANG))
f.write("\n \n ")
f.close()

# exit()

# Computing the average over the datset samples
Mean_over_trials_mismatches_X1_tr = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr = np.zeros(len(list_of_total_Features_in_large_database))

Mean_over_trials_mismatches_X1_tr_Sim_cor = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_Sim_cor = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_Sim_cor = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_Sim_cor = np.zeros(len(list_of_total_Features_in_large_database))

Mean_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_total_Features_in_large_database))
# Mean_over_trials_mismatches_X2_tr_KANG = np.zeros(len(list_of_total_Features_in_large_database))
# SD_over_trials_mismatches_X2_tr_KANG = np.zeros(len(list_of_total_Features_in_large_database))

x_axis = np.arange(len(list_of_total_Features_in_large_database))
x_axis1 = x_axis + 0.05
x_axis2 = x_axis + 0.1
for i in range(len(list_of_total_Features_in_large_database)):
    Mean_over_trials_mismatches_X1_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr[i] = np.round(np.std(F1_elongated_x1_tr_list[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr[i] = np.round(np.std(F1_elongated_x2_tr_list[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_Sim_cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_Sim_cor[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_Sim_cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_Sim_cor[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_Sim_cor[i] = np.round(np.std(F1_elongated_x1_tr_list_sim_Cor[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_Sim_cor[i] = np.round(np.std(F1_elongated_x2_tr_list_sim_Cor[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_KANG[i, :]), decimals=3)
    # Mean_over_trials_mismatches_X2_tr_KANG[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_KANG[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.std(F1_elongated_x1_tr_list_KANG[i, :]), decimals=4)
    # SD_over_trials_mismatches_X2_tr_KANG[i] = np.round(np.std(Frac_x2_tr_list_KANG[i, :]), decimals=4)


plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr, color='blue', label=" KMF -> ChimericE  ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr, yerr=SD_over_trials_mismatches_X1_tr, linestyle="solid",
             color='blue')
# plt.scatter(x_axis, Mean_over_trials_mismatches_X2_tr, color='green', label=" From cross correlation of X2 ", linestyle='None')
# plt.errorbar(x_axis, Mean_over_trials_mismatches_X2_tr, yerr=SD_over_trials_mismatches_X2_tr, linestyle="solid",
#              color='green')
plt.scatter(x_axis1, Mean_over_trials_mismatches_X1_tr_Sim_cor, color='red', label=" KMF ", linestyle='None')
plt.errorbar(x_axis1, Mean_over_trials_mismatches_X1_tr_Sim_cor, yerr=SD_over_trials_mismatches_X1_tr_Sim_cor, linestyle="solid",
             color='red')

plt.scatter(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, color='brown', label=" Kang ", linestyle='None')
plt.errorbar(x_axis2, Mean_over_trials_mismatches_X1_tr_KANG, yerr=SD_over_trials_mismatches_X1_tr_KANG, linestyle="solid",
             color='brown')
# plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_RG, color='red', label=" RadialGAN ", linestyle='None')
# plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_RG, yerr=SD_over_trials_mismatches_X1_tr_RG, linestyle="solid",
#              color='red')
plt.xticks(x_axis, np.array(list_of_total_Features_in_large_database))
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Number of total features in large feature sized XB")
plt.ylabel("F1 score across different dataset samples")
plt.title("Dataset no. " + str(dataset_number) + " with " +str(list_of_total_Features_in_large_database[0]) + " features in X_A (small) \n and varying features in X_B (large)")
plt.legend()
if num_xtra_feat_inX1 != 0:
    plt.savefig(saving_dir + "/F1_Not_All_matchesASym_Chim_Syn_" + str(dataset_number) + "_varyingData_num_total_fea_inX2_" + str(
    len(list_of_total_Features_in_large_database)) + "_Num_feat_sq_"+str(num_feat_sq_trans) +  ".pdf", bbox='tight')
    plt.savefig(saving_dir + "/F1_Not_All_matchesASym_Chim_Syn_" + str(dataset_number) + "_varyingData_num_total_fea_inX2_" + str(
    len(list_of_total_Features_in_large_database)) + "_Num_feat_sq_"+str(num_feat_sq_trans) +  ".png", bbox='tight')
else:
    plt.savefig(
        saving_dir + "/F1_All_matchesASym_Chim_Syn_" + str(dataset_number) + "_varyingData_num_total_fea_inX2_" + str(
            len(list_of_total_Features_in_large_database)) + "_Num_feat_sq_" + str(num_feat_sq_trans) + ".pdf",
        bbox='tight')
    plt.savefig(
        saving_dir + "/F1_All_matchesASym_Chim_Syn_" + str(dataset_number) + "_varyingData_num_total_fea_inX2_" + str(
            len(list_of_total_Features_in_large_database)) + "_Num_feat_sq_" + str(num_feat_sq_trans) + ".png",
        bbox='tight')
plt.close()
