"""
this file provides a pipeline of feature mapping problem. The components are:
1) Train an Autoencoder and compute the correlation matrix between the true feature values and the reconstructed features from cross AEs.
The model being used has no batch normalization and has simple orthogonalization ( in individual AEs).
2) Using hospital resident matching algorithm (with capacity 1) on the correlation matrix from 1 and get the final mappings.


The two methods that are being compared are two stage Chimeric AE approach and Simple correlation approach.

INPUT:

MIMIC dataset (continuous real data), the model details, number of permutations, number of partitioning of dataset, fraction of data to be permuted, number of mapped features

OUTPUT:

An array with rows as the number of mapped features and columns as the trial number and cell entry as the avg mismatches for the a fixed trial and the fixed number of mapped variables


This code has randomness over mapped features and unmapped features too
In this code, we see the change in fraction of mistakes as the difference between the number of features in two databases increases.
Also, all the features in smaller dataset need not be in the feature set of larger dataset.
This is with the motivation that a lot of features are redundant if the underlying factors are very small.


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
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise, mutual_info_score
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
from matching.games import StableMarriage, HospitalResident
import sys
import pingouin as pg
import datetime
import itertools
print(sys.getrecursionlimit())
sys.setrecursionlimit(3500)
print(sys.getrecursionlimit())


def Matching_via_HRM(C_X1_train, C_X2_train, P_x1_O_to_R,
                     num_mapped_axis):  # in this case here the small feature sized database is X1, so we need to treat it as hospital and there will be capacities on it.
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
    game_X1_train = HospitalResident.create_from_dictionaries(cross_recon_features_pref_X1_train,
                                                              true_features_pref_X1_train,
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

    game_X2_train = HospitalResident.create_from_dictionaries(true_features_pref_X2_train,
                                                              cross_recon_features_pref_X2_train,
                                                              capacities_X2_train)

    ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)

    x1_train_y = np.array([int(str(v[0]).split("C")[1]) for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v[0]).split("R")[1]) for v in matching_x2_train.values()])

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(np.transpose(C_X2_train).shape)

    print(" x1 matching shape ", matching_x1_train_matrix.shape)
    print(" x1 matching shape", matching_x2_train_matrix.shape)
    print(" true perm matrix shape ", P_x1_O_to_R.shape)

    for i in range(matching_x1_train_matrix.shape[0]):
        # print(i, x1_train_y[i]-1)
        matching_x1_train_matrix[i, x1_train_y[i] - 1] = 1

    for i in range(matching_x2_train_matrix.shape[0]):
        # print(i, x2_train_y[i]-1)
        matching_x2_train_matrix[i, x2_train_y[i] - 1] = 1

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

class AE_binary(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.no_of_cont = kwargs["input_shape"]
        self.drop_out_rate = kwargs["drop_out_p"]

        # print("input_dimension_total", self.no_of_cont)
        self.encoder_hidden_layer1 = nn.Linear(in_features= self.no_of_cont, out_features=80)
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer2 = nn.Linear(in_features= 80, out_features=50)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=50, out_features=hidden_dim)
        self.decoder_hidden_layer1 = nn.Linear(in_features=hidden_dim, out_features=50)
        self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer2 = nn.Linear(in_features=50, out_features=80)
        self.drop_layer4 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_output_layer = nn.Linear(in_features=80, out_features=kwargs["input_shape"])

    def forward(self, cont_data, cross):
        if cross != 1:
            # print("inside the normal loop")
            activation = self.encoder_hidden_layer1(cont_data)
            # activation = self.drop_layer1(activation)
            activation = self.encoder_hidden_layer2(activation)
            activation=torch.relu(activation)
            activation = self.drop_layer2(activation)
            code = self.encoder_output_layer(activation)
            # code = torch.tanh(code)
            activation = self.decoder_hidden_layer1(code)
            activation = self.drop_layer3(activation)
            activation = self.decoder_hidden_layer2(activation)
            activation=torch.relu(activation)
            # activation = self.drop_layer4(activation)
            activation = self.decoder_output_layer(activation)
            reconstructed_cat_bin = torch.sigmoid(activation)
        else:
            # print("inside the cross loop")
            code = cont_data
            activation = self.decoder_hidden_layer1(code)
            activation = self.drop_layer3(activation)
            activation = self.decoder_hidden_layer2(activation)
            activation=torch.relu(activation)
            # activation = self.drop_layer4(activation)
            activation = self.decoder_output_layer(activation)
            reconstructed_cat_bin = torch.sigmoid(activation)

        return code, reconstructed_cat_bin

class AE_2_hidden_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.no_of_cont = kwargs["input_shape"]
        self.batchnorm = kwargs['batchnorm']
        self.drop_out_rate = kwargs["drop_out_p"]

        print("input_dimension_total", self.no_of_cont)
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=120)
        self.bn1 = nn.BatchNorm1d(num_features=120)
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer2 = nn.Linear(in_features=120, out_features=70)
        self.bn2 = nn.BatchNorm1d(num_features=70)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=70, out_features=hidden_dim)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)
        self.decoder_hidden_layer1 = nn.Linear(in_features=hidden_dim, out_features=70)
        self.bn4 = nn.BatchNorm1d(num_features=70)
        self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer2 = nn.Linear(in_features=70, out_features=120)
        self.bn5 = nn.BatchNorm1d(num_features=120)
        self.drop_layer4 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_output_layer = nn.Linear(in_features=120, out_features=kwargs["input_shape"])

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


class AE_3_hidden_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.no_of_cont = kwargs["input_shape"]
        self.batchnorm = kwargs['batchnorm']
        self.drop_out_rate = kwargs["drop_out_p"]

        print("input_dimension_total", self.no_of_cont)
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=120)
        self.bn1 = nn.BatchNorm1d(num_features=120)
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer2 = nn.Linear(in_features=120, out_features=70)
        self.bn2 = nn.BatchNorm1d(num_features=70)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer3 = nn.Linear(in_features=70, out_features=45)
        self.bn3 = nn.BatchNorm1d(num_features=45)
        self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=45, out_features=hidden_dim)
        self.bn4 = nn.BatchNorm1d(num_features=hidden_dim)
        self.decoder_hidden_layer1 = nn.Linear(in_features=hidden_dim, out_features=45)
        self.bn5 = nn.BatchNorm1d(num_features=45)
        self.drop_layer4 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer2 = nn.Linear(in_features=45, out_features=70)
        self.bn6 = nn.BatchNorm1d(num_features=70)
        self.drop_layer5 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer3 = nn.Linear(in_features=70, out_features=120)
        self.bn7 = nn.BatchNorm1d(num_features=120)
        self.drop_layer6 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_output_layer = nn.Linear(in_features=120, out_features=kwargs["input_shape"])

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

            activation = self.encoder_hidden_layer3(activation)
            if self.batchnorm == 1:
                activation = self.bn3(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer3(activation)

            code0 = self.encoder_output_layer(activation)
            if self.batchnorm == 1:
                code0 = self.bn4(code0)

            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer4(activation)

            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn6(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer5(activation)

            activation = self.decoder_hidden_layer3(activation)
            if self.batchnorm == 1:
                activation = self.bn7(activation)
            reconstructed = self.decoder_output_layer(activation)
        else:
            # print("inside the cross loop")
            code0 = cont_data

            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer4(activation)

            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn6(activation)
            activation = F.tanh(activation)
            activation = self.drop_layer5(activation)

            activation = self.decoder_hidden_layer3(activation)
            if self.batchnorm == 1:
                activation = self.bn7(activation)
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
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig)

    num_features = len(reordered_column_names_r)
    num_NonCat_features_r = len(reordered_column_names_r)

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

    num_iter = 5000
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


    true_permutation = list(np.arange(mpfeatures)) +  [np.where(P_x1[a,:]==1)[0] + mpfeatures for a in range(len(P_x1))]
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_permutation[i]==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1

    print(" \n Mistakes by the KANG method on training data")

    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    print(" -------- KANG  methods  ends ------------- \n \n  ")

    del DF_holdout_r

    return correct_total_fromKANG


def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc, P_x1
                                      , reordered_column_names_orig, reordered_column_names_r,
                                      mapped_features, Cor_from_df, Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures

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
            CorMatrix_X1_unmap_mapped_P_value[i, j] = temp[1]

    for i in range(unmapped_features_r):
        for j in range(mpfeatures):
            temp = stats.pearsonr(df_rename_preproc.values[:, mpfeatures + i], df_rename_preproc.values[:, j])
            CorMatrix_X2_unmap_mapped[i, j] = temp[0]
            CorMatrix_X2_unmap_mapped_P_value[i, j] = temp[1]

    print("Checkpoint 1")

    # similarity between the correlation matrices

    if np.any(np.isnan(CorMatrix_X1_unmap_mapped)) == True or np.any(
            np.isnan(CorMatrix_X2_unmap_mapped)) == True:
        CorMatrix_X1_unmap_mapped = np.nan_to_num(CorMatrix_X1_unmap_mapped)
        CorMatrix_X2_unmap_mapped = np.nan_to_num(CorMatrix_X2_unmap_mapped)
        print("Here here")

    sim_cor_X1_to_X2 = np.matmul(CorMatrix_X1_unmap_mapped, np.transpose(CorMatrix_X2_unmap_mapped))
    sim_cor_X2_to_X1 = np.matmul(CorMatrix_X2_unmap_mapped, np.transpose(CorMatrix_X1_unmap_mapped))
    sim_cor_norm_X1_to_X2 = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,
                                                       dense_output=True)
    sim_cor_norm_X2_to_X1 = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped, CorMatrix_X1_unmap_mapped,
                                                       dense_output=True)

    print("Checkpoint 2")

    """ Calling the stable marriage algorithm for mappings  """

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
        sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,
        P_x1, len(mapped_features))

    test_statistic_num_fromX1 = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                 j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    test_statistic_num_fromX2 = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                 j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                     i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small

    print("Checkpoint 3")

    # Bootstrap samples to obtain the standard deviation methods to be later used in p value computation
    num_of_bts = 50
    bts_for_allthe_accepted_matches_fromX1 = np.zeros((unmapped_features_orig, num_of_bts))
    bts_for_allthe_accepted_matches_fromX2 = np.zeros((unmapped_features_orig, num_of_bts))

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
        if np.any(np.isnan(CorMatrix_X1_unmap_mapped_bts)) == True or np.any(
                np.isnan(CorMatrix_X2_unmap_mapped_bts)) == True:
            CorMatrix_X1_unmap_mapped_bts = np.nan_to_num(CorMatrix_X1_unmap_mapped_bts)
            CorMatrix_X2_unmap_mapped_bts = np.nan_to_num(CorMatrix_X2_unmap_mapped_bts)
            print("Here here")

        sim_cor_norm_X1_to_X2_bts = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped_bts,
                                                               CorMatrix_X2_unmap_mapped_bts, dense_output=True)
        sim_cor_norm_X2_to_X1_bts = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped_bts,
                                                               CorMatrix_X1_unmap_mapped_bts, dense_output=True)
        """ Calling the stable marriage algorithm for mappings  """

        # _ ,_ , x1_match_matrix_test_bts, x2_match_matrix_test_bts = Matching_via_HRM(sim_cor_norm_X1_to_X2_bts, sim_cor_norm_X2_to_X1_bts, P_x1, len(mapped_features))

        # we will use the matched found on the whole dataset and use the bootstraps only to get the dot product estimates
        bts_for_allthe_accepted_matches_fromX1[:, bts] = [sim_cor_norm_X1_to_X2_bts[i, j] for i in
                                                          range(x1_match_matrix_test.shape[0]) for
                                                          j in range(x1_match_matrix_test.shape[1]) if
                                                          x1_match_matrix_test[i, j] == 1]
        bts_for_allthe_accepted_matches_fromX2[:, bts] = [sim_cor_norm_X2_to_X1_bts[j, i] for i in
                                                          range(x2_match_matrix_test.shape[0]) for
                                                          j in range(x2_match_matrix_test.shape[1]) if
                                                          x2_match_matrix_test[
                                                              i, j] == 1]

    test_statistic_den_fromX1 = [np.std(bts_for_allthe_accepted_matches_fromX1[i, :]) for i in
                                 range(x1_match_matrix_test.shape[0])]
    test_statistic_den_fromX2 = [np.std(bts_for_allthe_accepted_matches_fromX2[i, :]) for i in
                                 range(x1_match_matrix_test.shape[0])]

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "MV_label", 'match_byGS', "match_byGS(CV_label)", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "MV_label", 'match_byGS', "match_byGS(CV_label)", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    # getting the p values that needs to be tested for significance

    test_statistic_for_cor_sig_fromX1 = np.array(test_statistic_num_fromX1) / np.array(test_statistic_den_fromX1)
    test_statistic_for_cor_sig_fromX2 = np.array(test_statistic_num_fromX2) / np.array(test_statistic_den_fromX2)

    temp_inf_x1.corr_p_value = [stats.norm.sf(abs(x)) * 2 for x in test_statistic_for_cor_sig_fromX1]
    temp_inf_x2.corr_p_value = [stats.norm.sf(abs(x)) * 2 for x in test_statistic_for_cor_sig_fromX2]

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
        temp_inf_x1.loc[i, "MV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(CV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0

    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "MV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(CV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0

    correct_with_no_match_from_CCx1_test = 0
    correct_with_no_match_from_CCx2_test = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
        if temp_inf_x2.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the simple correlation method on holdout data")
    print(" Sim_Correlation  X1_train mistakes number",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)

    print("\n Mistakes by the significance testing algorithm on holdout data")
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
            num_insign_dropp_x1 = num_insign_dropp_x1 + 1

        if Copy_temp_inf_x2.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x2 ", Copy_temp_inf_x2.ump_feature_in_X1[i], Copy_temp_inf_x2.match_byGS[i], Copy_temp_inf_x2.estimated_cross_corr[i])
            Copy_temp_inf_x2.drop([i], inplace=True)
            num_insign_dropp_x2 = num_insign_dropp_x2 + 1

    print(" Number of insignificant feature pair drops from x1 ", num_insign_dropp_x1)
    print(" Number of insignificant feature pair drops from x2 ", num_insign_dropp_x2)

    # ordering the
    Copy_temp_inf_x1 = Copy_temp_inf_x1.sort_values(by='estimated_cross_corr', ascending=False)
    Copy_temp_inf_x2 = Copy_temp_inf_x2.sort_values(by='estimated_cross_corr', ascending=False)

    num_additional_mapped_for_next_stage_x1 = int(len(Copy_temp_inf_x1) / 2)
    num_additional_mapped_for_next_stage_x2 = int(len(Copy_temp_inf_x2) / 2)

    # taking the intersection of the additional mapped features
    temp_x1_x1 = [Copy_temp_inf_x1.ump_feature_in_X1[i] for i in
                  list(Copy_temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x1_match = [Copy_temp_inf_x1.match_byGS[i] for i in
                     list(Copy_temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x2_x1 = [Copy_temp_inf_x2.ump_feature_in_X1[i] for i in
                  list(Copy_temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]
    temp_x2_match = [Copy_temp_inf_x2.match_byGS[i] for i in
                     list(Copy_temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]

    final_additional_mapped = list(set(temp_x1_x1).intersection(temp_x2_x1))
    final_additional_mapped_corr_match = []
    for i in final_additional_mapped:
        final_additional_mapped_corr_match.append(temp_x1_match[temp_x1_x1.index(i)])

    print(" -------- Sim_Correlation  methods  ends ------------- \n \n  ")


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

    del df_rename_preproc

    return correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, final_additional_mapped, final_additional_mapped_corr_match, F1_fromx1, F1_fromx2


def Train_cross_AE(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r,
                   mapped_features, mapped_features_updated_orig, mapped_features_updated_r, Cor_from_df,
                   Df_holdout_orig, DF_holdout_r, P_x1_true):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig)

    num_features = len(reordered_column_names_r)
    num_NonCat_features_r = len(reordered_column_names_r)

    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures

    print(" -------- Chimeric AE training starts with partial mapping -------------  ")

    dataset_orig = TabularDataset(data=df_train_preproc)
    train_loader_orig = DataLoader(dataset_orig, batch_size, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc)
    train_loader_r = DataLoader(dataset_r, batch_size, shuffle=True, num_workers=1)

    if datatype == 'b':
        model_orig = AE_binary(input_shape=num_NonCat_features_orig, drop_out_p=dropout_rate).to(device)
        model_r = AE_binary(input_shape=num_NonCat_features_r, drop_out_p=dropout_rate).to(device)
        criterion = nn.BCELoss()

    if datatype == 'c':
        if num_of_hidden_layers == 2:
            model_orig = AE_2_hidden_layer(input_shape=num_NonCat_features_orig, batchnorm=batchnorm, drop_out_p=dropout_rate).to(
                        device)
            model_r = AE_2_hidden_layer(input_shape=num_NonCat_features_r, batchnorm=batchnorm, drop_out_p=dropout_rate).to(device)
        if num_of_hidden_layers == 3:
            print("----- Check for 3 layer AE --------")
            model_orig = AE_3_hidden_layer(input_shape=num_NonCat_features_orig, batchnorm=batchnorm,
                                           drop_out_p=dropout_rate).to(
                device)
            model_r = AE_3_hidden_layer(input_shape=num_NonCat_features_r, batchnorm=batchnorm,
                                        drop_out_p=dropout_rate).to(device)

        criterion = nn.MSELoss()

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_r = optim.Adam(model_r.parameters(), lr=learning_rate, weight_decay=1e-5)

    # lr scheduler
    scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, patience=2, verbose=True)
    scheduler_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_orig, patience=2, verbose=True)

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
                L_after_first_cross_r, _ = model_r(output_cross_orig, 0)
                _, double_cross_orig = model_orig(L_after_first_cross_r, 1)

                L_after_first_cross_orig, _ = model_orig(output_cross_r, 0)
                _, double_cross_r = model_r(L_after_first_cross_orig, 1)

                train_loss_cycle_orig = criterion(double_cross_orig, x_o)
                train_loss_cycle_r = criterion(double_cross_r, x_r)

                # compute orthogonality loss
                if orthogonalization_type == 1:
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

                train_loss = weight_direct * (train_loss_orig + train_loss_r) + orth_weighting * (
                                     orthog_loss_orig + orthog_loss_r) + weight_cross * (
                                     train_loss_cross_orig + train_loss_cross_r) + weight_cycle * (
                                         train_loss_cycle_orig + train_loss_cycle_r)

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
        print("epoch : {}/{}, total loss = {:.8f}".format(epoch + 1, epochs, loss))
        print("epoch : {}/{}, recon loss ae orig= {:.8f}".format(epoch + 1, epochs, ae_orig_error))
        print("epoch : {}/{}, recon loss ae r= {:.8f}".format(epoch + 1, epochs, ae_r_error))
        print("epoch : {}/{}, cross recon loss  on ae orig when data is renamed = {:.8f}".format(epoch + 1,
                                                                                                 epochs,
                                                                                                 ae_orig_on_r_cross))
        print("epoch : {}/{}, cross recon loss on ae r when data is orig = {:.8f}".format(epoch + 1, epochs,
                                                                                          ae_r_on_orig_cross))
        print("epoch : {}/{}, cycle loss ae orig= {:.8f}".format(epoch + 1, epochs, cycle_for_epoch_orig))
        print("epoch : {}/{}, cycle loss ae r= {:.8f}".format(epoch + 1, epochs, cycle_for_epoch_r))
        print("epoch : {}/{}, ortho loss ae orig= {:.8f}".format(epoch + 1, epochs, ortho_for_epoch_orig))
        print("epoch : {}/{}, ortho loss ae r= {:.8f}".format(epoch + 1, epochs, ortho_for_epoch_r))

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

    ####################  Whole of evaluation and analysis on holdout samples *********************************

    print(" \n **********************************************************************")
    print(" -------------------  Holdout sample observations -------------------")
    print("********************************************************************** \n")
    # comparing actual reconstruction and cross recontruction on original data
    latent_code_Orig_fullTest_orig, recons_orig_Test_from_orig = model_orig(
        torch.Tensor(Df_holdout_orig.values).to(device), 0)
    _, recons_orig_Test_frommodelR = model_r(latent_code_Orig_fullTest_orig, 1)

    # comparing actual reconstruction and cross recontruction on renamed data
    latent_code_renamed_test, recons_rename_Test_frommodelR = model_r(
        torch.Tensor(DF_holdout_r.values).to(device), 0)
    _, recons_rename_Test_frommodelOrig = model_orig(latent_code_renamed_test, 1)

    features_reconst_from_crossR_test = recons_orig_Test_frommodelR.cpu().detach().numpy()
    features_true_orig_test = Df_holdout_orig.values

    features_reconst_from_crossO_test = recons_rename_Test_frommodelOrig.cpu().detach().numpy()
    features_true_renamed_test = DF_holdout_r.values

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

    # # correlation values of the accepted matches
    # print(" correlation values of the accepted matches from CC x1 (Holdout_sample)")
    # print(CC_values_for_testing_from_x1_test)
    # print(" correlation values of the accepted matches from CC x2 (Holdout_sample)")
    # print(CC_values_for_testing_from_x2_test)

    # testing whether some of the proposed matches are such that there exist no match in reality but GS assigned one;
    # False in the reject list below can be interpreted as the case where the  testing procedure says there wasn't any match originally
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
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
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
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0

    correct_with_no_match_from_CCx1_test = 0
    correct_with_no_match_from_CCx2_test = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
        if temp_inf_x2.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the 2stage Chimeric method on holdout data")
    print(" Chimeric  X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" Chimeric  X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)

    print("Mistakes by the significance testing algorithm on holdout data (2stage chimeric)")
    print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)

    # print(" DF for post-hoc analysis from x1")
    # print(temp_inf_x1)
    # print(" DF for post-hoc analysis from x2")
    # print(temp_inf_x2)

    print(" -------- Chimeric AE method training ends ------------- \n \n  ")


    del df_rename_preproc

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


"""  MAIN starting here """

# Reading data files (Moose location)
# CV_full = pd.read_csv('/home/trips/MIMIC_feature_confusion/Final_MIMIC_lab_chart_CV.csv')
# MV_full = pd.read_csv('/home/trips/MIMIC_feature_confusion/Final_MIMIC_lab_chart_MV.csv')
# # Getting list of all items along with the source and label
# item_id_dbsource = pd.read_csv('/home/trips/d_items_chartevents.csv')
# itemid_labs = pd.read_csv('/home/trips/d_items_labevents.csv')

# Reading data files (Tantra location)
CV_full = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/Final_MIMIC_lab_chart_CV.csv')
MV_full = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/Final_MIMIC_lab_chart_MV.csv')
# Getting list of all items along with the source and label
item_id_dbsource = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/d_items_chartevents.csv')
itemid_labs = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/d_items_labevents.csv')

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
itemid_label_dict = dict(zip(list(item_id_dbsource.itemid), list(item_id_dbsource.label)))

""" preparing true match matrix """

# Itemids with match in order of matching (ultiple case as of now; will be removed in a while)
CV_itemids_with_match_t = [211,8549,5815,51,8368,52,5813,8547,113,455,8441,456,618,779,834,814,778,646,506,813,861,1127,1542,770,788,1523,791,1525,811,821,1532,769,1536,8551,5817,678,8554,5820,780,1126,470,190,50,8553,5819,763,683,682,684,450,619,614,615,535,543,444,578,776,773,1162,781,786,1522,784,796,797,798,799,800,807,816,818,1531,827,1534,848,1538,777,762,837,1529,920,1535,785,772,828,829,1286,824,1533,825,1530,815,6206,6207]
MV_itemids_with_match_t = [220045,220046,220047,220050,220051,220052,220056,220058,220074,220179,220180,220181,220210,220224,220227,220228,220235,220277,220339,220545,220546,220546,220546,220587,220602,220602,220615,220615,220621,220635,220635,220644,220645,223751,223752,223761,223769,223770,223830,223830,223834,223835,223876,224161,224162,224639,224684,224685,224686,224687,224688,224689,224690,224695,224696,224697,224701,224828,225612,225624,225624,225625,225625,225634,225639,225640,225641,225642,225643,225664,225667,225668,225668,225677,225677,225690,225690,225698,226512,226534,226537,226707,227442,227445,227456,227457,227464,227465,227465,227466,227466,227467,227467,227565,227566]

# converting the above integers into strings
CV_itemids_with_match_t = [str(i) for i in CV_itemids_with_match_t]
MV_itemids_with_match_t = [str(i) for i in MV_itemids_with_match_t]

match_df = pd.DataFrame(columns = ['CV_itemids', 'CV_labels', 'MV_itemids', 'MV_labels'])
match_df['CV_itemids'] = CV_itemids_with_match_t
match_df['MV_itemids'] = MV_itemids_with_match_t
for i in range(len(match_df)):
    match_df.loc[i, "CV_labels"] = itemid_label_dict[int(match_df.loc[i,'CV_itemids'])]
    match_df.loc[i, "MV_labels"] = itemid_label_dict[int(match_df.loc[i,'MV_itemids'])]

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
Cor_CV = CV_full[list_lab_ids+onlychart_cont_CV].corr()
Cor_MV = MV_full[list_lab_ids+onlychart_cont_MV].corr()

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

match_df.drop(match_df[match_df['CV_itemids'].isin(high_cor_lab_chart_pairsCV.values())].index, inplace= True)
match_df.drop(match_df[match_df['MV_itemids'].isin(high_cor_lab_chart_pairsMV.values())].index, inplace= True)

CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

for i in high_cor_lab_chart_pairsCV.values():
    onlychart_CV.remove(str(i))
    onlychart_cont_CV.remove(str(i))

for i in high_cor_lab_chart_pairsMV.values():
    onlychart_MV.remove(str(i))
    onlychart_cont_MV.remove(str(i))


#final matching dict
match_dic = dict(zip(CV_itemids_with_match, MV_itemids_with_match))

# itemids with no match
CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]
MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]

print( " CV_itemids_with match ", len(CV_itemids_with_match))
print( " MV_itemids_with match ", len(MV_itemids_with_match))

print( " CV_itemids_with NO match ", len(CV_itemids_withnomatch))
print( " MV_itemids_with NO match ", len(MV_itemids_withnomatch))


"""  # data details """
mpfeatures = len(list_lab_ids)
num_xtra_feat_inX1 = len(CV_itemids_withnomatch)
datatype = 'c'  # b for the case whent he data is binarized

alpha = 1.4  # used in KANG method

# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization
# weight_direct = 1
# weight_cross = 1.4  # 0 denotes no cross loss, 1 denotes cross loss
weight_cycle = 0.8

# model architecture and parameter details
# hidden_dim = 25
num_of_hidden_layers = 2   # 5 as a face value for the hidden data
# batch_size = 64
epochs = 40
# learning_rate = 1e-2
# dropout_rate = 0.4


# listing the parameters to search over
weight_direct_list =[0.4,0.5,0.7,1.0]
weight_cross_list =[0.8,1.0,1.2,1.4]
dropout_rate_list = [0.5,0.6,0.7,0.8]
hidden_dim_list = [20,30,40]
batch_size_list = [32, 64]
learning_rate_list = [1e-2, 1e-3]

# weight_direct_list =[0.5]
# weight_cross_list =[0.8,]
# dropout_rate_list = [0.5,0.6]
# hidden_dim_list = [20]
# batch_size_list = [32]
# learning_rate_list = [1e-2, 1e-3]

""" partitioning both the datasets into train and holdout """

# readmission variable was given a thought to be taken as outcome variable to be stratified on
# but since it was not available in both eras so didn't pursue further.

df_train_CV, df_holdout_CV = model_selection.train_test_split(CV_full[list_lab_ids+onlychart_cont_CV], test_size=0.2,
                                                                    random_state=42)
df_train_MV, df_holdout_MV = model_selection.train_test_split(MV_full[list_lab_ids+onlychart_cont_MV], test_size=0.2,
                                                                    random_state=42)

""" # imputation on the partitioned data """
# use the mean of training sets on the test set too so that there is no information leakage

df_train_CV.fillna(df_train_CV.mean(), inplace= True)
df_holdout_CV.fillna(df_train_CV.mean(), inplace= True)

df_train_MV.fillna(df_train_MV.mean(), inplace= True)
df_holdout_MV.fillna(df_train_MV.mean(), inplace= True)

features_CV = list(df_train_CV.columns)
features_MV = list(df_train_MV.columns)

""" # data pre-processing normalization """
normalizing_values_CV = {}
normalizing_values_CV['mean'] = df_train_CV[features_CV].mean(axis=0)
normalizing_values_CV['std'] = df_train_CV[features_CV].std(axis=0)
normalizing_values_CV['min'] = df_train_CV[features_CV].min(axis=0)
normalizing_values_CV['max'] = df_train_CV[features_CV].max(axis=0)

normalizing_values_MV = {}
normalizing_values_MV['mean'] = df_train_MV[features_MV].mean(axis=0)
normalizing_values_MV['std'] = df_train_MV[features_MV].std(axis=0)
normalizing_values_MV['min'] = df_train_MV[features_MV].min(axis=0)
normalizing_values_MV['max'] = df_train_MV[features_MV].max(axis=0)

# normalizing the continuous variables
df_train_CV_proc = normalization(df_train_CV, 'mean_std', normalizing_values_CV, features_CV)
df_holdout_CV_proc = normalization(df_holdout_CV, 'mean_std', normalizing_values_CV, features_CV)

df_train_MV_proc = normalization(df_train_MV, 'mean_std', normalizing_values_MV, features_MV)
df_holdout_MV_proc = normalization(df_holdout_MV, 'mean_std', normalizing_values_MV, features_MV)

""" # true permutation matrix  """

P_x1 = np.zeros((len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))

print("Shape of P_x1 ", P_x1.shape)

for i in range(len(df_train_CV_proc.columns)):
    for j in range(len(df_train_MV_proc.columns)):
        if df_train_CV_proc.columns[i] == df_train_MV_proc.columns[j]:
            P_x1[i,j]=1
        elif df_train_CV_proc.columns[i] in CV_itemids_with_match:
            if (match_dic[df_train_CV_proc.columns[i]]== df_train_MV_proc.columns[j]):
                P_x1[i,j]=1
        elif (df_train_CV_proc.columns[i] in CV_itemids_withnomatch )& (df_train_MV_proc.columns[j] in MV_itemids_withnomatch):
            #print("counter for both wothout match")
            P_x1[i,j]=2
            #ctr_with_no_match = ctr_with_no_match + 1


# generating the correlation matrix for true matches
Cor_btw_df = np.zeros((len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))
num_obs_for_cor = min(len(df_train_CV_proc), len(df_train_MV_proc))
for i in range(len(df_train_CV_proc.columns)):
    for j in range(len(df_train_MV_proc.columns)):
        temp = stats.pearsonr(df_train_CV_proc.values[:num_obs_for_cor,i], df_train_MV_proc.values[:num_obs_for_cor, j])
        Cor_btw_df[i,j] = temp[0]

# converting the correlation matrix to a dataframe
Cor_df = pd.DataFrame(Cor_btw_df,
                 index=df_train_CV_proc.columns,
                 columns=df_train_MV_proc.columns)

# file saving logistics
saving_dir = './ChimericAE_Final_'+str(datetime.date.today()) +  '/#ofhidden_layers_' + str(num_of_hidden_layers) + "/MIMIC_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) +"hyperparameter_tuning"

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

# # calling the Kang function
# correct_with_match_from_x1_test_Kang = Kang_MI_HC_opt(
#                     df_train_CV_proc.copy(), df_train_MV_proc.copy(), P_x1[len(list_lab_ids):,
#                                                                        len(list_lab_ids):], df_train_CV_proc.columns, df_train_MV_proc.columns, list_lab_ids, Cor_df, df_holdout_CV_proc, df_holdout_MV_proc)


# calling the KMF function only once because the hyper parameters are only for
correct_with_match_from_x1_test_sim_cor, correct_with_match_from_x2_test_sim_cor, correct_with_no_match_from_CCx1_test_sim_cor, correct_with_no_match_from_CCx2_test_sim_cor, temp_infer_from_x1_sim_cor, temp_infer_from_x2_sim_cor, mapp_fea_to_add, mapp_fea_to_add_match, F1_fromx1_simcor, F1_fromx2_simcor = Simple_maximum_sim_viaCorrelation(
                    df_train_CV_proc.copy(), df_train_MV_proc.copy(), P_x1[len(list_lab_ids):,
                                                                       len(list_lab_ids):],
                    df_train_CV_proc.columns, df_train_MV_proc.columns, list_lab_ids, Cor_df, df_holdout_CV_proc,
                    df_holdout_MV_proc)


""" Preparing for the second stage """

list_lab_ids_chart_updated_CV = list_lab_ids + mapp_fea_to_add
list_lab_ids_chart_updated_MV = list_lab_ids + mapp_fea_to_add_match

remaining_unmapped_feature_CV = [col for col in df_train_CV_proc.columns if
                                   col not in list_lab_ids_chart_updated_CV]
remaining_unmapped_feature_MV = [col for col in df_train_MV_proc.columns if
                                col not in list_lab_ids_chart_updated_MV]

reordered_column_names_updated_CV = list_lab_ids_chart_updated_CV + remaining_unmapped_feature_CV
reordered_column_names_updated_MV = list_lab_ids_chart_updated_MV + remaining_unmapped_feature_MV

df_train_CV_proc = df_train_CV_proc.reindex(columns= reordered_column_names_updated_CV)
df_holdout_CV_proc = df_holdout_CV_proc.reindex(columns= reordered_column_names_updated_CV)
df_train_MV_proc = df_train_MV_proc.reindex(columns= reordered_column_names_updated_MV)
df_holdout_MV_proc = df_holdout_MV_proc.reindex(columns= reordered_column_names_updated_MV)

# update permutation matrix
P_x1_updated = np.zeros((len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))

print("Shape of P_x1 ", P_x1_updated.shape)

for i in range(len(df_train_CV_proc.columns)):
    for j in range(len(df_train_MV_proc.columns)):
        if df_train_CV_proc.columns[i] == df_train_MV_proc.columns[j]:
            P_x1_updated[i,j]=1
        elif df_train_CV_proc.columns[i] in CV_itemids_with_match:
            if (match_dic[df_train_CV_proc.columns[i]]== df_train_MV_proc.columns[j]):
                P_x1_updated[i,j]=1
        elif (df_train_CV_proc.columns[i] in CV_itemids_withnomatch )& (df_train_MV_proc.columns[j] in MV_itemids_withnomatch):
            #print("counter for both wothout match")
            P_x1_updated[i,j]=2
            #ctr_with_no_match = ctr_with_no_match + 1

# filename_for_saving_PM_quality = "/MIMIC_cont_data_PM_quality_part#_" +"_#mfeat_"+str(len(list_lab_ids))+"_L_dim_"+str(hidden_dim)+"feat_inLargerDB_"+str(len(reordered_column_names_updated_MV))


file_name_fromx1 = saving_dir + "/" + "F1_metric_fromX1_Real_data_orthoStatus_" + str(
    orthogonalization_type) + "_BNStatus_" + str(batchnorm) + "2layer.txt"

if os.path.exists(file_name_fromx1):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_fromx1)

f = open(file_name_fromx1, 'w')
f.write("\n \n *** Chimeric AE Present file settings ***")
f.write("\n \n MIMIC  data HP tuning F1 values from X1 exhaustive")
f.write("\n Orthogonalization status {0}\t ".format(orthogonalization_type))
f.write("\n Batch norm {0}\t ".format(batchnorm))
#f.write("\n Size of L {0}\t".format(hidden_dim))
#f.write("\n Weight for direct AE loss {0}\t ".format(weight_direct))
#f.write("\n Weight for cross AE loss {0}\t ".format(weight_cross))
f.write("\n Number of epochs {0}\t".format(epochs))
#f.write("\n Starting learning rate {0}\t ".format(learning_rate))
#f.write("\n Batch size {0}\t".format(batch_size))
f.write("\n")
f.close()

file_name_fromx2 = saving_dir + "/" + "F1_metric_fromX2_Real_data_orthoStatus_" + str(
    orthogonalization_type) + "_BNStatus_" + str(batchnorm) + "2layer.txt"

if os.path.exists(file_name_fromx2):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_fromx2)

f = open(file_name_fromx2, 'w')
f.write("\n \n *** Chimeric AE Present file settings ***")
f.write("\n \n MIMIC  data HP tuning F1 values from X2 exhaustive")
f.write("\n Orthogonalization status {0}\t ".format(orthogonalization_type))
f.write("\n Batch norm {0}\t ".format(batchnorm))
#f.write("\n Size of L {0}\t".format(hidden_dim))
#f.write("\n Weight for direct AE loss {0}\t ".format(weight_direct))
#f.write("\n Weight for cross AE loss {0}\t ".format(weight_cross))
f.write("\n Number of epochs {0}\t".format(epochs))
#f.write("\n Starting learning rate {0}\t ".format(learning_rate))
#f.write("\n Batch size {0}\t".format(batch_size))
f.write("\n")
f.close()


F1_holdout_over_Dataset_samples_X1_tr_ump = np.zeros(len(weight_direct_list)*len(weight_cross_list)*len(batch_size_list)*len(dropout_rate_list)*len(hidden_dim_list)*len(learning_rate_list))
F1_holdout_over_Dataset_samples_X2_tr_ump = np.zeros(len(weight_direct_list)*len(weight_cross_list)*len(batch_size_list)*len(dropout_rate_list)*len(hidden_dim_list)*len(learning_rate_list))

hyperparameter_list =[]
ctr = 0 # counter to keep track of all possible hyperparamter combinations
for weight_direct, weight_cross, batch_size, dropout_rate, hidden_dim, learning_rate in itertools.product(weight_direct_list, weight_cross_list, batch_size_list, dropout_rate_list, hidden_dim_list, learning_rate_list):
    print(" Hyperparameter values wd, wc, btsize, p, l, lr", weight_direct, weight_cross, batch_size, dropout_rate, hidden_dim, learning_rate)
    hyperparameter_list.append((weight_direct, weight_cross, batch_size, dropout_rate, hidden_dim, learning_rate))
    print("\n ********************************************************")
    print(" \n Run STARTS for hyperparameter tuple ", (weight_direct, weight_cross, batch_size, dropout_rate, hidden_dim, learning_rate), "\n")
    print(" ******************************************************** \n")

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_infer_from_x1, temp_infer_from_x2, F1_fromx1, F1_fromx2 = \
    Train_cross_AE(df_train_CV_proc.copy(), df_train_MV_proc.copy(), P_x1_updated[len(list_lab_ids):,
                                      len(list_lab_ids):], reordered_column_names_updated_CV, reordered_column_names_updated_MV, list_lab_ids,list_lab_ids_chart_updated_CV, list_lab_ids_chart_updated_MV, Cor_df, df_holdout_CV_proc, df_holdout_MV_proc, P_x1)

    F1_holdout_over_Dataset_samples_X1_tr_ump[ctr] = F1_fromx1
    F1_holdout_over_Dataset_samples_X2_tr_ump[ctr] = F1_fromx2


    f = open(file_name_fromx1,'a')
    f.write("{0}".format(F1_holdout_over_Dataset_samples_X1_tr_ump[ctr]))
    f.write("\n ")
    f.close()

    f = open(file_name_fromx2,'a')
    f.write("{0}".format(F1_holdout_over_Dataset_samples_X2_tr_ump[ctr]))
    f.write("\n ")
    f.close()


    ctr= ctr + 1
    print("\n ********************************************************")
    print(" \n Run ENDS for hyperparameter tuple ", (weight_direct, weight_cross, batch_size, dropout_rate, hidden_dim, learning_rate), "\n")
    print(" ******************************************************** \n")


a = itertools.product(weight_direct_list, weight_cross_list, batch_size_list, dropout_rate_list, hidden_dim_list, learning_rate_list)

textfile = open("hyperparameterlist_file_" + str(len(hyperparameter_list)) +"case.txt", "w")

for element in a:

    textfile.write(str(element) + "\n")

textfile.close()

    
# comparison between F1 from simcor and two stage Chimeric
print(" F1 for KMF (from X1)", F1_fromx1_simcor)
print(" F1 for KMF (from X2)", F1_fromx2_simcor, "\n")

idx_from_X1 = np.argmax(F1_holdout_over_Dataset_samples_X1_tr_ump)
idx_from_X2 = np.argmax(F1_holdout_over_Dataset_samples_X2_tr_ump)
print(" Hyperparameter maximizing F1 (from X1) ", hyperparameter_list[idx_from_X1])
print(" Hyperparameter maximizing F1 (from X2) ", hyperparameter_list[idx_from_X2])
print(" Possible F1 for 2 stage Chimeric (from X1)", F1_holdout_over_Dataset_samples_X1_tr_ump[idx_from_X1], F1_holdout_over_Dataset_samples_X1_tr_ump[idx_from_X2])
print(" Possible F1 for 2 stage Chimeric (from X2)", F1_holdout_over_Dataset_samples_X2_tr_ump[idx_from_X1], F1_holdout_over_Dataset_samples_X2_tr_ump[idx_from_X2])

# plotting
x_axis = np.arange(ctr)

plt.plot(x_axis, F1_holdout_over_Dataset_samples_X1_tr_ump, label=" # mapped_feature "+str(len(list_lab_ids)))
plt.xticks(x_axis, np.array(hyperparameter_list), rotation = 65, fontsize = 4)
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Different hyperparameter combination")
plt.ylabel("F1 score")
plt.title("MIMIC Results from X1 cross correlation matrix (UnMapped)")
plt.legend()
plt.savefig(saving_dir + "/Hyper_X1_tr_MIMIC_" + str(len(hyperparameter_list)) + ".pdf", bbox='tight')
plt.savefig(saving_dir + "/Hyper_X1_tr_MIMIC_" + str(len(hyperparameter_list)) + ".png", bbox='tight')
plt.close()

plt.plot(x_axis, F1_holdout_over_Dataset_samples_X2_tr_ump, label=" # mapped_feature "+str(len(list_lab_ids)))
plt.xticks(x_axis, np.array(hyperparameter_list), rotation = 65, fontsize = 4)
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Different hyperparameter combination")
plt.ylabel("F1 score")
plt.title("MIMIC Results from X2 cross correlation matrix (UnMapped)")
plt.legend()
plt.savefig(saving_dir + "/Hyper_X2_tr_MIMIC" + str(len(hyperparameter_list)) + ".pdf", bbox='tight')
plt.savefig(saving_dir + "/Hyper_X2_tr_MIMIC" + str(len(hyperparameter_list)) + ".png", bbox='tight')
plt.close()

