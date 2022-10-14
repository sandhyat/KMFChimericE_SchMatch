"""

Code for matching features based on finger prints. USes Gale Shapley after the correlation has been computed.

Input:

Output:

"""

# TODO module to remove the repetitive labs that exist in chart events. In my wokr, I had simply used the correlation among features with a threshold of 0.97



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
print(sys.getrecursionlimit())
sys.setrecursionlimit(3500)
print(sys.getrecursionlimit())

def Covaraince_matrix_factor_method(dimension, factors):
    n = dimension
    k = factors

    W = np.random.standard_normal(size = (n,k))  # k< n and hence not full rank
    print("W matrix")
    # print(W)
    list_random = np.random.randint(1,20, n)
    # list_random = np.random.randint(100,200, n)  # only for dataset 8
    print("List of random numbers used to generate the diagonal matrix for covariance")
    print(list_random)
    # exit()
    D = np.identity(n)
    D[np.diag_indices_from(D)] = list_random
    cov = np.matmul(W, np.transpose(W)) + D  # adding D to make the matrix full rank

    return cov

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


def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc, P_x1
                                      , reordered_column_names_orig, reordered_column_names_r,
                                      mapped_features, Cor_from_df, Df_holdout_orig, DF_holdout_r):
    """
    :param df_train_preproc:  dataset 1 training set
    :param df_rename_preproc: dataset 2 trainng set
    :param P_x1:  true permutation matrix between only unmatched features
    :param reordered_column_names_orig:  column names of dataset 1
    :param reordered_column_names_r: column names of dataset 2
    :param mapped_features: names of mapped features common between dataset 1 nd dataset 2
    :param Cor_from_df:correlation between the features of two datasets (includes both mapped and unmapped)
    :param Df_holdout_orig: dataset 1 holdout set for bootstrapping
    :param DF_holdout_r: dataset 2 holdout set for bootstrapping
    :return:
    """


    mpfeatures = len(mapped_features)
    unmapped_features_orig = [ i for i in reordered_column_names_orig if i not in mapped_features]
    unmapped_features_r = [ i for i in reordered_column_names_r if i not in mapped_features]

    # device = torch.device('cuda')
    # # computing the correlation matrix between original feature values and cross reconstruction
    # CorMatrix_X1_unmap_mapped = np.zeros((unmapped_features_orig, mpfeatures))
    # CorMatrix_X2_unmap_mapped = np.zeros((unmapped_features_r, mpfeatures))
    # CorMatrix_X1_unmap_mapped_P_value = np.zeros((unmapped_features_orig, mpfeatures))
    # CorMatrix_X2_unmap_mapped_P_value = np.zeros((unmapped_features_r, mpfeatures))
    #
    # for i in range(unmapped_features_orig):
    #     for j in range(mpfeatures):
    #         temp = stats.pearsonr(df_train_preproc.values[:, mpfeatures + i], df_train_preproc.values[:, j])
    #         CorMatrix_X1_unmap_mapped[i, j] = temp[0]
    #         CorMatrix_X1_unmap_mapped_P_value[i, j] = temp[1]
    #
    # for i in range(unmapped_features_r):
    #     for j in range(mpfeatures):
    #         temp = stats.pearsonr(df_rename_preproc.values[:, mpfeatures + i], df_rename_preproc.values[:, j])
    #         CorMatrix_X2_unmap_mapped[i, j] = temp[0]
    #         CorMatrix_X2_unmap_mapped_P_value[i, j] = temp[1]

    CorMatrix_X1_unmap_mapped = df_train_preproc.corr().loc[unmapped_features_orig, mapped_features]
    CorMatrix_X2_unmap_mapped = df_rename_preproc.corr().loc[unmapped_features_r, mapped_features]


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
    num_of_bts = 10
    bts_for_allthe_accepted_matches_fromX1 = np.zeros((len(unmapped_features_orig), num_of_bts))
    bts_for_allthe_accepted_matches_fromX2 = np.zeros((len(unmapped_features_orig), num_of_bts))

    for bts in range(num_of_bts):
        Df_holdout_orig_bts = Df_holdout_orig.sample(n=len(Df_holdout_orig), replace=True, random_state=bts, axis=0)
        DF_holdout_r_bts = DF_holdout_r.sample(n=len(DF_holdout_r), replace=True, random_state=bts, axis=0)
        # CorMatrix_X1_unmap_mapped_bts = np.zeros((unmapped_features_orig, mpfeatures))
        # CorMatrix_X2_unmap_mapped_bts = np.zeros((unmapped_features_r, mpfeatures))
        #
        # for i in range(unmapped_features_orig):
        #     for j in range(mpfeatures):
        #         temp = stats.pearsonr(Df_holdout_orig_bts.values[:, mpfeatures + i], Df_holdout_orig_bts.values[:, j])
        #         CorMatrix_X1_unmap_mapped_bts[i, j] = temp[0]
        #
        # for i in range(unmapped_features_r):
        #     for j in range(mpfeatures):
        #         temp = stats.pearsonr(DF_holdout_r_bts.values[:, mpfeatures + i], DF_holdout_r_bts.values[:, j])
        #         CorMatrix_X2_unmap_mapped_bts[i, j] = temp[0]

        CorMatrix_X1_unmap_mapped_bts = Df_holdout_orig_bts.corr().loc[unmapped_features_orig, mapped_features]
        CorMatrix_X2_unmap_mapped_bts = DF_holdout_r_bts.corr().loc[unmapped_features_r, mapped_features]

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
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", 'true_correlation',
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
        # temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        # temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
        #     int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
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
        # temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        # temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
        #     int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
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
          len(unmapped_features_orig) - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          len(unmapped_features_orig) - num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number",
          len(unmapped_features_orig) - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          len(unmapped_features_orig) - num_xtra_feat_inX1)

    print("\n Mistakes by the significance testing algorithm on holdout data")
    print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)


    print(" -------- Sim_Correlation  methods  ends ------------- \n \n  ")

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

    print( "Sim cor F values ", F1_fromx1, F1_fromx2)


    return correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2



""" some initial things related to MIMIC """
if False:

    # Getting list of all items along with the source and label
    item_id_dbsource = pd.read_csv('/d_items_chartevents.csv')
    itemid_labs = pd.read_csv('/d_items_labevents.csv')


    # creating label dictionary for later
    itemid_label_lab = dict(zip(list(itemid_labs.itemid), list(itemid_labs.label)))
    itemid_label_chart = dict(zip(list(item_id_dbsource.itemid), list(item_id_dbsource.label)))
    itemid_label_dict = {**itemid_label_chart, **itemid_label_lab}  # merging two dictionaries here

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

    """ something to fill inbetween to take care of the duplicate labs in chartevents """

    CV_itemids_with_match = list(match_df['CV_itemids'])
    MV_itemids_with_match = list(match_df['MV_itemids'])

    #final matching dict
    match_dic = dict(zip(CV_itemids_with_match, MV_itemids_with_match))

    # itemids with no match
    CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]  # onlychart_cont_CV is the set of chartevents from CV
    MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]  # onlychart_cont_MV is the set of chartevents from MV

    print( " CV_itemids_with match ", len(CV_itemids_with_match))
    print( " MV_itemids_with match ", len(MV_itemids_with_match))

    print( " CV_itemids_with NO match ", len(CV_itemids_withnomatch))
    print( " MV_itemids_with NO match ", len(MV_itemids_withnomatch))

    num_xtra_feat_inX1 = len(CV_itemids_withnomatch)

    """ # true permutation matrix  """

    P_x1 = np.zeros((len(df_train_preproc.columns), len(df_rename_preproc.columns)))  # df_train_prepoc is the training part of dataset 1 and df_rename_prepproc is the train part of dataset 2

    print("Shape of P_x1 ", P_x1.shape)

    for i in range(len(df_train_preproc.columns)):
        for j in range(len(df_rename_preproc.columns)):
            if df_train_preproc.columns[i] == df_rename_preproc.columns[j]:
                P_x1[i, j] = 1
            elif df_train_preproc.columns[i] in CV_itemids_with_match:
                if (match_dic[df_train_preproc.columns[i]] == df_rename_preproc.columns[j]):
                    P_x1[i, j] = 1
            elif (df_train_preproc.columns[i] in CV_itemids_withnomatch) & (
                    df_rename_preproc.columns[j] in MV_itemids_withnomatch):
                P_x1[i, j] = 2


if True:
    """  test example  """

    num_features = 7
    num_samples = 5000
    num_factors = 3
    num_xtra_feat_inX1 = 0


    mean_pos = np.random.randint(10, 20, num_features)
    mean_neg = np.random.randint(10, 20, num_features)

    cov = Covaraince_matrix_factor_method(num_features,num_factors)

    dataset1_pos_train = np.random.multivariate_normal(mean_pos, cov, num_samples)
    dataset1_neg_train = np.random.multivariate_normal(mean_neg, cov,num_samples)

    dataset1 = pd.DataFrame(np.concatenate([dataset1_pos_train, dataset1_neg_train]), columns=[5,2, 0,1,3,4,6])

    dataset2_pos_train = np.random.multivariate_normal(mean_pos, cov, num_samples)
    dataset2_neg_train = np.random.multivariate_normal(mean_neg, cov, num_samples)

    dataset2 = pd.DataFrame(np.concatenate([dataset2_pos_train, dataset2_neg_train]), columns=[5,2, 0,1,3,4,6])

    dataset2 = dataset2[[5,2, 6,0,1,3,4]]


    dataset1_tr, dataset1_holdout = model_selection.train_test_split(dataset1, test_size=0.2, random_state=42)
    dataset2_tr, dataset2_holdout = model_selection.train_test_split(dataset2, test_size=0.2, random_state=42)


    mapped_features = [5,2]

    Cor_btw_df = np.zeros((len(dataset1_tr.columns), len(dataset2_tr.columns)))
    for i in dataset2_tr.columns:
        Cor_btw_df[:,i] = dataset1_tr.corrwith(dataset2_tr[i])
    Cor_df = pd.DataFrame(Cor_btw_df,
                     index=dataset1_tr.columns,
                     columns=dataset2_tr.columns)



    P_x1 = np.zeros((len(dataset1_tr.columns), len(dataset2_tr.columns)))
    for i in range(len(dataset1_tr.columns)):
        for j in range(len(dataset2_tr.columns)):
            if dataset1_tr.columns[i] == dataset2_tr.columns[j]:
                P_x1[i, j] = 1

    # inducing nans
    dataset1_tr.iloc[100:200, 3] = np.NaN
    dataset2_tr.iloc[300:500, 1] = np.NaN


    ## calling the KMF function

    Simple_maximum_sim_viaCorrelation(dataset1_tr.copy(), dataset2_tr.copy(), P_x1[len(mapped_features):,
                                                                                           len(mapped_features):],
                                        dataset1_tr.columns, dataset2_tr.columns, mapped_features, Cor_df, dataset1_holdout,
                                        dataset2_holdout)

# Simple_maximum_sim_viaCorrelation(df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):,
#                                                                                        len(mapped_features):],
#                                     df_train_preproc.columns, df_rename_preproc.columns, mapped_features, Cor_df, DF_holdout_orig,
#                                     Df_holdout_r)