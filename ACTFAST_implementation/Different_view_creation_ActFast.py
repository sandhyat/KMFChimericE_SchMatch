"""
This file is to prepare the different views that are available for the two eras of the  ActFast datasets : procedure text bow, preops, preops labs, outcomes

Notes about the preprocessing
1) Ignore categorical labs, there are only 3 with matches so ignoring them. In MV not reading the nonnumeric file; in epic dropping these three labs
2) Mse for all setups: after min-maxing the binary variables they should be good.
3) drop the common labs between the preops and labs from metavision (atleast 4 or 5 of them)
4) For the categorical variables, the levels have been mapped so they will be treated as individual features now
5) In metavision, DVT and PE are separate in the features. In Epic DVT, PE and DVT_PE are three separate outcomes (and features too) and the latter is derived from the first two so can be dropped
6) Dropped the 'MUCOUS, URINE' variable for now
7) Didn't process the embedded representation at all (only filled the NAs with 0, rest is the same). ## TODO: Maybe should process this?
8) For the outcomes, after getting them in the train and test format: for binary use the fillna and then min-max normalization, for continuous mean imp and then mean-std scaling
9) For ICULos and crit_care_time, where the missing is basically for the patients that were not admitted to the icu it will be 0
10) In MV, dropping Disposition variable as it takes 4 values and is related to icu_status so at the mapping time it is anyway going to be only 1 of them.
11) Need to check the difference between endtime and case_duration ## TODO:
13) The mvlabs that are transformed to binary are treated as binary. (Email thread: Do you need 1 more experiment for schema matching?)
14) Dropping the Coombs_Lab from both the eras
15) 'Secondary Diagnosis' is treated as continuous even though in the current data it only takes 0 and 15 as the value




## other mapping facts
    1) ARF is the same concept as postop_vent_duration > 2days
    2) unplanned_icu is the same concept as admit_day > .029 = 7hrs
    3) In survival_times , -inf is the min of a zero length group, so patient with no recorded event (survived to end of study)

"""

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
from sklearn.model_selection import train_test_split
from scipy import linalg, stats
import xgboost as xgb
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
# from datetime import datetime
import json, sys, argparse
from pyarrow import feather


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

def mv_labs_proc_module(data_all, train_idx, test_idx):

    mv_ordinal = list(pd.read_csv(data_dir + '/mv/ordinal_vars_mv.txt', delimiter= "\t",header=None)[0])
    mv_ordinal = [i for i in mv_ordinal if (i in data_all.columns)]

    # mv_categorical_labs_to_drop = ['ABO_AND_RHO[D]_TYPING_P3-20220', 'CLARITY_URINE_P3-90001Y', 'COLOR_URINE_P3-90002Y']  # not needed because the categoricals are separate

    # setting the caseids as index
    data_all = data_all.set_index('caseid')
    data_all.drop(columns=['COOMBS_TEST_INDIRECT_NOS_P3-22020'], inplace =True) # dropping this since Cooombs_Lab is not in the combined epic preops lab file


    mv_binary = []
    for i in data_all.columns:
        if len(data_all[i].value_counts().index) == 2 and (i not in mv_ordinal):
            mv_binary.append(i)

    # this round about thing is being done because not all preops have all the labs
    temp_df_train = pd.DataFrame(columns=data_all.columns)
    temp_df_train['caseid'] = list(set(train_idx).difference(data_all.index))
    data_all = data_all.reset_index()
    temp_df_train = pd.concat([data_all, temp_df_train], axis=0)
    train = temp_df_train.set_index('caseid').loc[train_idx]


    temp_df_test = pd.DataFrame(columns=data_all.columns)
    temp_df_test['caseid'] = list(set(test_idx).difference(data_all.set_index('caseid').index))
    temp_df_test = pd.concat([data_all, temp_df_test], axis=0)
    test = temp_df_test.set_index('caseid').loc[test_idx]

    # we had done a reset in between
    data_all = data_all.set_index('caseid')
    mv_continuous = [ i for i in data_all.columns if i not in (mv_ordinal+mv_binary)]


    meta_Data = {}

    # currently not dropping any of the labs even though some of them are missing a lot (above 90%)

    train[mv_continuous] = train[mv_continuous].fillna(train[mv_continuous].mean())
    test[mv_continuous] = test[mv_continuous].fillna(train[mv_continuous].mean()) # this is weird but inplace nan filling was not working correctly so had to do explict assignment
    # train[mv_continuous].fillna(train[mv_continuous].mean(), inplace=True)  ## warning about copy
    # test[mv_continuous].fillna(train[mv_continuous].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in mv_continuous:
        if train[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in mv_continuous]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = mv_continuous
    normalizing_values_cont['mean'] = list(train[mv_continuous].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[mv_continuous].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[mv_continuous].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[mv_continuous].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, mv_continuous)
    test = normalization(test, 'mean_std', normalizing_values_cont, mv_continuous)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # imputing

    train[mv_ordinal] = train[mv_ordinal].fillna(train[mv_ordinal].median())
    test[mv_ordinal] = test[mv_ordinal].fillna(train[mv_ordinal].median())

    meta_Data["train_median_ord"] = [train[i].median() for i in mv_ordinal]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = mv_ordinal
    normalizing_values_ord['mean'] = list(train[mv_ordinal].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[mv_ordinal].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[mv_ordinal].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[mv_ordinal].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, mv_ordinal)
    test = normalization(test, 'mean_std', normalizing_values_ord, mv_ordinal)

    meta_Data['norm_value_ord'] = normalizing_values_ord


    # imputing binary variables

     ## this random thing is needed because somewhere when I was creating the train and test df, these variables decided to flip to object type
    for i in train.columns:
        if train[i].dtype == 'O':
            train[i] = train[i].astype('float')
        if test[i].dtype == 'O':
            test[i] = test[i].astype('float')

    for i in mv_binary: # this is again very random. Phew fillna has got to be crazy!!!!!
        temp_mode = train[i].mode()[0]
        train[i].fillna(temp_mode, inplace=True)
        test[i].fillna(temp_mode, inplace=True)

    # train[mv_binary] = train[mv_binary].fillna(train[mv_binary].mode())
    # test[mv_binary] = test[mv_binary].fillna(train[mv_binary].mode())

    meta_Data["train_median_bin"] = [train[i].mode()[0] for i in mv_binary]

    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = mv_binary
    normalizing_values_bin['mean'] = list(train[mv_binary].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[mv_binary].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[mv_binary].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[mv_binary].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, mv_binary)
    test = normalization(test, 'min_max', normalizing_values_bin, mv_binary)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")


    meta_Data["ordinal_variables"] = mv_ordinal
    meta_Data["continuous_variables"] = mv_continuous
    meta_Data["binary_variables"] = mv_binary


    output_file_name = data_dir + '/mv/labs_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    print("hola")

    return train, test

def mv_preops_proc_module(data_all, test_size=0.2):
    # these were added seperately and we are already using them as seperate view so removing here
    features_frm_freetxt = ['txwv' + str(i) for i in range(1,
                                                           51)]
    features_frm_diagncode = ['dxwv' + str(i) for i in range(1, 51)]
    data_all.drop(columns=features_frm_freetxt + features_frm_diagncode, inplace=True)

    # including the cases where atleast one preop evaluation is valid; dropping this criterion too
    # index_preop = data_all.loc[(data_all['neval_valid'] > 0)].index  # removed the blank_preop filter [data_all['blank_preop'] == 0]
    # data_all = data_all.loc[index_preop]
    data_all.drop(columns=['Location', 'age_missing', 'case_duration', 'Intubation', 'blank_preop', 'neval', 'neval_valid'],
                   inplace=True)  # dropping the neval and neval_Valid as they are basically counts as to how many evaluations were done and how many of them valid

    # hardcoding the categorical and ordinal variables
    mv_categorical = ["Anesthesia_Type","CPAP.Usage", "PAP_Type", "Surg_Type", "SEX", "RACE"]
    mv_ordinal = list(pd.read_csv(data_dir + '/mv/ordinal_vars_mv.txt', delimiter= "\t",header=None)[0])
    mv_ordinal = [i for i in mv_ordinal if (i in data_all.columns)]

    # Originally this variable was object type so needed to convert
    data_all.loc[data_all['CHF_Diastolic_Function'] == 'Other', 'CHF_Diastolic_Function'] = 0
    data_all['CHF_Diastolic_Function'] = data_all['CHF_Diastolic_Function'].astype('int')

    # Originally Sex variable has the two values in string format too so fixing that and droping the other category due to small number
    # data_all.drop(index=data_all[data_all['SEX'] == 'Other'].index, inplace=True)  # not dropping it anymore and making it a third category
    data_all.loc[data_all['SEX'] == 'Other', 'SEX'] = 3
    data_all['SEX'] = data_all['SEX'].astype('int')

    # rescaling the Dialysis_History variable from {1,2} --> {0,1}; it shouldn't matter because we are min-max scaling the binaries
    data_all.loc[data_all['Dialysis_History'] == 1, 'Dialysis_History'] = 0
    data_all.loc[data_all['Dialysis_History'] == 2, 'Dialysis_History'] = 1

    # setting the caseids as index
    data_all = data_all.set_index('caseid')

    mv_binary = []
    mv_continuous = ['StopBang_Pressure', 'StopBang_Tired', 'StopBang_Total', 'StopBang_Observed', 'StopBang_Snore']

    for a in data_all.columns:
        if data_all[a].dtype == 'int32' or data_all[a].dtype == 'int64':
            if len(data_all[a].unique()) < 10 and len(data_all[a].unique()) > 2 and (a not in mv_ordinal+mv_categorical+mv_continuous):
                data_all[a] = data_all[a].astype('category')
                mv_categorical.append(a)
        if len(data_all[a].unique()) <= 2 and (a not in mv_ordinal+mv_categorical+mv_continuous):
            mv_binary.append(a)

    mv_categorical = [*set(mv_categorical)]
    mv_continuous = mv_continuous + [i for i in data_all.columns if i not in ( mv_binary + mv_categorical +mv_ordinal)]

    mv_continuous = [*set(mv_continuous)]

    for name in mv_categorical:
        data_all[name] = data_all[name].astype('category')

    # one hot encoding
    meta_Data = {}

    meta_Data["levels"] = {}

    preops_ohe = data_all.copy()
    preops_ohe.drop(columns=mv_categorical, inplace=True)
    import itertools
    encoded_variables = list()
    for i in mv_categorical:
        meta_Data["levels"][i] = list(data_all[i].cat.categories)
        temp = pd.get_dummies(data_all[i], dummy_na=True, prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))


    # creating the train test partitioning
    upto_test_idx = int(test_size * len(preops_ohe))
    test = preops_ohe.iloc[:upto_test_idx]
    train = preops_ohe.iloc[upto_test_idx:]
    # train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state)

    train_index = train.index
    test_index = test.index

    # NOTE: this dataset preops do not have any missing values

    #scaling continuous variables
    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = mv_continuous
    normalizing_values_cont['mean'] = list(train[mv_continuous].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[mv_continuous].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[mv_continuous].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[mv_continuous].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, mv_continuous)
    test = normalization(test, 'mean_std', normalizing_values_cont, mv_continuous)
    meta_Data['norm_value_cont'] = normalizing_values_cont


    # scaling ordinal variables
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = mv_ordinal
    normalizing_values_ord['mean'] = list(train[mv_ordinal].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[mv_ordinal].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[mv_ordinal].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[mv_ordinal].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_ord, mv_ordinal)
    test = normalization(test, 'mean_std', normalizing_values_ord, mv_ordinal)
    meta_Data['norm_value_ord'] = normalizing_values_ord

    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = mv_binary
    normalizing_values_bin['mean'] = list(train[mv_binary].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[mv_binary].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[mv_binary].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[mv_binary].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, mv_binary)
    test = normalization(test, 'min_max', normalizing_values_bin, mv_binary)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["encoded_var"] = encoded_variables

    meta_Data["binary_var_name"] = mv_binary

    meta_Data["categorical_name"] = mv_categorical
    meta_Data["ordinal_variables"] = mv_ordinal
    meta_Data["continuous_variables"] = mv_continuous
    meta_Data["column_all_names"] = list(preops_ohe.columns)

    output_file_name = data_dir + '/mv/preops_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    print("hola")

    return train, test, train_index, test_index

def mv_outcomes_proc_module(data_all, train_idx, test_idx):

    # setting the caseids as index
    data_all = data_all.set_index('caseid')
    data_all.drop(columns = ['disposition'], inplace=True)

    # filling the ICULoS variable beforehand with 0
    data_all['ICULoS'].fillna(0, inplace=True)
    data_all['crit_care_time'].fillna(0, inplace=True)

    # replacing True and False by 1 and 0
    data_all.replace(regex={False: 0, True: 1}, inplace=True) # these are already bool so they shouldn't be in the strings


    mv_binary = []
    for i in data_all.columns:
        if len(data_all[i].value_counts().index) == 2:
            mv_binary.append(i)

    # this round about thing is being done because not all preops have all the labs; this is futile after dropping the blank_preop filter but letting it stay.
    temp_df_train = pd.DataFrame(columns=data_all.columns)
    temp_df_train['caseid'] = list(set(train_idx).difference(data_all.index))
    data_all = data_all.reset_index()
    temp_df_train = pd.concat([data_all, temp_df_train], axis=0)
    train = temp_df_train.set_index('caseid').loc[train_idx]


    temp_df_test = pd.DataFrame(columns=data_all.columns)
    temp_df_test['caseid'] = list(set(test_idx).difference(data_all.set_index('caseid').index))
    temp_df_test = pd.concat([data_all, temp_df_test], axis=0)
    test = temp_df_test.set_index('caseid').loc[test_idx]

    # we had done a reset in between
    data_all = data_all.set_index('caseid')
    mv_continuous = [ i for i in data_all.columns if i not in (mv_binary)]


    meta_Data = {}



    train[mv_continuous] = train[mv_continuous].fillna(train[mv_continuous].mean())
    test[mv_continuous] = test[mv_continuous].fillna(train[mv_continuous].mean()) # this is weird but inplace nan filling was not working correctly so had to do explict assignment
    # train[mv_continuous].fillna(train[mv_continuous].mean(), inplace=True)  ## warning about copy
    # test[mv_continuous].fillna(train[mv_continuous].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in mv_continuous:
        if train[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in mv_continuous]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = mv_continuous
    normalizing_values_cont['mean'] = list(train[mv_continuous].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[mv_continuous].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[mv_continuous].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[mv_continuous].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, mv_continuous)
    test = normalization(test, 'mean_std', normalizing_values_cont, mv_continuous)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    for i in mv_binary: # this is again very random. Phew fillna has got to be crazy!!!!!
        temp_mode = train[i].mode()[0]
        train[i].fillna(temp_mode, inplace=True)
        test[i].fillna(temp_mode, inplace=True)


    meta_Data["train_median_bin"] = [train[i].mode()[0] for i in mv_binary]

    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = mv_binary
    normalizing_values_bin['mean'] = list(train[mv_binary].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[mv_binary].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[mv_binary].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[mv_binary].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, mv_binary)
    test = normalization(test, 'min_max', normalizing_values_bin, mv_binary)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["continuous_variables"] = mv_continuous
    meta_Data["binary_variables"] = mv_binary


    output_file_name = data_dir + '/mv/outcomes_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    print("hola")

    return train, test

def epic_labs_proc_module(data_all, train_idx, test_idx):


    ordinal_variables = list(pd.read_csv(data_dir + '/epic/ordinal_vars_epic.txt', delimiter= "\t",header=None)[0])
    # data_all.drop(columns=['MUCOUS, URINE'], inplace=True)  ### TODO: check this later.  this is temporary because this variable is alwayss 0; dropped from the original dataset


    lab_cats = pd.read_csv(data_dir + '/epic/categories_labs.csv')
    preop_labs_categorical = lab_cats[lab_cats['all_numeric'] == 0.0]['LAB_TEST'].unique()
    num_lab_cats = [i for i in lab_cats['LAB_TEST'].unique() if
                    (i in data_all.columns) and (i not in preop_labs_categorical) and (i not in ordinal_variables)]

    ordinal_variables = [i for i in ordinal_variables if (i in data_all.columns)]

    categorical_variables = [i for i in preop_labs_categorical if i in data_all.columns]

    binary_variables = []
    continuous_variables = []


    for i in num_lab_cats:
        if len(data_all[i].value_counts().index) == 2 :
            data_all[i].fillna(0, inplace=True)
            binary_variables.append(i)
        elif len(data_all[i].value_counts(
                dropna=False).index) == 2:  # the variables that are reported only when present ([NA, value] form) can be traansformed to binary
            data_all[i].fillna(0, inplace=True)
            binary_variables.append(i)
        elif len(data_all[i].unique()) > 2 and len(data_all[i].value_counts().index) < 10:  # for the variables that have more than 2 categories
            categorical_variables.append(i)
        elif len(data_all[i].value_counts().index) > 10:
            continuous_variables.append(i)
        else:
            ordinal_variables.append(i)

    # need to rescale the binaries
    for i in binary_variables:
        data_all.loc[data_all[i] == 1, i] = 0
        data_all.loc[data_all[i] == 2, i] = 1

    # setting the caseids as index
    data_all = data_all.set_index('orlogid_encoded')


    # dropping the categoricals
    data_all.drop(columns=categorical_variables, inplace=True)

    # this round about thing is being done because not all preops have all the labs
    temp_df_train = pd.DataFrame(columns=data_all.columns)
    temp_df_train['orlogid_encoded'] = list(set(train_idx).difference(data_all.index))
    data_all = data_all.reset_index()
    temp_df_train = pd.concat([data_all, temp_df_train], axis=0)
    train = temp_df_train.set_index('orlogid_encoded').loc[train_idx]


    temp_df_test = pd.DataFrame(columns=data_all.columns)
    temp_df_test['orlogid_encoded'] = list(set(test_idx).difference(data_all.set_index('orlogid_encoded').index))
    temp_df_test = pd.concat([data_all, temp_df_test], axis=0)
    test = temp_df_test.set_index('orlogid_encoded').loc[test_idx]

    # we had done a reset in between
    data_all = data_all.set_index('orlogid_encoded')
    continuous_variables = [ i for i in data_all.columns if i not in (ordinal_variables+binary_variables)]


    meta_Data = {}

    # currently not dropping any of the labs even though some of them are missing a lot (above 90%)

    train[continuous_variables] = train[continuous_variables].fillna(train[continuous_variables].mean())
    test[continuous_variables] = test[continuous_variables].fillna(train[continuous_variables].mean()) # this is weird but inplace nan filling was not working correctly so had to do explict assignment


    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # imputing

    train[ordinal_variables] = train[ordinal_variables].fillna(train[ordinal_variables].median())
    test[ordinal_variables] = test[ordinal_variables].fillna(train[ordinal_variables].median())

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord

    # binary variables do not need fillna processing here but scaling is needed

    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = binary_variables
    normalizing_values_bin['mean'] = list(train[binary_variables].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[binary_variables].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[binary_variables].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[binary_variables].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, binary_variables)
    test = normalization(test, 'min_max', normalizing_values_bin, binary_variables)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")


    meta_Data["ordinal_variables"] = ordinal_variables
    meta_Data["continuous_variables"] = continuous_variables
    meta_Data["binary_variables"] = binary_variables


    output_file_name = data_dir + '/epic/labs_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    return train, test

def epic_preops_proc_module(data_all, test_size=0.2):
    ordinal_variables = list(pd.read_csv(data_dir + '/epic/ordinal_vars_epic.txt', delimiter= "\t",header=None)[0])
    ordinal_variables = [i for i in ordinal_variables if (i in data_all.columns)]


    # # making sure that Sex variable has 0 and 1 values instead of 1 and 2; this is not needed anymore as the mv dataset has 1 and 2 also
    # data_all.loc[data_all['Sex'] == 1, 'Sex'] = 0
    # data_all.loc[data_all['Sex'] == 2, 'Sex'] = 1
    # replacing True and False by 1 and 0
    data_all.replace(regex={False: 0, True: 1}, inplace=True) # these are already bool so they shouldn't be in the strings
    data_all.drop(columns = ['DVT_PE'], inplace=True) # DVT and PE already exist

    # encoding the plannedDispo from text to number
    # {"OUTPATIENT": 0, '23 HOUR ADMIT': 1, "FLOOR": 1, "OBS. UNIT": 2, "ICU": 3}
    data_all.loc[data_all['plannedDispo'] == 'Outpatient', 'plannedDispo'] = 0
    data_all.loc[data_all['plannedDispo'] == 'Floor', 'plannedDispo'] = 1
    data_all.loc[data_all['plannedDispo'] == 'Obs. unit', 'plannedDispo'] = 2
    data_all.loc[data_all['plannedDispo'] == 'ICU', 'plannedDispo'] = 3
    data_all['plannedDispo'] = data_all['plannedDispo'].astype('float') # needed to convert this to float because the nans were not getting converted to int and this variable is object type

    # setting the caseids as index
    data_all = data_all.set_index('orlogid_encoded')

    categorical_variables = ['Sex', 'PlannedAnesthesia'] # this is explicit because otherwise it is going into the binary list
    binary_variables = []
    continuous_variables = ['Secondary Diagnosis']
    for a in data_all.columns:
        if data_all[a].dtype == 'bool':
            data_all[a] = data_all[a].astype('int32')
        if data_all[a].dtype == 'int32'or data_all[a].dtype == 'int64':
            if len(data_all[a].value_counts().index) < 10 and len(data_all[a].value_counts().index) > 2 and (a not in ordinal_variables):  # using value_counts instead of unique becasue unique randimly decides to include or exclude nans
                data_all[a] = data_all[a].astype('category')
                categorical_variables.append(a)
        if len(data_all[a].value_counts().index) <= 2 and (a not in ordinal_variables+binary_variables+categorical_variables+continuous_variables):
            binary_variables.append(a)
        if data_all[a].dtype == 'O' and (a not in ordinal_variables+binary_variables+categorical_variables+continuous_variables):
            data_all[a] = data_all[a].astype('category')
            categorical_variables.append(a)

    # following inf is more or less hardcoded based on how the data was at the training time.
    categorical_variables.append('SurgService_Name')
    # preops['SurgService_Name'].replace(['NULL', ''], [np.NaN, np.NaN], inplace=True)
    if 'plannedDispo' in data_all.columns:
        data_all['plannedDispo'].replace('', np.NaN, inplace=True)

    dif_dtype = [a for a in data_all.columns if data_all[a].dtype not in ['int32', 'int64', 'float64',
                                                                                'category']]  # columns with non-numeric datatype; this was used at the code development time
    for a in dif_dtype:
        data_all[a] = data_all[a].astype('category')
        categorical_variables.append(a)

    # this is kind of hardcoded; check your data beforehand for this; fixed this
    temp_list = [i for i in data_all['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in data_all[
        'PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list != []:
        data_all['PlannedAnesthesia'].replace(temp_list, np.NaN,
                                                 inplace=True)  # this is done because there were two values for missing token (nan anf -inf)
    categorical_variables.append('PlannedAnesthesia')

    # remove if there are any duplicates in any of the variable name lists
    categorical_variables = [*set(categorical_variables)]

    continuous_variables = continuous_variables + [i for i in data_all.columns if
                                                   i not in (
                                                           binary_variables + categorical_variables + ordinal_variables)]
    continuous_variables = [*set(continuous_variables)]

    # since the categorical labs were float type earlier
    for name in categorical_variables:
        data_all[name] = data_all[name].astype('category')


    # one hot encoding
    meta_Data = {}

    meta_Data["levels"] = {}

    preops_ohe = data_all.copy()
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    import itertools
    encoded_variables = list()
    for i in categorical_variables:
        meta_Data["levels"][i] = list(data_all[i].cat.categories)
        temp = pd.get_dummies(data_all[i], dummy_na=True, prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))

    # creating the train test partitioning
    upto_test_idx = int(test_size * len(preops_ohe))
    test = preops_ohe.iloc[:upto_test_idx]
    train = preops_ohe.iloc[upto_test_idx:]
    # train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state)

    train_index = train.index
    test_index = test.index

    # mean imputing and scaling the continuous variables

    train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # median Imputing_ordinal variables

    for i in ordinal_variables:
        if np.isnan(data_all[i].unique().min()) == True:
            train[i].replace(train[i].unique().min(), train[i].median(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].median(), inplace=True)

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord

    # filling the nans in binary vars
    for i in binary_variables:  # this is again very random. Phew fillna has got to be crazy!!!!!
        temp_mode = train[i].mode()[0]
        train[i].fillna(temp_mode, inplace=True)
        test[i].fillna(temp_mode, inplace=True)

    meta_Data["train_median_bin"] = [train[i].mode()[0] for i in binary_variables]


    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = binary_variables
    normalizing_values_bin['mean'] = list(train[binary_variables].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[binary_variables].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[binary_variables].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[binary_variables].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, binary_variables)
    test = normalization(test, 'min_max', normalizing_values_bin, binary_variables)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["encoded_var"] = encoded_variables

    meta_Data["binary_var_name"] = binary_variables

    meta_Data["categorical_name"] = categorical_variables
    meta_Data["ordinal_variables"] = ordinal_variables
    meta_Data["continuous_variables"] = continuous_variables
    meta_Data["column_all_names"] = list(preops_ohe.columns)

    output_file_name = data_dir + '/epic/preops_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    print("hola")

    return train, test, train_index, test_index

def epic_outcomes_proc_module(data_all, train_idx, test_idx):

    # setting the caseids as index
    data_all = data_all.set_index('orlogid_encoded')
    data_all.drop(columns = ['DVT_PE'], inplace=True)
    data_all.dropna(subset=['case_duration'], inplace=True) # dropping the rows with missing case_duration


    # filling the ICULoS variable beforehand with 0
    data_all['ICULoS'].fillna(0, inplace=True)

    # replacing True and False by 1 and 0
    data_all.replace(regex={False: 0, True: 1}, inplace=True) # these are already bool so they shouldn't be in the strings

    binary_variables = []
    # not sure if initialting this is correct or now ## TODO: think?
    ordinal_variables = ['n_glu_high', 'n_glucose_low','severe_count_0','severe_count_1']  # this is needed becasue post_aki_status and pain scores and opioids counts are ordinals. Also, N_glu_measured is not added here because it is a general measurement whereas these ones have the order which could reprersent emergency

    for i in data_all.columns:
        if len(data_all[i].value_counts().index) == 2:
            binary_variables.append(i)
        elif len(data_all[i].value_counts().index) >  2 and data_all[i].dtype=='int' and (i not in ordinal_variables):  # the u
            ordinal_variables.append(i)
        elif len(data_all[i].value_counts().index) >  2 and len(data_all[i].value_counts().index) <12  and (i not in ordinal_variables): # this specific number 12 is to make sure that worst_pain outcome is added to ordinal
            ordinal_variables.append(i)

    continuous_variables = [i for i in data_all.columns if i not in (binary_variables+ordinal_variables)]

    # dont need to do the extra train test index mathcing because it fits perfectly here
    train = data_all.loc[train_idx]
    test = data_all.loc[test_idx]

    meta_Data = {}

    train[continuous_variables] = train[continuous_variables].fillna(train[continuous_variables].mean())
    test[continuous_variables] = test[continuous_variables].fillna(train[
                                                         continuous_variables].mean())  # this is weird but inplace nan filling was not working correctly so had to do explict assignment
    # train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    # test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]


    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # imputing

    train[ordinal_variables] = train[ordinal_variables].fillna(train[ordinal_variables].median())
    test[ordinal_variables] = test[ordinal_variables].fillna(train[ordinal_variables].median())

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord



    for i in binary_variables:  # this is again very random. Phew fillna has got to be crazy!!!!!
        temp_mode = train[i].mode()[0]
        train[i].fillna(temp_mode, inplace=True)
        test[i].fillna(temp_mode, inplace=True)

    meta_Data["train_median_bin"] = [train[i].mode()[0] for i in binary_variables]

    # min-max normalizing the binary variables; this is done so that  we dont have to worry about separate treatment of the binary and continuous
    normalizing_values_bin = {}
    normalizing_values_bin["bin_names"] = binary_variables
    normalizing_values_bin['mean'] = list(train[binary_variables].mean(axis=0).values)
    normalizing_values_bin['std'] = list(train[binary_variables].std(axis=0).values)
    normalizing_values_bin['min'] = list(train[binary_variables].min(axis=0).values)
    normalizing_values_bin['max'] = list(train[binary_variables].max(axis=0).values)
    train = normalization(train, 'min_max', normalizing_values_bin, binary_variables)
    test = normalization(test, 'min_max', normalizing_values_bin, binary_variables)
    meta_Data['norm_value_bin'] = normalizing_values_bin

    if (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["continuous_variables"] = continuous_variables
    meta_Data["binary_variables"] = binary_variables
    meta_Data["ordinal_variables"] = ordinal_variables

    output_file_name = data_dir + '/epic/outcomes_metadata.json'
    # # #
    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile,  default=str)  # the 'str' is important because the default int formatis not serializable

    print("hola")

    return train, test


data_dir = "/research2/trips/Feature_confusion/data/ActFast_twoeras"

if True:
    """ MV reading and processing """
    mv_preops = pd.read_csv(data_dir + "/mv/preops.csv")
    mv_preops = mv_preops.drop(index = mv_preops[mv_preops['Age']<18].index)

    mv_preops_train, mv_preops_test, mv_train_index, mv_test_index =  mv_preops_proc_module(mv_preops)

    mv_proc_emb = pd.read_csv(data_dir + "/mv/mv_epic_tokens.csv")
    mv_proc_emb.drop(columns=['Unnamed: 0'], inplace=True)
    mv_proc_emb.set_axis(['bow'+str(i) for i in mv_proc_emb.columns if i != 'caseid'] + ['caseid'], axis=1, inplace = True)

    mv_labs = pd.read_csv(data_dir + "/mv/more_numeric_labs.csv")
    # mv_labs_non_num = pd.read_csv(data_dir + "/mv/more_nonnumeric_labs.csv")  # not including these
    mv_outcomes = pd.read_csv(data_dir + "/mv/outcomes.csv")
    mv_outcomes.dropna(subset=['caseid'], inplace=True) # dropping the rows with missing caseid

    mv_outcomes_train, mv_outcomes_test = mv_outcomes_proc_module(mv_outcomes, mv_train_index, mv_test_index)

    mv_proc_labs_train, mv_proc_labs_test  = mv_labs_proc_module(mv_labs, mv_train_index, mv_test_index)

    # saving the Metavision outcomes, preops and labs train test files
    mv_preops_train.to_csv(data_dir+"/mv/mv_train_preops_Ageabove18.csv")
    mv_preops_test.to_csv(data_dir+"/mv/mv_test_preops_Ageabove18.csv")
    mv_proc_labs_train.to_csv(data_dir+"/mv/mv_train_labs_Ageabove18.csv")
    mv_proc_labs_test.to_csv(data_dir+"/mv/mv_test_labs_Ageabove18.csv")
    mv_outcomes_train.to_csv(data_dir+"/mv/mv_train_outcomes_Ageabove18.csv")
    mv_outcomes_test.to_csv(data_dir+"/mv/mv_test_outcomes_Ageabove18.csv")


    """ EPIC reading and processing """

    epic_preops_lab_comb = pd.read_csv(data_dir + "/epic/epic_preop.csv")
    epic_preops_lab_comb = epic_preops_lab_comb.drop(index = epic_preops_lab_comb[epic_preops_lab_comb['age']<18].index)
    ## separating the epic combined data into labs and preops
    epic_labs_preop_names = list(epic_preops_lab_comb.columns)
    epic_labs_names = epic_labs_preop_names[74:152]  # in the update there were 6 labs that were dropped so from 158 to 152 now
    epic_preops_names = list(set(epic_labs_preop_names).difference(set(epic_labs_names)))
    epic_preops_names.remove('MRN_encoded')
    epic_labs_names = epic_labs_names + ['orlogid_encoded']

    epic_preops = epic_preops_lab_comb[epic_preops_names]

    epic_labs = epic_preops_lab_comb[epic_labs_names]

    epic_proc_emb = pd.read_csv(data_dir + "/epic/epic_procedure_bow.csv")
    epic_outcomes = pd.read_csv(data_dir + "/epic/epic_outcomes.csv")
    epic_outcomes.drop(columns=['unit'], inplace=True)

    epic_preops_train, epic_preops_test, epic_train_index, epic_test_index =  epic_preops_proc_module(epic_preops)

    epic_outcomes_train, epic_outcomes_test = epic_outcomes_proc_module(epic_outcomes, epic_train_index, epic_test_index)

    epic_labs_train, epic_labs_test = epic_labs_proc_module(epic_labs, epic_train_index, epic_test_index)

    # saving the Epic outcomes, preops and labs train test files
    epic_preops_train.to_csv(data_dir+"/epic/epic_train_preops.csv")
    epic_preops_test.to_csv(data_dir+"/epic/epic_test_preops.csv")
    epic_labs_train.to_csv(data_dir+"/epic/epic_train_labs.csv")
    epic_labs_test.to_csv(data_dir+"/epic/epic_test_labs.csv")
    epic_outcomes_train.to_csv(data_dir+"/epic/epic_train_outcomes.csv")
    epic_outcomes_test.to_csv(data_dir+"/epic/epic_test_outcomes.csv")

    # minor processing of the embedded text from both eras consistently

    epic_proc_emb.set_index('orlogid_encoded', inplace=True)
    mv_proc_emb.set_index('caseid', inplace=True)

    emb_tokensto_drop = list(mv_proc_emb.sum(axis=0).sort_values().index[:100]) # this is being done because these tokens do not exist in any of the mv caseids so will drop from both eras; added 30 extra ones because later on train and test was creating a problem

    epic_proc_emb.drop(columns=emb_tokensto_drop, inplace=True)
    mv_proc_emb.drop(columns=emb_tokensto_drop, inplace=True)

    # adding the additional caseid in mv_emb

    # this round about thing is being done because not all preops have all the  procedure text
    temp_df_train = pd.DataFrame(columns=mv_proc_emb.columns)
    temp_df_train['caseid'] = list(set(mv_train_index).difference(mv_proc_emb.index))
    mv_proc_emb = mv_proc_emb.reset_index()
    temp_df_train = pd.concat([mv_proc_emb, temp_df_train], axis=0)
    mv_proc_emb_train = temp_df_train.set_index('caseid').loc[mv_train_index]
    mv_proc_emb_train.fillna(0, inplace=True)

    temp_df_test = pd.DataFrame(columns=mv_proc_emb.columns)
    temp_df_test['caseid'] = list(set(mv_test_index).difference(mv_proc_emb.set_index('caseid').index))
    temp_df_test = pd.concat([mv_proc_emb, temp_df_test], axis=0)
    mv_proc_emb_test = temp_df_test.set_index('caseid').loc[mv_test_index]
    mv_proc_emb_test.fillna(0, inplace=True)


    mv_proc_emb.set_index('caseid', inplace=True) # for the sake of completeness


    # adding the additional caseid in epic_emb

    temp_df_train = pd.DataFrame(columns=epic_proc_emb.columns)
    temp_df_train['orlogid_encoded'] = list(set(epic_train_index).difference(epic_proc_emb.index))
    epic_proc_emb = epic_proc_emb.reset_index()
    temp_df_train = pd.concat([epic_proc_emb, temp_df_train], axis=0)
    epic_proc_emb_train = temp_df_train.set_index('orlogid_encoded').loc[epic_train_index]
    epic_proc_emb_train.fillna(0, inplace=True)

    temp_df_test = pd.DataFrame(columns=epic_proc_emb.columns)
    temp_df_test['orlogid_encoded'] = list(set(epic_test_index).difference(epic_proc_emb.set_index('orlogid_encoded').index))
    temp_df_test = pd.concat([epic_proc_emb, temp_df_test], axis=0)
    epic_proc_emb_test = temp_df_test.set_index('orlogid_encoded').loc[epic_test_index]
    epic_proc_emb_test.fillna(0, inplace=True)

    epic_proc_emb.set_index('orlogid_encoded', inplace=True)


    # saving embedding mv epic  the train and test files
    mv_proc_emb_train.to_csv(data_dir+"/mv/mv_train_text_emb.csv")
    mv_proc_emb_test.to_csv(data_dir+"/mv/mv_test_text_emb.csv")
    epic_proc_emb_train.to_csv(data_dir+"/epic/epic_train_text_emb.csv")
    epic_proc_emb_test.to_csv(data_dir+"/epic/epic_test_text_emb.csv")



# gold standard files outcomes
mapped_outcomes  = pd.read_csv(data_dir + '/outcome_map.csv')
mapped_outcomes.drop(index=mapped_outcomes[mapped_outcomes['epic outcome'].isin(['DVT_PE','pneumonia_combined'])].index, inplace=True) # DVT_PE is redundant with DVT and PE and pneumonia_combined does not exist int he outcomes
mapped_outcomes.dropna(how='all', inplace=True)
mapped_outcomes.drop(columns=['Unnamed: 3'], inplace=True)  # dropping this column is only removing the old_Mortality30d variable because we are creating one to one matches here
temp_df_mapped = mapped_outcomes.dropna(how='any', subset = ['epic outcome', 'mv outcome'])  # getting the mappped outcome pairs only df
temp_df_mapped.drop_duplicates(subset=['mv outcome'], inplace=True)
mapped_outcome_dict = dict(zip(list(temp_df_mapped['epic outcome']), list(temp_df_mapped['mv outcome'])))
epic_withnomatch = [i for i in mapped_outcomes['epic outcome'] if i not in list(temp_df_mapped['epic outcome']) + [np.nan]]
mv_withnomatch = [i for i in mapped_outcomes['mv outcome'] if i not in list(temp_df_mapped['mv outcome']) + [np.nan]]

# gold standard files preops and labs
mapped_preops_labs  = pd.read_csv(data_dir + '/rwb_map.csv')
mapped_preops_labs.drop(index=mapped_preops_labs[mapped_preops_labs['ClarityFeature'].isin(['Coombs_Lab', 'MUCOUS, URINE', 'HBSAG', 'HCG,URINE, POC', 'HEPATITIS C AB', 'HIV 1/2 AB + P24 AG', 'COVID-19 CORONAVIRUS RNA' ])].index, inplace = True)
# seperating the labs and preops

###***********************************
#this is being done to make sure labs and preops reach their respective files
# epic_preops_lab_comb = pd.read_csv(data_dir + "/epic/epic_preop.csv")
# ## separating the epics combined data into labs and preops
# epic_labs_preop_names = list(epic_preops_lab_comb.columns)
# epic_labs_names = epic_labs_preop_names[74:152]
# epic_preops_names = list(set(epic_labs_preop_names).difference(set(epic_labs_names)))
# epic_preops_names.remove('MRN_encoded')

temp_df_lab = mapped_preops_labs[['ClarityFeature', 'MV Lab']]
temp_df_lab.dropna(how='all', inplace=True)
temp_df_lab.dropna(subset=['MV Lab'], inplace=True) # this is also dropping some extra variables
temp_df_lab = temp_df_lab.reset_index().drop(columns=['index'])
# to_include_labs_epic_wo_match =['COVID-19 CORONAVIRUS RNA']
# for i in to_include_labs_epic_wo_match:  # this is to make sure that all the variables are there
#     temp_df_lab.loc[len(temp_df_lab.index)] = [i, np.NaN]
categorical_labs_to_drop_mv = ['ABO_AND_RHO[D]_TYPING_P3-20220', 'CLARITY_URINE_P3-90001Y', 'COLOR_URINE_P3-90002Y'] # not necessary; can leave them as it is too

mapped_labs = temp_df_lab.dropna()
mapped_lab_dict = dict(zip(list(mapped_labs['ClarityFeature']), list(mapped_labs['MV Lab'])))


###***********************************

epic_preop_cat = ["RACE", "SurgService_Name", "PlannedAnesthesia", "ETHNICITY", "Sex"]
epic_preop_encoded_cat = ["RACE_-1", "RACE_0", "RACE_1", "RACE_2", "RACE_3", "RACE_nan", "SurgService_Name_1.0", "SurgService_Name_2.0", "SurgService_Name_3.0", "SurgService_Name_4.0", "SurgService_Name_5.0", "SurgService_Name_6.0", "SurgService_Name_7.0", "SurgService_Name_8.0", "SurgService_Name_9.0", "SurgService_Name_10.0", "SurgService_Name_11.0", "SurgService_Name_12.0", "SurgService_Name_13.0", "SurgService_Name_14.0", "SurgService_Name_15.0", "SurgService_Name_16.0", "SurgService_Name_nan", "PlannedAnesthesia_0.0", "PlannedAnesthesia_1.0", "PlannedAnesthesia_nan", "ETHNICITY_-1", "ETHNICITY_0", "ETHNICITY_1", "ETHNICITY_nan", "Sex_1", "Sex_2", "Sex_nan"]
mv_preop_cat = ["'CPAP.Usage", "Surg_Type", " 'PAP_Type", "SEX", "Anesthesia_Type", "RACE"]
mv_preop_encoded_cat = ["CPAP.Usage_27", "CPAP.Usage_29", "CPAP.Usage_32", "CPAP.Usage_Other", "CPAP.Usage_Unknown", "CPAP.Usage_nan", "Surg_Type_ACCS", "Surg_Type_Cardiothoracic", "Surg_Type_Colorectal", "Surg_Type_GYNECOLOGY", "Surg_Type_Hepatobiliary", "Surg_Type_Minimally Invasive Surgery", "Surg_Type_Neurosurgery", "Surg_Type_Orthopaedic", "Surg_Type_Other", "Surg_Type_Otolaryngology", "Surg_Type_Plastic", "Surg_Type_Transplant", "Surg_Type_UNKNOWN", "Surg_Type_Urology", "Surg_Type_Vascular", "Surg_Type_nan", "PAP_Type_DPAP (on ward)", "PAP_Type_IPAP", "PAP_Type_Other", "PAP_Type_Unknown", "PAP_Type_nan", "SEX_1", "SEX_2", "SEX_3", "SEX_nan", "Anesthesia_Type_1", "Anesthesia_Type_2", "Anesthesia_Type_3", "Anesthesia_Type_4", "Anesthesia_Type_Other", "Anesthesia_Type_nan", "RACE_Asian", "RACE_Black", "RACE_Other", "RACE_Unknown", "RACE_White", "RACE_nan"]


categ_mapped_dict_preops = {"RACE_-1": "RACE_Unknown" , "RACE_0":"RACE_White", "RACE_1":"RACE_Black", "RACE_2":"RACE_Asian", "RACE_3":"RACE_Other",
                            "SurgService_Name_1.0":"Surg_Type_ACCS", "SurgService_Name_2.0":"Surg_Type_Cardiothoracic",
                            "SurgService_Name_3.0":"Surg_Type_Colorectal", "SurgService_Name_4.0":"Surg_Type_Other", "SurgService_Name_5.0":"Surg_Type_Hepatobiliary",
                            "SurgService_Name_6.0":"Surg_Type_Minimally Invasive Surgery", "SurgService_Name_7.0":"Surg_Type_Neurosurgery", "SurgService_Name_8.0":"Surg_Type_GYNECOLOGY",
                            "SurgService_Name_9.0":"Surg_Type_UNKNOWN", "SurgService_Name_10.0": "Surg_Type_Other", "SurgService_Name_11.0":"Surg_Type_Orthopaedic",
                            "SurgService_Name_12.0":"Surg_Type_Otolaryngology",
                            "SurgService_Name_13.0": "Surg_Type_Plastic", "SurgService_Name_14.0":"Surg_Type_Transplant", "SurgService_Name_15.0":"Surg_Type_Urology",
                            "SurgService_Name_16.0":"Surg_Type_Vascular", "SurgService_Name_nan":"Surg_Type_UNKNOWN", "PlannedAnesthesia_1.0":"Anesthesia_Type_1",
                            "PlannedAnesthesia_0.0":"Anesthesia_Type_2", "PlannedAnesthesia_0.0":"Anesthesia_Type_3", "PlannedAnesthesia_0.0":"Anesthesia_Type_4",
                            "PlannedAnesthesia_0.0":"Anesthesia_Type_Other", "Sex_1":"SEX_1", "Sex_2":"SEX_2"}  # the repeated keys are not being picked up dictionary because they are supposed to be unique



epic_enc_vars_to_add = [i for i in epic_preop_encoded_cat if i not in categ_mapped_dict_preops.keys()]
mv_enc_vars_to_add = [i for i in mv_preop_encoded_cat if i not in categ_mapped_dict_preops.values()] # this has some vars double counted because in the dictionary the maps are not unique

temp_df_preops = mapped_preops_labs[['ClarityFeature', 'MV Feature']]
# idx_to_Drop_withlabs = mapped_preops_labs[mapped_preops_labs['MV Lab'].notnull() ==True].index  #[mapped_preops_labs['MV Feature'].isna()==True].index  # rows with labs in them
idx_to_Drop_withlabs = mapped_preops_labs[mapped_preops_labs['ClarityFeature'].isin(temp_df_lab['ClarityFeature'])].index # rows with labs in them
temp_df_preops.drop(index = idx_to_Drop_withlabs, inplace=True )
# temp_df_preops.drop(index = temp_df_preops.loc[(temp_df_preops['ClarityFeature'].isin(to_include_labs_epic_wo_match))].index, inplace =True)
temp_df_preops = temp_df_preops.reset_index().drop(columns=['index'])

# adding the mapped categoricals
for i in categ_mapped_dict_preops.keys():  # this is to make sure that all the variables are there
    temp_df_preops.loc[len(temp_df_preops.index)] = [i, categ_mapped_dict_preops[i]]

# adding the remaining encoded categoricals
for i in epic_enc_vars_to_add:  # this is to make sure that all the variables are there
    temp_df_preops.loc[len(temp_df_preops.index)] = [i, np.NaN]
for i in mv_enc_vars_to_add:  # this is to make sure that all the variables are there
    temp_df_preops.loc[len(temp_df_preops.index)] = [ np.NaN, i]

# dropping the actual categorical names
temp_df_preops.drop(index = temp_df_preops.loc[(temp_df_preops['ClarityFeature'].isin(epic_preop_cat))].index, inplace =True)
temp_df_preops.drop(index = temp_df_preops.loc[(temp_df_preops['MV Feature'].isin(mv_preop_cat))].index, inplace =True)  # CPAP.Usage and PAP_Type did not get dropped because they have an extra '

# dropping the nans
mapped_preops = temp_df_preops.dropna().drop_duplicates(subset=['MV Feature'])

mapped_preops_dict = dict(zip(list(mapped_preops['ClarityFeature']), list(mapped_preops['MV Feature'])))

###***********************************


### Saving all three mapped files

temp_df_lab.to_csv(data_dir+"/mapped_labs_epic_mv.csv", index =False)
temp_df_preops.to_csv(data_dir+"/mapped_preops_epic_mv.csv", index =False)
mapped_outcomes.to_csv(data_dir+"/mapped_outcomes_epic_mv.csv", index =False)

print("--- Finished ----")


# some stats printing
# print("MV era")
# print("preops ", mv_preops.shape)
# print("numerical labs", mv_labs.shape)
# print(" procedure emb", mv_proc_emb.shape)
# print("outcomes ", mv_outcomes.shape)
# print("\n")
# print(" Epic era")
# print(" preops ", epic_preops.shape)
# print(" labs ", epic_labs.shape)
# print(" procedure text ", epic_proc_emb.shape)
# print(" outcomes ", epic_outcomes.shape)