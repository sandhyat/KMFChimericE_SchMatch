"""
this file provides a pipeline of univariate feature mapping solutions. The two methods are:
    1) Computing the basic summary stats from the describe function of pandas along with coefficient of variation and using them  to obtain the euclidean distance between the features from two sources.
    2) Using the KS statistic as a similarity measure betweent he empirical distribution of the features from the two sources

Using hospital resident matching algorithm (with capacity 1) on the similarity matrices for both the above methods and get the final mappings.


INPUT:

ACTFAST dataset (continuous real data) (from Metavision and EPIC era)

OUTPUT:

F1 scores and Match MSE values for both the methods saved in a file
The matches from both the methods in the form of a dataframe saved to a csv for both the methods.


Note: Cv here refers to the epic era. For convenience the notation is common across the code

"""

# importing packages
import numpy as np
from scipy import linalg, stats
from scipy.spatial.distance import cdist
import os.path
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise,mutual_info_score, mean_squared_error
from matching.games import StableMarriage, HospitalResident
import datetime
import json, sys, argparse


print(sys.getrecursionlimit())
sys.setrecursionlimit(3500)
print(sys.getrecursionlimit())

def Matching_via_HRM(C_X1_train, C_X2_train, P_x1_O_to_R):  # in this case here the small feature sized database is X1, so we need to treat it as hospital and there will be capacities on it.
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

    # print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    # print(matching_x1_train)

    # print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    # print(matching_x2_train)

    x1_train_y = np.array([int(str(v[0]).split("C")[1]) for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v[0]).split("R")[1]) for v in matching_x2_train.values()])

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(np.transpose(C_X2_train).shape)

    # print(" x1 matching shape ", matching_x1_train_matrix.shape)
    # print(" x1 matching shape", matching_x2_train_matrix.shape)
    # print(" true perm matrix shape ", P_x1_O_to_R.shape)

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

def svd(X, n_components=2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]

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

parser = argparse.ArgumentParser(description='HP for CL optimization')

## dataset and setup parameters
parser.add_argument("--dataset_number",  default='ACTFAST') # could be integer or string
parser.add_argument("--outcome",  default="Y")  # this is not very relevant but kep for the sake of completeness
parser.add_argument("--frac_renamed", default=0.5, type=float)
parser.add_argument("--randomSeed", default=6936, type=int )
parser.add_argument("--testDatasetSize", default=0.2, type=float) # this top fraction of data is not used by this code

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing

"""  for iterating and initial stuff  """

input_dir = "/research2/trips/Feature_confusion/data/ActFast_twoeras"
# data_dir = '/input'
# input_dir = "/research2/trips/Feature_confusion/data/ActFast_twoeras"
# input_dir = '/input/'

output_dir = './'
# output_dir = '/output/'

# reading the data and matching files

mv_labs_train =  pd.read_csv(input_dir+"/mv/mv_train_labs_Ageabove18.csv")
mv_labs_test = pd.read_csv(input_dir+"/mv/mv_test_labs_Ageabove18.csv")
mv_preops_train =  pd.read_csv(input_dir+"/mv/mv_train_preops_Ageabove18.csv")
mv_preops_test = pd.read_csv(input_dir+"/mv/mv_test_preops_Ageabove18.csv")

mv_preops_train.set_index('caseid', inplace=True)
mv_preops_test.set_index('caseid', inplace=True)
mv_labs_train.set_index('caseid', inplace=True)
mv_labs_test.set_index('caseid', inplace=True)

epic_labs_train =  pd.read_csv(input_dir+"/epic/epic_train_labs.csv")
epic_labs_test = pd.read_csv(input_dir+"/epic/epic_test_labs.csv")
epic_preops_train =  pd.read_csv(input_dir+"/epic/epic_train_preops.csv")
epic_preops_test = pd.read_csv(input_dir+"/epic/epic_test_preops.csv")

epic_preops_train.set_index('orlogid_encoded', inplace=True)
epic_preops_test.set_index('orlogid_encoded', inplace=True)
epic_labs_train.set_index('orlogid_encoded', inplace=True)
epic_labs_test.set_index('orlogid_encoded', inplace=True)


# gold standard files
mapped_labs_df = pd.read_csv(input_dir+"/mapped_labs_epic_mv.csv")
mapped_preops_df =  pd.read_csv(input_dir+"/mapped_preops_epic_mv.csv")


mapped_preops = mapped_preops_df.dropna().drop_duplicates(subset=['MV Feature'])
mapped_preops_dict = dict(zip(list(mapped_preops['ClarityFeature']), list(mapped_preops['MV Feature'])))



match_df = mapped_labs_df.dropna()
match_dic = dict(zip(list(match_df['ClarityFeature']), list(match_df['MV Lab'])))

epic_withmatch = list(match_df['ClarityFeature'])
epic_withnomatch = [i for i in epic_labs_train.columns if i not in list(match_df['ClarityFeature'])+['orlogid_encoded']]
mv_withnomatch = [i for i in mv_labs_train.columns if i not in list(match_df['MV Lab'])+['caseid']]

print( " Epic labs with match ", len(match_df))
print( " MV labs with match ", len(match_df))

print( " Epic labs with NO match ", len(epic_withnomatch))
print( " MV labs with NO match ", len(mv_withnomatch))


num_xtra_feat_inX1 = len(epic_withnomatch)

# breakpoint()

# arrange the preops in the same order; only use the ones that have a match
epic_preops_train_proc = epic_preops_train[[i for i in mapped_preops_dict.keys()]]
mv_preops_train_proc = mv_preops_train[[i for i in mapped_preops_dict.values()]]

epic_preops_test_proc = epic_preops_test[epic_preops_train_proc.columns]
mv_preops_test_proc = mv_preops_test[mv_preops_train_proc.columns]

# need to create a common column name for mapped features as it will be easier
known_columnslist = ['KM_' + str(i + 1) for i in range(len(mapped_preops_dict))]

epic_preops_train_proc.set_axis(known_columnslist, axis=1, inplace=True)  # renamed the known mapped preops columns
mv_preops_train_proc.set_axis(known_columnslist, axis=1, inplace=True)  # renamed the known mapped preops columns
epic_preops_test_proc.set_axis(known_columnslist, axis=1, inplace=True)  # renamed the known mapped preops columns
mv_preops_test_proc.set_axis(known_columnslist, axis=1, inplace=True)  # renamed the known mapped preops columns

list_of_number_mapped_variables = [15, 25, 35, 45, len(known_columnslist)]

# concatenate the preops and labs
df_train_CV_proc = epic_preops_train_proc.join(epic_labs_train)
df_train_MV_proc = mv_preops_train_proc.join(mv_labs_train)

df_holdout_CV_proc = epic_preops_test_proc.join(epic_labs_test)
df_holdout_MV_proc = mv_preops_test_proc.join(mv_labs_test)

print("Disfrutar")

## plotting the correlograms
if False:
    Cor_from_df_CV = df_train_CV_proc.corr()
    Cor_from_df_MV = df_train_MV_proc.corr()
    breakpoint()

    simple_CV_cor_plot = sns.heatmap(Cor_from_df_CV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    simple_CV_cor_plot.set_title("ACTFAST Epic era era feature correlations")
    simple_CV_cor_plot.hlines([len(known_columnslist)],*simple_CV_cor_plot.get_xlim(), colors='black')
    simple_CV_cor_plot.vlines([len(known_columnslist)],*simple_CV_cor_plot.get_ylim(), colors='black')
    fig = simple_CV_cor_plot.get_figure()
    fig.savefig("Epic_ACTFAST_Block_correlation.pdf", bbox='tight')
    fig.savefig("Epic_ACTFAST_Block_correlation.png", bbox='tight')
    plt.close()
    breakpoint()

    simple_MV_cor_plot = sns.heatmap(Cor_from_df_MV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    simple_MV_cor_plot.set_title("ACTFAST Metavision (MV) era feature correlations")
    simple_MV_cor_plot.hlines([len(known_columnslist)],*simple_MV_cor_plot.get_xlim(), colors='black')
    simple_MV_cor_plot.vlines([len(known_columnslist)],*simple_MV_cor_plot.get_ylim(), colors='black')
    fig = simple_MV_cor_plot.get_figure()
    fig.savefig("MV_ACTFAST_Block_correlation.pdf", bbox='tight')
    fig.savefig("MV_ACTFAST_Block_correlation.png", bbox='tight')
    plt.close()
    breakpoint()

# breakpoint()
# generating the correlation matrix for true matches; computing on a smaller dataset
if False:
    Cor_btw_df = np.zeros((len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))
    num_obs_for_cor = min(min(len(df_train_CV_proc), len(df_train_MV_proc)), 10000)
    for i in range(len(df_train_CV_proc.columns)):
        for j in range(len(df_train_MV_proc.columns)):
            temp = stats.pearsonr(df_train_CV_proc.values[:num_obs_for_cor,i], df_train_MV_proc.values[:num_obs_for_cor, j])
            Cor_btw_df[i,j] = temp[0]
else:
    Cor_btw_df = np.random.uniform(size=(len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))

# breakpoint()
# converting the correlation matrix to a dataframe
if False:
    Cor_df = pd.DataFrame(Cor_btw_df,
                 index=df_train_CV_proc.columns,
                 columns=df_train_MV_proc.columns)
else:
    Cor_df = pd.read_csv(input_dir+'/Cor_matrix_preops_labs_comb.csv').set_index('Unnamed: 0')

total_CV_data_proc = pd.concat([df_train_CV_proc, df_holdout_CV_proc], axis=0)
total_MV_data_proc = pd.concat([df_train_MV_proc, df_holdout_MV_proc], axis=0)

## initial permuation matrix calculation
P_x1 = np.zeros((len(df_train_CV_proc.columns), len(df_train_MV_proc.columns)))  # this includes the mapped labs that will removed

print("Shape of P_x1 ", P_x1.shape)
# breakpoint()
# exit()
for i in range(len(df_train_CV_proc.columns)):
    for j in range(len(df_train_MV_proc.columns)):
        if df_train_CV_proc.columns[i] == df_train_MV_proc.columns[j]:
            P_x1[i, j] = 1
        elif df_train_CV_proc.columns[i] in epic_withmatch:
            if (match_dic[df_train_CV_proc.columns[i]] == df_train_MV_proc.columns[j]):
                P_x1[i, j] = 1
        elif (df_train_CV_proc.columns[i] in epic_withnomatch) & (
                df_train_MV_proc.columns[j] in mv_withnomatch):
            P_x1[i, j] = 2


P_x1_final = P_x1[len(known_columnslist):, len(known_columnslist):]

"""  Summary statistic based  """

stats_CV = total_CV_data_proc[epic_labs_train.columns].describe().swapaxes("index", "columns")
stats_CV['CoV'] = stats_CV['std']/stats_CV['mean']
stats_CV.drop(columns=['count'], inplace=True)

stats_MV = total_MV_data_proc[mv_labs_train.columns].describe().swapaxes("index", "columns")
stats_MV['CoV'] = stats_MV['std']/stats_MV['mean']
stats_MV.drop(columns=['count'], inplace=True)

similarity_matrix_CV_MV = cdist(stats_CV, stats_MV, 'euclid')

# matching by GS
correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
    similarity_matrix_CV_MV,
    np.transpose(similarity_matrix_CV_MV),
    P_x1_final)

temp_inf_x1 = pd.DataFrame(
    columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation', 'no_match_or_not'])

for i in range(x1_match_matrix_test.shape[0]):
    matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if
                     x1_match_matrix_test[i, j] == 1]
    temp_inf_x1.loc[i, "ump_feature_in_X1"] = epic_labs_train.columns[i]
    # temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(epic_labs_train.columns[i])]
    temp_inf_x1.loc[i, "match_byGS"] = mv_labs_train.columns[matched_index[0]]
    # temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[int(mv_labs_train.columns[matched_index[0]])]
    temp_inf_x1.loc[i, "true_correlation"] = Cor_df.loc[epic_labs_train.columns[i], mv_labs_train.columns[matched_index[0]]]
    if np.any(P_x1_final[i] == 2):
        temp_inf_x1.loc[i, "no_match_or_not"] = 1
        temp_inf_x1.loc[i, "Correct_Match"] = "NA"
    else:
        temp_inf_x1.loc[i, "no_match_or_not"] = 0
        temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                                  match_df[
                                                      match_df['ClarityFeature'] == temp_inf_x1.loc[
                                                          i, 'ump_feature_in_X1']][
                                                      'MV Lab'])

print("\n \n List of mismatched feature number when # of mapped features are \n ")

print(" \n Mistakes by the Summary statistic method")
print(" Summary Statistic  X1_train mistakes number on holdout set",
      len(epic_labs_train.columns) - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
      len(epic_labs_train.columns) - num_xtra_feat_inX1)
print(" Summary Statistic   X2_train mistakes number on holdout set",
      len(epic_labs_train.columns) - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
      len(epic_labs_train.columns) - num_xtra_feat_inX1)

# evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                 list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                   list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))


# obtaining the index to compute the loss on incorrect or correct matches
correct_match_idx_orig_from_x1 = []
for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
    list(df_holdout_CV_proc.columns[len(known_columnslist):]).index(i))
correct_match_idx_r_from_x1 = []
for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
    list(df_holdout_MV_proc.columns[len(known_columnslist):]).index(i))

incorrect_match_idx_orig_from_x1 = []
for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
    list(df_holdout_CV_proc.columns[len(known_columnslist):]).index(i))
incorrect_match_idx_r_from_x1 = []
for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
    list(df_holdout_MV_proc.columns[len(known_columnslist):]).index(i))

# breakpoint()
predicted_match_dic_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['no_match_or_not'] == 0]['ump_feature_in_X1']),
                                  list(temp_inf_x1[temp_inf_x1['no_match_or_not'] == 0]['match_byGS'])))

final_dic_for_compar_matching = {}
for key, val in match_dic.items():
    if val in predicted_match_dic_x1.values():
        final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[
            list(predicted_match_dic_x1.values()).index(val)]

# Matching metric error
overall_quality_error_matching_only = mean_squared_error(
    df_holdout_CV_proc[final_dic_for_compar_matching.keys()].values,
    df_holdout_CV_proc[final_dic_for_compar_matching.values()])


TP_x1 = 0
FP_x1 = 0
TN_x1 = 0
FN_x1 = 0
for i in range(P_x1_final.shape[0]):
    for j in range(P_x1_final.shape[1]):
        if (P_x1_final[i, j] == 1) & (x1_match_matrix_test[i, j] == 1):
            TP_x1 = TP_x1 + 1
        elif (P_x1_final[i, j] == 1) & (x1_match_matrix_test[i, j] == 0):
            FN_x1 = FN_x1 + 1
        elif (P_x1_final[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
            TN_x1 = TN_x1 + 1
        elif (P_x1_final[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
            FP_x1 = FP_x1 + 1

TP_x2 = 0
FP_x2 = 0
TN_x2 = 0
FN_x2 = 0
for i in range(P_x1_final.shape[0]):
    for j in range(P_x1_final.shape[1]):
        if (P_x1_final[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
            TP_x2 = TP_x2 + 1
        elif (P_x1_final[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
            FN_x2 = FN_x2 + 1
        elif (P_x1_final[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
            TN_x2 = TN_x2 + 1
        elif (P_x1_final[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
            FP_x2 = FP_x2 + 1

F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)


print(" confusion mat vals from the summary stats method, TP: ", TP_x1, " FP ", FP_x1, " FN ", FN_x1, " TN ", TN_x1)
print("F1 from Summary statistic value ", F1_fromx1)
print('Matching metric from Summary statistic value ', overall_quality_error_matching_only)

"""  KS statistic based  """

KS_stat_matrix = np.zeros((len(epic_labs_train.columns), len(mv_labs_train.columns)))
KS_stat_pvalue = np.zeros((len(epic_labs_train.columns), len(mv_labs_train.columns)))

for cv_id in range(len(epic_labs_train.columns)):
    for mv_id in range(len(mv_labs_train.columns)):
        ks_sta_val, ks_pvalue = stats.kstest(total_CV_data_proc.iloc[:,len(known_columnslist)+cv_id], total_MV_data_proc.iloc[:,len(known_columnslist)+mv_id])
        KS_stat_matrix[cv_id, mv_id] = ks_sta_val
        KS_stat_pvalue[cv_id, mv_id] = ks_pvalue


# matching by GS
correct_with_match_from_x1_test_KS, correct_with_match_from_x2_test_KS, x1_match_matrix_test_KS, x2_match_matrix_test_KS = Matching_via_HRM(
    KS_stat_matrix,
    np.transpose(KS_stat_matrix),
    P_x1_final)

temp_inf_x1_KS = pd.DataFrame(
    columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation', 'no_match_or_not'])

for i in range(x1_match_matrix_test_KS.shape[0]):
    matched_index = [j for j in range(x1_match_matrix_test_KS.shape[1]) if
                     x1_match_matrix_test_KS[i, j] == 1]
    temp_inf_x1_KS.loc[i, "ump_feature_in_X1"] = epic_labs_train.columns[i]
    temp_inf_x1_KS.loc[i, "match_byGS"] = mv_labs_train.columns[matched_index[0]]
    temp_inf_x1_KS.loc[i, "true_correlation"] = Cor_df.loc[epic_labs_train.columns[i], mv_labs_train.columns[matched_index[0]]]
    if np.any(P_x1_final[i] == 2):
        temp_inf_x1_KS.loc[i, "no_match_or_not"] = 1
        temp_inf_x1_KS.loc[i, "Correct_Match"] = "NA"
    else:
        temp_inf_x1_KS.loc[i, "no_match_or_not"] = 0
        temp_inf_x1_KS.loc[i, "Correct_Match"] = sum(temp_inf_x1_KS.loc[i, 'match_byGS'] ==
                                                  match_df[
                                                      match_df['ClarityFeature'] == temp_inf_x1_KS.loc[
                                                          i, 'ump_feature_in_X1']][
                                                      'MV Lab'])

print("\n \n List of mismatched feature number when # of mapped features are \n ")

print(" \n Mistakes by the KS statistic method")
print(" KS Statistic  X1_train mistakes number on holdout set",
      len(epic_labs_train.columns) - correct_with_match_from_x1_test_KS - num_xtra_feat_inX1, "out of ",
      len(epic_labs_train.columns) - num_xtra_feat_inX1)
print(" KS Statistic   X2_train mistakes number on holdout set",
      len(epic_labs_train.columns) - correct_with_match_from_x2_test_KS - num_xtra_feat_inX1, "out of ",
      len(epic_labs_train.columns) - num_xtra_feat_inX1)

# evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
correct_match_dict_x1_KS = dict(zip(list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 1]['ump_feature_in_X1']),
                                 list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 1]['match_byGS'])))
incorrect_match_dict_x1_KS = dict(zip(list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 0]['ump_feature_in_X1']),
                                   list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 0]['match_byGS'])))


# obtaining the index to compute the loss on incorrect or correct matches
correct_match_idx_orig_from_x1_KS = []
for i in list(correct_match_dict_x1_KS.keys()): correct_match_idx_orig_from_x1_KS.append(
    list(df_holdout_CV_proc.columns[len(known_columnslist):]).index(i))
correct_match_idx_r_from_x1_KS = []
for i in list(correct_match_dict_x1_KS.values()): correct_match_idx_r_from_x1_KS.append(
    list(df_holdout_MV_proc.columns[len(known_columnslist):]).index(i))

incorrect_match_idx_orig_from_x1_KS = []
for i in list(incorrect_match_dict_x1_KS.keys()): incorrect_match_idx_orig_from_x1_KS.append(
    list(df_holdout_CV_proc.columns[len(known_columnslist):]).index(i))
incorrect_match_idx_r_from_x1_KS = []
for i in list(incorrect_match_dict_x1_KS.values()): incorrect_match_idx_r_from_x1_KS.append(
    list(df_holdout_MV_proc.columns[len(known_columnslist):]).index(i))

# breakpoint()
predicted_match_dic_x1_KS = dict(zip(list(temp_inf_x1_KS[temp_inf_x1_KS['no_match_or_not'] == 0]['ump_feature_in_X1']),
                                  list(temp_inf_x1_KS[temp_inf_x1_KS['no_match_or_not'] == 0]['match_byGS'])))

final_dic_for_compar_matching_KS = {}
for key, val in match_dic.items():
    if val in predicted_match_dic_x1_KS.values():
        final_dic_for_compar_matching_KS[key] = list(predicted_match_dic_x1_KS.keys())[
            list(predicted_match_dic_x1_KS.values()).index(val)]

# Matching metric error
overall_quality_error_matching_only_KS = mean_squared_error(
    df_holdout_CV_proc[final_dic_for_compar_matching_KS.keys()].values,
    df_holdout_CV_proc[final_dic_for_compar_matching_KS.values()])


TP_x1 = 0
FP_x1 = 0
TN_x1 = 0
FN_x1 = 0
for i in range(P_x1_final.shape[0]):
    for j in range(P_x1_final.shape[1]):
        if (P_x1_final[i, j] == 1) & (x1_match_matrix_test_KS[i, j] == 1):
            TP_x1 = TP_x1 + 1
        elif (P_x1_final[i, j] == 1) & (x1_match_matrix_test_KS[i, j] == 0):
            FN_x1 = FN_x1 + 1
        elif (P_x1_final[i, j] == 0) & (x1_match_matrix_test_KS[i, j] == 0):
            TN_x1 = TN_x1 + 1
        elif (P_x1_final[i, j] == 0) & (x1_match_matrix_test_KS[i, j] == 1):
            FP_x1 = FP_x1 + 1

TP_x2 = 0
FP_x2 = 0
TN_x2 = 0
FN_x2 = 0
for i in range(P_x1_final.shape[0]):
    for j in range(P_x1_final.shape[1]):
        if (P_x1_final[i, j] == 1) & (x2_match_matrix_test_KS[i, j] == 1):
            TP_x2 = TP_x2 + 1
        elif (P_x1_final[i, j] == 1) & (x2_match_matrix_test_KS[i, j] == 0):
            FN_x2 = FN_x2 + 1
        elif (P_x1_final[i, j] == 0) & (x2_match_matrix_test_KS[i, j] == 0):
            TN_x2 = TN_x2 + 1
        elif (P_x1_final[i, j] == 0) & (x2_match_matrix_test_KS[i, j] == 1):
            FP_x2 = FP_x2 + 1

F1_fromx1_KS = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
F1_fromx2_KS = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

print(" confusion mat vals from the KS stat method, TP: ", TP_x1, " FP ", FP_x1, " FN ", FN_x1, " TN ", TN_x1)
print("F1 from KS statistic comparison ", F1_fromx1_KS)
print('Matching metric from KS statistic value ', overall_quality_error_matching_only_KS)

breakpoint()

# saving all the files

saving_dir = output_dir + 'UnivariateStats_basedcomparison_'+str(datetime.date.today()) +  '/ACTFAST_data'+str(datetime.datetime.now())

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

temp_inf_x1.to_csv(saving_dir+"/Post-hoc_from_x1_Summary_Statisticbased_ACTFAST.csv", index=False)
temp_inf_x1_KS.to_csv(saving_dir+"/Post-hoc_from_x1_KS_Statisticbased_ACTFAST.csv", index=False)

file_name = saving_dir+"/Univariate_stats_based_ACTFAST_mistakes.txt"

if os.path.exists(file_name):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name)

f = open(file_name, 'w')
f.write("\n \n ACTFAST data results  ")
f.write("\n \n F1 for summary statistic based {0}".format(F1_fromx1))
f.write("\n \n F1 for KS statistic based {0}".format(F1_fromx1_KS))
f.write("\n \n ")
f.write("\n \n MatchMSE for summary statistic based {0}".format(overall_quality_error_matching_only))
f.write("\n \n MatchMSE for KS statistic based {0}".format(overall_quality_error_matching_only_KS))
f.write("\n \n ")
f.close()
