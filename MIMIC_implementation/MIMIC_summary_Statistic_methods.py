"""
this file provides a pipeline of univariate feature mapping solutions. The two methods are:
    1) Computing the basic summary stats from the describe function of pandas along with coefficient of variation and using them  to obtain the euclidean distance between the features from two sources.
    2) Using the KS statistic as a similarity measure betweent he empirical distribution of the features from the two sources

Using hospital resident matching algorithm (with capacity 1) on the similarity matrices for both the above methods and get the final mappings.


INPUT:

MIMIC dataset (continuous real data) from the two eras (Carevue and Metavision)

OUTPUT:

F1 scores and Match MSE values for both the methods saved in a file
The matches from both the methods in the form of a dataframe saved to a csv for both the methods.


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
parser.add_argument("--dataset_number",  default='MIMIC') # could be integer or string
parser.add_argument("--outcome",  default="Y")  # this is not very relevant but kep for the sake of completeness
parser.add_argument("--frac_renamed", default=0.5, type=float)
parser.add_argument("--randomSeed", default=6936, type=int )
parser.add_argument("--testDatasetSize", default=0.2, type=float) # this top fraction of data is not used by this code

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing

"""  for iterating and initial stuff  """

input_dir = "/home/trips/"
# input_dir = '/input/'

output_dir = './'
# output_dir = '/output/'

# Reading data files (Moose location)
CV_full = pd.read_csv(input_dir + 'MIMIC_feature_confusion/Final_MIMIC_lab_chart_CV.csv')
MV_full = pd.read_csv(input_dir + 'MIMIC_feature_confusion/Final_MIMIC_lab_chart_MV.csv')
# Getting list of all items along with the source and label
item_id_dbsource = pd.read_csv(input_dir + 'd_items_chartevents.csv')
itemid_labs = pd.read_csv(input_dir + 'd_items_labevents.csv')

# temporary patching; need fixing
item_id_dbsource =item_id_dbsource.drop_duplicates(subset=['label','dbsource'], keep='last')


# Reading data files (Tantra location)
# CV_full = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/Final_MIMIC_lab_chart_CV.csv')
# MV_full = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/Final_MIMIC_lab_chart_MV.csv')
# # Getting list of all items along with the source and label
# item_id_dbsource = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/d_items_chartevents.csv')
# itemid_labs = pd.read_csv('/project/tantra/Sandhya/DeployModel/FEatureconfusion_JAn_revisiion/MIMIC_related/d_items_labevents.csv')

# reseting the index to hadm_id for both
CV_full.set_index('hadm_id', inplace=True)
MV_full.set_index('hadm_id', inplace=True)
# ##breakpoint()
CV_full.drop(columns='742', inplace=True) # Dropping because this variable only takes 1 as the value.

# dropping the labs that are in the blood gas category
to_drop_labs = ['50800', '50801', '50803', '50804', '50807', '50810', '50812', '50815', '50816', '50817', '50819', '50823', '50825', '50826', '50827', '50828', '50829', '51544', '51545']

list_lab_ids0 = list(CV_full.columns)[:58] # initial set of labs
to_drops_labs_current = list(set(list_lab_ids0).intersection(to_drop_labs))

CV_full.drop(columns=to_drops_labs_current, inplace=True) # Dropping these labs that are blood gas category
MV_full.drop(columns=to_drops_labs_current, inplace=True) # Dropping these labs that are blood gas category

# creating label dictionary for later
itemid_label_lab = dict(zip(list(itemid_labs.itemid), list(itemid_labs.label)))
itemid_label_chart = dict(zip(list(item_id_dbsource.itemid), list(item_id_dbsource.label)))
itemid_label_dict = {**itemid_label_chart, **itemid_label_lab}  # merging two dictionaries here

Actual_chart_expertKnowledge_CV =  ['NBP [Diastolic]',
 'NBP [Systolic]',
 'NBP Mean',
 'Arterial BP [Systolic]',
 'Arterial BP [Diastolic]',
 'Arterial BP Mean',
 'Previous Weight',
 'Daily Weight',
 'Admit Wt',
 'Weight Change',
 'Previous WeightF',
 'Admit Ht',
 'BSA',
 'Respiratory Rate',
 'Resp Rate (Spont)',
 'Resp Rate (Total)',
 'Respiratory Rate Set',
 'PAP [Diastolic]',
 'O2 Flow (lpm)',
 'FiO2 Set',
 'Braden Score',
 'Temperature C (calc)',
 'Temperature F',
 'Plateau Pressure',
 'Minute Volume(Obser)',
 'Tidal Volume (Spont)',
 'Tidal Volume (Set)',
 'Tidal Volume (Obser)',
 'Mean Airway Pressure',
 'Compliance (40-60ml)',
 'Peak Insp. Pressure',
 'Pressure Support',
 'PEEP Set',
 'Heart Rate',
 'Tank A psi.',
 'Tank B psi.',
 'SpO2',
 'CVP',
 'GCS Total',
'ABP Alarm [High]',
'Resp Alarm [High]',
'Low Exhaled Min Vol',
'NBP Alarm [Low]',
'Ve High',
'Resp Alarm [Low]',
'Sensitivity-Vent',
'SpO2 Alarm [High]',
'High Insp. Pressure',
'NBP Alarm [High]',
'Apnea Time Interval',
'HR Alarm [Low]',
 'SpO2 Alarm [Low]',
 'ABP Alarm [Low]',
'HR Alarm [High]',
'High Resp. Rate']

Actual_chart_expertKnowledge_MV = [
 'Height',
 'Height (cm)',
 'PSV Level',
 'Central Venous Pressure',
 'Arterial Blood Pressure diastolic',
 'Arterial Blood Pressure systolic',
 'Arterial Blood Pressure mean',
 'Non Invasive Blood Pressure systolic',
 'Non Invasive Blood Pressure diastolic',
 'Non Invasive Blood Pressure mean',
 'Ventilator Tank #1',
 'Ventilator Tank #2',
 'Respiratory Rate (Total)',
 'Respiratory Rate',
 'Respiratory Rate (spontaneous)',
 'Respiratory Rate (Set)',
 'Inspiratory Time',
 'Cuff Pressure',
 'Mean Airway Pressure',
 'Inspired O2 Fraction',
 'Total PEEP Level',
 'Minute Volume',
 'Daily Weight',
 'Admission Weight (Kg)',
 'Admission Weight (lbs.)',
 'Temperature Fahrenheit',
 'Tidal Volume (observed)',
 'Tidal Volume (spontaneous)',
 'Tidal Volume (set)',
 'PEEP set',
 'Peak Insp. Pressure',
 'Plateau Pressure',
 'Heart Rate',
 'Inspiratory Ratio',
 'O2 Flow',
 'Apnea Interval',
 'Expiratory Ratio',
 'O2 saturation pulseoxymetry',
 'SpO2 Desat Limit',
 'Heart rate Alarm - High',
 'Heart Rate Alarm - Low',
 'Fspn High',
 'Minute Volume Alarm - High',
 'Central Venous Pressure Alarm - High',
 'Arterial Blood Pressure Alarm - High',
 'Arterial Blood Pressure Alarm - Low',
 'Minute Volume Alarm - Low',
 'Central Venous Pressure  Alarm - Low',
 'Non-Invasive Blood Pressure Alarm - High',
 'Resp Alarm - Low',
 'Resp Alarm - High',
 'O2 Saturation Pulseoxymetry Alarm - High',
 'Vti High',
 'O2 Saturation Pulseoxymetry Alarm - Low',
 'Non-Invasive Blood Pressure Alarm - Low',
 'Paw High']

itemid_label_chart_CV = dict(zip(list(item_id_dbsource[item_id_dbsource.dbsource=='carevue'].itemid), list(item_id_dbsource[item_id_dbsource.dbsource=='carevue'].label)))
itemid_label_chart_MV = dict(zip(list(item_id_dbsource[item_id_dbsource.dbsource=='metavision'].itemid), list(item_id_dbsource[item_id_dbsource.dbsource=='metavision'].label)))
# ##breakpoint()
#
onlychart_MV = [str(k) for k,v in itemid_label_chart_MV.items() if v in Actual_chart_expertKnowledge_MV]
onlychart_CV = [str(k) for k,v in itemid_label_chart_CV.items() if v in Actual_chart_expertKnowledge_CV]

# getting the itemids to be used later
list_lab_ids = list(CV_full.columns)[:58-len(to_drops_labs_current)]
# onlychart_CV = list(CV_full.columns)[58-len(to_drops_labs_current):] # commenting since going to use the expert identified chart features
# onlychart_MV = list(MV_full.columns)[58-len(to_drops_labs_current):]
# ##breakpoint()
onlychart_binary_MV = [i for i in onlychart_MV if len(MV_full[i].value_counts()) < 3] # since metavision has all binaries
onlychart_cont_MV = [i for i in onlychart_MV if i not in onlychart_binary_MV]
onlychart_cat_CV = [ i for i in onlychart_CV if CV_full[i].dtypes == 'object']
onlychart_cont_CV = [i for i in onlychart_CV if CV_full[i].dtypes == 'float64']

#item_id_dbsource[item_id_dbsource['itemid'].isin([int(i) for i in onlychart_binary_MV])]
# item_id_dbsource[item_id_dbsource['itemid'].isin([int(i) for i in onlychart_cat_CV])]

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
# ##breakpoint()
# removing the rows that are beyond one to one matching
match_df.drop_duplicates(subset=['MV_itemids'], inplace=True)
# ##breakpoint()
CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

CV_itemids_to_drop = [i for i in CV_itemids_with_match_t if i not in CV_itemids_with_match]
# run the following routine once only since it is inplace
for i in CV_itemids_to_drop:
    try:
        onlychart_CV.remove(str(i))
        onlychart_cont_CV.remove(str(i))
    except:
        pass
# ##breakpoint()
""" Correlation computation """  # this is done because there are some repetititve lab features in chartevents
Cor_CV = CV_full[list_lab_ids+onlychart_cont_CV].corr()
Cor_MV = MV_full[list_lab_ids+onlychart_cont_MV].corr()

# computing the max in each row
high_cor_lab_chart_pairsCV = {}
for i in list_lab_ids:
    if Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1)[0] > 0.97:
        high_cor_lab_chart_pairsCV[i] = Cor_CV[i][Cor_CV[i].index.isin(onlychart_cont_CV)].nlargest(1).index[0]

high_cor_lab_chart_pairsMV = {}
for i in list_lab_ids:
    if Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1)[0] > 0.97:
        high_cor_lab_chart_pairsMV[i] = Cor_MV[i][Cor_MV[i].index.isin(onlychart_cont_MV)].nlargest(1).index[0]

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

# breakpoint()
#final matching dict
match_dic = dict(zip(CV_itemids_with_match, MV_itemids_with_match))

for id_col in CV_itemids_with_match:
    if id_col not in onlychart_cont_CV:
        # breakpoint()
        match_dic.pop(id_col)

# breakpoint()

# itemids with no match
CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]
MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]

print( " CV_itemids_with match ", len(CV_itemids_with_match))
print( " MV_itemids_with match ", len(MV_itemids_with_match))

print( " CV_itemids_with NO match ", len(CV_itemids_withnomatch))
print( " MV_itemids_with NO match ", len(MV_itemids_withnomatch))


num_xtra_feat_inX1 = len(CV_itemids_withnomatch)


""" partitioning both the datasets into train and holdout """
# this is being done at the start and for different number of mapped features and further sampling this will remain common.

# readmission variable was given a thought to be taken as outcome variable to be stratified on
# but since it was not available in both eras so didn't pursue further.

# df_train_CV, df_holdout_CV = model_selection.train_test_split(CV_full[list_lab_ids+onlychart_cont_CV], test_size=0.2,
#                                                                     random_state=42)
# df_train_MV, df_holdout_MV = model_selection.train_test_split(MV_full[list_lab_ids+onlychart_cont_MV], test_size=0.2,
#                                                                     random_state=42)
# ##breakpoint()
# exit()
full_data_CV0 = CV_full[list_lab_ids+onlychart_cont_CV]
full_data_MV0 = MV_full[list_lab_ids+onlychart_cont_MV]

upto_test_idx_CV = int(testDatasetSize * len(full_data_CV0))
df_holdout_CV = full_data_CV0.iloc[:upto_test_idx_CV]  # this part of the dataset wass not touched during HP tuning
df_train_CV = full_data_CV0.iloc[upto_test_idx_CV:] # this was the dataset which was divided into two parts for hp tuning; using it fully to train now

upto_test_idx_MV = int(testDatasetSize * len(full_data_MV0))
df_holdout_MV = full_data_MV0.iloc[:upto_test_idx_MV] # this part of the dataset wass not touched during HP tuning
df_train_MV = full_data_MV0.iloc[upto_test_idx_MV:] # this was the dataset which was divided into two parts for hp tuning; using it fully to train now


# this is only permutation case
# df_train_CV, df_holdout_CV = model_selection.train_test_split(CV_full[list_lab_ids+CV_itemids_with_match], test_size=0.2,
#                                                                     random_state=42)
# df_train_MV, df_holdout_MV = model_selection.train_test_split(MV_full[list_lab_ids+MV_itemids_with_match], test_size=0.2,
#                                                                     random_state=42)

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
        elif df_train_CV_proc.columns[i] in CV_itemids_with_match:
            if (match_dic[df_train_CV_proc.columns[i]] == df_train_MV_proc.columns[j]):
                P_x1[i, j] = 1
        elif (df_train_CV_proc.columns[i] in CV_itemids_withnomatch) & (
                df_train_MV_proc.columns[j] in MV_itemids_withnomatch):
            P_x1[i, j] = 2

P_x1_final = P_x1[len(list_lab_ids):, len(list_lab_ids):]


"""  Summary statistic based  """

stats_CV = total_CV_data_proc[onlychart_cont_CV].describe().swapaxes("index", "columns")
stats_CV['CoV'] = stats_CV['std']/stats_CV['mean']
stats_CV.drop(columns=['count'], inplace=True)

stats_MV = total_MV_data_proc[onlychart_cont_MV].describe().swapaxes("index", "columns")
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
    temp_inf_x1.loc[i, "ump_feature_in_X1"] = onlychart_cont_CV[i]
    temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(onlychart_cont_CV[i])]
    temp_inf_x1.loc[i, "match_byGS"] = onlychart_cont_MV[matched_index[0]]
    temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[int(onlychart_cont_MV[matched_index[0]])]
    temp_inf_x1.loc[i, "true_correlation"] = Cor_df.loc[onlychart_cont_CV[i], onlychart_cont_MV[matched_index[0]]]
    if np.any(P_x1_final[i] == 2):
        temp_inf_x1.loc[i, "no_match_or_not"] = 1
        temp_inf_x1.loc[i, "Correct_Match"] = "NA"
    else:
        temp_inf_x1.loc[i, "no_match_or_not"] = 0
        temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                                  match_df[
                                                      match_df['CV_itemids'] == temp_inf_x1.loc[
                                                          i, 'ump_feature_in_X1']][
                                                      'MV_itemids'])

print("\n \n List of mismatched feature number when # of mapped features are \n ")

print(" \n Mistakes by the Summary statistic method")
print(" Summary Statistic  X1_train mistakes number on holdout set",
      len(onlychart_cont_CV) - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
      len(onlychart_cont_CV) - num_xtra_feat_inX1)
print(" Summary Statistic   X2_train mistakes number on holdout set",
      len(onlychart_cont_CV) - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
      len(onlychart_cont_CV) - num_xtra_feat_inX1)

# evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                 list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                   list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))


# obtaining the index to compute the loss on incorrect or correct matches
correct_match_idx_orig_from_x1 = []
for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
    list(df_holdout_CV_proc.columns[len(list_lab_ids):]).index(i))
correct_match_idx_r_from_x1 = []
for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
    list(df_holdout_MV_proc.columns[len(list_lab_ids):]).index(i))

incorrect_match_idx_orig_from_x1 = []
for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
    list(df_holdout_CV_proc.columns[len(list_lab_ids):]).index(i))
incorrect_match_idx_r_from_x1 = []
for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
    list(df_holdout_MV_proc.columns[len(list_lab_ids):]).index(i))

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

KS_stat_matrix = np.zeros((len(onlychart_cont_CV), len(onlychart_cont_MV)))
KS_stat_pvalue = np.zeros((len(onlychart_cont_CV), len(onlychart_cont_MV)))

for cv_id in range(len(onlychart_cont_CV)):
    for mv_id in range(len(onlychart_cont_MV)):
        ks_sta_val, ks_pvalue = stats.kstest(total_CV_data_proc.iloc[:,len(list_lab_ids)+cv_id], total_MV_data_proc.iloc[:,len(list_lab_ids)+mv_id])
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
    temp_inf_x1_KS.loc[i, "ump_feature_in_X1"] = onlychart_cont_CV[i]
    temp_inf_x1_KS.loc[i, "CV_label"] = itemid_label_dict[int(onlychart_cont_CV[i])]
    temp_inf_x1_KS.loc[i, "match_byGS"] = onlychart_cont_MV[matched_index[0]]
    temp_inf_x1_KS.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[int(onlychart_cont_MV[matched_index[0]])]
    temp_inf_x1_KS.loc[i, "true_correlation"] = Cor_df.loc[onlychart_cont_CV[i], onlychart_cont_MV[matched_index[0]]]
    if np.any(P_x1_final[i] == 2):
        temp_inf_x1_KS.loc[i, "no_match_or_not"] = 1
        temp_inf_x1_KS.loc[i, "Correct_Match"] = "NA"
    else:
        temp_inf_x1_KS.loc[i, "no_match_or_not"] = 0
        temp_inf_x1_KS.loc[i, "Correct_Match"] = sum(temp_inf_x1_KS.loc[i, 'match_byGS'] ==
                                                  match_df[
                                                      match_df['CV_itemids'] == temp_inf_x1_KS.loc[
                                                          i, 'ump_feature_in_X1']][
                                                      'MV_itemids'])

print("\n \n List of mismatched feature number when # of mapped features are \n ")

print(" \n Mistakes by the KS statistic method")
print(" KS Statistic  X1_train mistakes number on holdout set",
      len(onlychart_cont_CV) - correct_with_match_from_x1_test_KS - num_xtra_feat_inX1, "out of ",
      len(onlychart_cont_CV) - num_xtra_feat_inX1)
print(" KS Statistic   X2_train mistakes number on holdout set",
      len(onlychart_cont_CV) - correct_with_match_from_x2_test_KS - num_xtra_feat_inX1, "out of ",
      len(onlychart_cont_CV) - num_xtra_feat_inX1)

# evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
correct_match_dict_x1_KS = dict(zip(list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 1]['ump_feature_in_X1']),
                                 list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 1]['match_byGS'])))
incorrect_match_dict_x1_KS = dict(zip(list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 0]['ump_feature_in_X1']),
                                   list(temp_inf_x1[temp_inf_x1_KS['Correct_Match'] == 0]['match_byGS'])))


# obtaining the index to compute the loss on incorrect or correct matches
correct_match_idx_orig_from_x1_KS = []
for i in list(correct_match_dict_x1_KS.keys()): correct_match_idx_orig_from_x1_KS.append(
    list(df_holdout_CV_proc.columns[len(list_lab_ids):]).index(i))
correct_match_idx_r_from_x1_KS = []
for i in list(correct_match_dict_x1_KS.values()): correct_match_idx_r_from_x1_KS.append(
    list(df_holdout_MV_proc.columns[len(list_lab_ids):]).index(i))

incorrect_match_idx_orig_from_x1_KS = []
for i in list(incorrect_match_dict_x1_KS.keys()): incorrect_match_idx_orig_from_x1_KS.append(
    list(df_holdout_CV_proc.columns[len(list_lab_ids):]).index(i))
incorrect_match_idx_r_from_x1_KS = []
for i in list(incorrect_match_dict_x1_KS.values()): incorrect_match_idx_r_from_x1_KS.append(
    list(df_holdout_MV_proc.columns[len(list_lab_ids):]).index(i))

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

# saving all the files

saving_dir = output_dir + 'UnivariateStats_basedcomparison_'+str(datetime.date.today()) +  '/MIMIC_data'+str(datetime.datetime.now())

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

temp_inf_x1.to_csv(saving_dir+"/Post-hoc_from_x1_Summary_Statisticbased_MIMIC.csv", index=False)
temp_inf_x1_KS.to_csv(saving_dir+"/Post-hoc_from_x1_KS_Statisticbased_MIMIC.csv", index=False)

file_name = saving_dir+"/Univariate_stats_based_MIMIC_mistakes.txt"

if os.path.exists(file_name):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name)

f = open(file_name, 'w')
f.write("\n \n MIMIC data results  ")
f.write("\n \n F1 for summary statistic based {0}".format(F1_fromx1))
f.write("\n \n F1 for KS statistic based {0}".format(F1_fromx1_KS))
f.write("\n \n ")
f.write("\n \n MatchMSE for summary statistic based {0}".format(overall_quality_error_matching_only))
f.write("\n \n MatchMSE for KS statistic based {0}".format(overall_quality_error_matching_only_KS))
f.write("\n \n ")
f.close()
