"""
This code reads the files that have all fraction of mistakes for every run of the experiment for all the features.
It also reads the MAtchMSe but the notations/variable name are kept same as for F1 score
Then it applies the wilcoxon signed rank test to prove the improvement due to different methods.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import random

random.seed(100)
np.random.seed(100)  # fixing the seed for reproducibility

# test_for = "vs_mapped"
# test_for = "binarized"
# test_for = "sq4"
# test_for = "varying_sample"
test_for = "onto"
# test_for = "PM"
# test_for = "Emb_PM"

# data details
dataset = 'ACTFAST_data'  # option list  {"MIMIC_data", 'ACTFAST_data', 'Nomao', 'superconductor', 'Syn_1', 'Syn_5'}

# metric to be used F1, MatchMSE, ReconMSE
metric = 'ReconMSE'

# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization

# model architecture and parameter details
if dataset in ["MIMIC_data", 'ACTFAST_data']:
    hidden_dim = 20
elif (test_for=='onto') or (test_for=='PM' and dataset in ['Nomao', 'superconductor']) :
    hidden_dim = 15
else:
    hidden_dim=5
# if test_for == "vs_mapped":
#     filename = "./F1_For_violin_Mismatch_metric_L_" + str(hidden_dim) + "_" + dataset + "_orthoStatus_" + str(
#         orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"
# else:
    # filename = "./F1_For_violin_Mismatch_metric_L_"+ str(hidden_dim) + "_Real_data_orthoStatus_" +str(orthogonalization_type) +"_BNStatus_"+str(batchnorm)+"_"+ str(test_for)+ ".txt"
filename = "./" +str(metric) +"_For_violin_Mismatch_metric_L_"+ str(hidden_dim) + "_" + dataset + "_orthoStatus_" +str(orthogonalization_type) +"_BNStatus_"+str(batchnorm)+"_"+ str(test_for)+ ".txt"


f = open(filename, 'r')
temp = f.readlines()
f.close()


# data extraction

Two_stage_X1 = []
KMF_X1 = []
Kang_X1 = []
CL_X1 = []
CL_Dec_X1 = []
CL_KMFl_X1 = []
CL_KMFen_X1 = []
RadialGAN_X1 = []

for i in range(len(temp)):
    if len(temp[i].split(" "))>1:
        if temp[i].split(" ")[1]=='X1_train':
            # Two_stage_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                Two_stage_X1.append(temp[j])
                j = j + 1
            print("Two stage ",Two_stage_X1)
        if temp[i].split(" ")[1]=='X1_train_sim_Cor' or temp[i].split(" ")[1]=='X1_train_Sim_cor':
            # KMF_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                KMF_X1.append(temp[j])
                j = j + 1
            print("KMF ",KMF_X1)
        if temp[i].split(" ")[1]=='X1_train_KANG':
            # Kang_X1 =[]
            j=i+1
            # breakpoint()
            print(temp[j])
            while list(temp[j])[1] != "X":
                Kang_X1.append(temp[j])
                j = j + 1
            print("Kang ",Kang_X1)
        if temp[i].split(" ")[1]=='X1_train_CL':
            # CL_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                CL_X1.append(temp[j])
                j=j+1
            print("CL ",CL_X1)
        if temp[i].split(" ")[1]=='X1_train_CL_Dec':
            # CL_Dec_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                CL_Dec_X1.append(temp[j])
                j=j+1
            print("CL_Dec ",CL_Dec_X1)
        if temp[i].split(" ")[1]=='X1_train_CL_Dec_KMFl':
            # CL_Dec_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                CL_KMFl_X1.append(temp[j])
                j=j+1
            print("CL_Dec_KMFl ",CL_KMFl_X1)
        if temp[i].split(" ")[1]=='X1_train_CL_Dec_KMFen':
            # CL_Dec_X1 =[]
            j=i+1
            while list(temp[j])[1] != "X":
                CL_KMFen_X1.append(temp[j])
                j=j+1
            print("CL_Dec_KMFen ",CL_KMFen_X1)
        if temp[i].split(" ")[1] == 'X1_train_RG':
            # RadialGAN =[]
            j = i + 1
            while list(temp[j])[1] != "X":
                RadialGAN_X1.append(temp[j])
                j = j + 1
            print("RadialGAN ", RadialGAN_X1)



Two_stage_X1_f =[]
KMF_X1_f = []
Kang_X1_f = []
RadialGAN_X1_f = []
CL_X1_f = []
CL_Dec_X1_f = []
CL_Dec_KMFl_X1_f = []
CL_Dec_KMFen_X1_f = []

for element in Two_stage_X1:
    for t in element.split():
        t = t.strip()
        if t != ']]':
            try:
                Two_stage_X1_f.append(float(t))
            except ValueError:
                try:
                    Two_stage_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    Two_stage_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])
                #             # print(j, temp)
                #         # print(temp)
                #         Two_stage_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "[" or "]":
                #                 temp = temp + str(list(t)[j])
                #             # print(j, temp)
                #         # print(temp)
                #         Two_stage_X1_f.append(float(temp))
                #         break
                pass
print(" Two stage final",Two_stage_X1_f)

for element in KMF_X1:
    for t in element.split():
        if t != ']]':
            try:
                KMF_X1_f.append(float(t))
            except ValueError:
                try:
                    KMF_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    KMF_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         KMF_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         KMF_X1_f.append(float(temp))
                #         break
                pass
print(" KMF final",KMF_X1_f)

for element in Kang_X1:
    for t in element.split():
        if t != ']]':
            try:
                Kang_X1_f.append(float(t))
            except ValueError:
                try:
                    Kang_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    Kang_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         Kang_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         Kang_X1_f.append(float(temp))
                #         break
                pass
print(" KANG final",Kang_X1_f)

for element in RadialGAN_X1:
    for t in element.split():
        try:
            RadialGAN_X1_f.append(float(t))
        except ValueError:
            try:
                RadialGAN_X1_f.append(float(t.strip('[[')))
            except ValueError:
                RadialGAN_X1_f.append(float(t.strip(']]')))
                pass
            pass
print(" RadialGAN final",RadialGAN_X1_f)

# testing here

for element in CL_X1:
    for t in element.split():
        if t != ']]':
            try:
                CL_X1_f.append(float(t))
            except ValueError:
                try:
                    CL_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    CL_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_X1_f.append(float(temp))
                #         break
                pass
print(" CL final",CL_X1_f)

for element in CL_Dec_X1:
    for t in element.split():
        if t != ']]':
            try:
                CL_Dec_X1_f.append(float(t))
            except ValueError:
                try:
                    CL_Dec_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    CL_Dec_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_X1_f.append(float(temp))
                #         break
                pass
print(" CL_Dec final",CL_Dec_X1_f)

for element in CL_KMFl_X1:
    for t in element.split():
        if t != ']]':
            try:
                CL_Dec_KMFl_X1_f.append(float(t))
            except ValueError:
                try:
                    CL_Dec_KMFl_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    CL_Dec_KMFl_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_KMFl_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_KMFl_X1_f.append(float(temp))
                #         break
                pass
print(" CL_Dec final",CL_Dec_KMFl_X1_f)

for element in CL_KMFen_X1:
    for t in element.split():
        if t != ']]':
            try:
                CL_Dec_KMFen_X1_f.append(float(t))
            except ValueError:
                try:
                    CL_Dec_KMFen_X1_f.append(float(t.strip('[[')))
                except ValueError:
                    CL_Dec_KMFen_X1_f.append(float(t.strip(']]')))
                    pass
                # for l in range(len(list(t))):
                #     if list(t)[l] == str(0):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(0)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_KMFen_X1_f.append(float(temp))
                #         break
                #     if list(t)[l] == str(1):
                #         # print(list(t))
                #         # print("blah",len(list(t)))
                #         temp = str(1)
                #         for j in range(l+1,len(list(t))-1):
                #             if list(t)[j] != "]" :
                #                 # print(list(t)[j])
                #                 temp = temp + str(list(t)[j])                        # print(j, temp)
                #         # print(temp)
                #         CL_Dec_KMFen_X1_f.append(float(temp))
                #         break
                pass
print(" CL_Dec final",CL_Dec_KMFen_X1_f)


print("Dataset no ", dataset, " with ", test_for, "L dimension", hidden_dim, "\n")

print("\n Average " + str(metric) +  " score across all possible partitions \n ")

print(" 2stage ChimericE ", np.mean(Two_stage_X1_f))
print(" KMF ", np.mean(KMF_X1_f))
print(" Kang", np.mean(Kang_X1_f))
print(" RadialGAN", np.mean(RadialGAN_X1_f))
print(" CL", np.mean(CL_X1_f))
print(" CL_Dec", np.mean(CL_Dec_X1_f))
print(" CL_Dec_KMFl", np.mean(CL_Dec_KMFl_X1_f))
print(" CL_Dec_KMFen", np.mean(CL_Dec_KMFen_X1_f))
# breakpoint()

print(" \n ********* P value ************ \n ")

_, p_2stage_vs_Kang_2sided = stats.wilcoxon(Two_stage_X1_f,Kang_X1_f)
_, p_2stage_vs_KMF_2sided = stats.wilcoxon(Two_stage_X1_f, KMF_X1_f)
_, p_2stage_vs_CL_2sided = stats.wilcoxon(Two_stage_X1_f, CL_X1_f)
_, p_2stage_vs_CLDec_2sided = stats.wilcoxon(Two_stage_X1_f, CL_Dec_X1_f)
_, p_2stage_vs_CL_DecKMFl_2sided = stats.wilcoxon(Two_stage_X1_f, CL_Dec_KMFl_X1_f)
_, p_2stage_vs_CL_DecKMFen_2sided = stats.wilcoxon(Two_stage_X1_f, CL_Dec_KMFen_X1_f)
_, p_KMF_vs_Kang_2sided = stats.wilcoxon(KMF_X1_f, Kang_X1_f)
_, p_KMF_vs_CL_2sided = stats.wilcoxon(KMF_X1_f, CL_X1_f)
_, p_KMF_vs_CL_Dec_2sided = stats.wilcoxon(KMF_X1_f, CL_Dec_X1_f)
_, p_KMF_vs_CL_DecKMFl_2sided = stats.wilcoxon(KMF_X1_f, CL_Dec_KMFl_X1_f)
_, p_KMF_vs_CL_DecKMFen_2sided = stats.wilcoxon(KMF_X1_f, CL_Dec_KMFen_X1_f)
_, p_CL_vs_CL_DecKMFen_2sided = stats.wilcoxon(CL_X1_f, CL_Dec_KMFen_X1_f)
_, p_CL_vs_CL_DecKMFl_2sided = stats.wilcoxon(CL_X1_f, CL_Dec_KMFl_X1_f)
_, p_CL_vs_CL_Dec_2sided = stats.wilcoxon(CL_X1_f, CL_Dec_X1_f)
_, p_CL_DecKMFl_vs_CL_DecKMFen_2sided = stats.wilcoxon(CL_Dec_KMFl_X1_f, CL_Dec_KMFen_X1_f)



print( "\n  2stage - Kang, two sided p value ", np.round(p_2stage_vs_Kang_2sided, decimals=3) )
print(" 2 stage - KMF, two sided p value  ", np.round(p_2stage_vs_KMF_2sided, decimals=3))
print(" 2 stage - CL, two sided p value  ", np.round(p_2stage_vs_CL_2sided, decimals=3))
print(" 2 stage - CLDec, two sided p value  ", np.round(p_2stage_vs_CLDec_2sided, decimals=3))
print(" 2 stage - CLDecKMFl, two sided p value  ", np.round(p_2stage_vs_CL_DecKMFl_2sided, decimals=3))
print(" 2 stage - CLDecKMFen, two sided p value  ", np.round(p_2stage_vs_CL_DecKMFen_2sided, decimals=3))
print(" KMF - Kang, two sided p value ", np.round(p_KMF_vs_Kang_2sided, decimals=3))
print(" KMF - CL, two sided p value ", np.round(p_KMF_vs_CL_2sided, decimals=3))
print(" KMF - CLDec, two sided p value ", np.round(p_KMF_vs_CL_Dec_2sided, decimals=3))
print(" KMF - CLDecKMFl, two sided p value ", np.round(p_KMF_vs_CL_DecKMFl_2sided, decimals=3))
print(" KMF - CLDecKMFen, two sided p value ", np.round(p_KMF_vs_CL_DecKMFen_2sided, decimals=3))
print(" CL - CLDecKMFen, two sided p value ", np.round(p_CL_vs_CL_DecKMFen_2sided, decimals=3))
print(" CL - CLDecKMFl, two sided p value ", np.round(p_CL_vs_CL_DecKMFl_2sided, decimals=3))
print(" CL - CLDec, two sided p value ", np.round(p_CL_vs_CL_Dec_2sided, decimals=3))
print(" CLDecKMFl - CLDecKMFen, two sided p value ", np.round(p_CL_DecKMFl_vs_CL_DecKMFen_2sided, decimals=3))



_, p_2stage_vs_Kang_1sided = stats.wilcoxon(Two_stage_X1_f,Kang_X1_f, alternative='less')
_, p_2stage_vs_KMF_1sided = stats.wilcoxon(Two_stage_X1_f, KMF_X1_f, alternative='less')
_, p_KMF_vs_Kang_1sided = stats.wilcoxon(KMF_X1_f, Kang_X1_f, alternative='less')

print( " \n 2stage - Kang < 0 (H1) p value ", np.round(p_2stage_vs_Kang_1sided, decimals=3) )
print(" 2 stage - KMF < 0 (H1) p value  ", np.round(p_2stage_vs_KMF_1sided, decimals=3))
print(" KMF - Kang < 0 (H1) p value ", np.round(p_KMF_vs_Kang_1sided, decimals=3))

_, p_2stage_vs_Kang_1sided_g = stats.wilcoxon(Two_stage_X1_f,Kang_X1_f, alternative='greater')
_, p_2stage_vs_KMF_1sided_g = stats.wilcoxon(Two_stage_X1_f, KMF_X1_f, alternative='greater')
_, p_KMF_vs_Kang_1sided_g = stats.wilcoxon(KMF_X1_f, Kang_X1_f, alternative='greater')

print( " \n 2stage - Kang > 0 (H1) p value ", np.round(p_2stage_vs_Kang_1sided_g, decimals=3)  )
print(" 2 stage - KMF > 0 (H1) p value  ", np.round(p_2stage_vs_KMF_1sided_g, decimals=3))
print(" KMF - Kang > 0 (H1) p value ", np.round(p_KMF_vs_Kang_1sided_g, decimals=3))

print("Finished")
