"""
This code reads the files that have all fraction of mistakes for every run of the experiment for all the features.
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

test_for = "vs_mapped"
# test_for = "binarized"
# test_for = "sq4"
# test_for = "varying_sample"
# test_for = "onto"
# test_for = "PM"

# data details
dataset = "MIMIC_data"

# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization

# model architecture and parameter details
hidden_dim = 20
if test_for == "vs_mapped":
    filename = "./F1_For_violin_Mismatch_metric_L_" + str(hidden_dim) + "_" + dataset + "_orthoStatus_" + str(
        orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"
else:
    filename = "./F1_For_violin_Mismatch_metric_L_"+ str(hidden_dim) + "_" + dataset + "_orthoStatus_" +str(orthogonalization_type) +"_BNStatus_"+str(batchnorm)+"_"+ str(test_for)+ ".txt"

f = open(filename, 'r')
temp = f.readlines()
f.close()


# data extraction

Two_stage_X1 = []
KMF_X1 = []
Kang_X1 = []
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
            print(temp[j])
            while list(temp[j])[1] != "X":
                Kang_X1.append(temp[j])
                j = j + 1
            print("Kang ",Kang_X1)
        # if temp[i].split(" ")[1]=='X1_train_RG':
        #     # RadialGAN_X1 =[]
        #     j=i+1
        #     while temp[j].split(" ")[-1] != "]]\n":
        #         RadialGAN_X1.append(temp[j])
        #         j=j+1
        #     RadialGAN_X1.append(temp[j])
        #     print("RadialGAN",RadialGAN_X1)

Two_stage_X1_f =[]
KMF_X1_f = []
Kang_X1_f = []
RadialGAN_X1_f = []

for element in Two_stage_X1:
    for t in element.split():
        try:
            Two_stage_X1_f.append(float(t))
        except ValueError:
            for l in range(len(list(t))):
                if list(t)[l] == str(0):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(0)
                    for j in range(l+1,len(list(t))-1):
                        if list(t)[j] != "]" :
                            # print(list(t)[j])
                            temp = temp + str(list(t)[j])
                        # print(j, temp)
                    # print(temp)
                    Two_stage_X1_f.append(float(temp))
                    break
                if list(t)[l] == str(1):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(1)
                    for j in range(l+1,len(list(t))-1):
                        if j != "[" or "]":
                            temp = temp + str(list(t)[j])
                        # print(j, temp)
                    # print(temp)
                    Two_stage_X1_f.append(float(temp))
                    break
            pass
print(" Two stage final",Two_stage_X1_f)

for element in KMF_X1:
    for t in element.split():
        try:
            KMF_X1_f.append(float(t))
        except ValueError:
            for l in range(len(list(t))):
                if list(t)[l] == str(0):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(0)
                    for j in range(l+1,len(list(t))-1):
                        if list(t)[j] != "]" :
                            # print(list(t)[j])
                            temp = temp + str(list(t)[j])                        # print(j, temp)
                    # print(temp)
                    KMF_X1_f.append(float(temp))
                    break
                if list(t)[l] == str(1):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(1)
                    for j in range(l+1,len(list(t))-1):
                        if list(t)[j] != "]" :
                            # print(list(t)[j])
                            temp = temp + str(list(t)[j])                        # print(j, temp)
                    # print(temp)
                    KMF_X1_f.append(float(temp))
                    break
            pass
print(" KMF final",KMF_X1_f)

for element in Kang_X1:
    for t in element.split():
        try:
            Kang_X1_f.append(float(t))
        except ValueError:
            for l in range(len(list(t))):
                if list(t)[l] == str(0):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(0)
                    for j in range(l+1,len(list(t))-1):
                        if list(t)[j] != "]" :
                            # print(list(t)[j])
                            temp = temp + str(list(t)[j])                        # print(j, temp)
                    # print(temp)
                    Kang_X1_f.append(float(temp))
                    break
                if list(t)[l] == str(1):
                    # print(list(t))
                    # print("blah",len(list(t)))
                    temp = str(1)
                    for j in range(l+1,len(list(t))-1):
                        if list(t)[j] != "]" :
                            # print(list(t)[j])
                            temp = temp + str(list(t)[j])                        # print(j, temp)
                    # print(temp)
                    Kang_X1_f.append(float(temp))
                    break
            pass
print(" KANG final",Kang_X1_f)

# for element in RadialGAN_X1:
#     for t in element.split():
#         try:
#             RadialGAN_X1_f.append(float(t))
#         except ValueError:
#             for l in range(len(list(t))):
#                 if list(t)[l] == str(0):
#                     # print(list(t))
#                     # print("blah",len(list(t)))
#                     temp = str(0)
#                     for j in range(l+1,len(list(t))-1):
#                         temp = temp + str(list(t)[j])
#                         # print(j, temp)
#                     # print(temp)
#                     RadialGAN_X1_f.append(float(temp))
#                     break
#                 if list(t)[l] == str(1):
#                     # print(list(t))
#                     # print("blah",len(list(t)))
#                     temp = str(1)
#                     for j in range(l+1,len(list(t))-1):
#                         temp = temp + str(list(t)[j])
#                         # print(j, temp)
#                     # print(temp)
#                     RadialGAN_X1_f.append(float(temp))
#                     break
#             pass
# print(" RadialGAN final",RadialGAN_X1_f)

# testing here


print("Dataset no ", dataset, " with ", test_for, "L dimension", hidden_dim, "\n")

print("\n Average F1 score across all possible partitions \n ")

print(" 2stage ChimericE ", np.mean(Two_stage_X1_f))
print(" KMF ", np.mean(KMF_X1_f))
print(" Kang", np.mean(Kang_X1_f))


print(" \n ********* P value ************ \n ")

_, p_2stage_vs_Kang_2sided = stats.wilcoxon(Two_stage_X1_f,Kang_X1_f)
_, p_2stage_vs_KMF_2sided = stats.wilcoxon(Two_stage_X1_f, KMF_X1_f)
_, p_KMF_vs_Kang_2sided = stats.wilcoxon(KMF_X1_f, Kang_X1_f)

print( "\n  2stage - Kang, two sided p value ", np.round(p_2stage_vs_Kang_2sided, decimals=3) )
print(" 2 stage - KMF, two sided p value  ", np.round(p_2stage_vs_KMF_2sided, decimals=3))
print(" KMF - Kang, two sided p value ", np.round(p_KMF_vs_Kang_2sided, decimals=3))

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
