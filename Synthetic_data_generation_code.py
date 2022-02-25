"""

this file is used to generate multidimentional synthetic dataset. The covariance matrix is generated using either of the following two methods

1) Factor method from https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices

2) https://stats.stackexchange.com/questions/215497/how-to-create-an-arbitrary-covariance-matrix


Dataset number 1: 2-cluster Gaussian simulated (20-D)
Dataset number 2: Multivariate Gaussian simulated (20-D)
Dataset number 3: 20-dimensionalGaussian dataset with covariance as identity matrix
Dataset number 5: Multivariate Gaussian simulated (50-D)
"""

# importing packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import  datetime


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


# fixing the seed for reproducibility
np.random.seed(100)

# input data
n = 5 # number of features; to be changed based on the dataset interested ot be generated
k = 10 # factor dimension
m = 5000  # double the number of samples
Dataset = 7  # number depends on the mean which you want, 3 is with identity correlation
num_various_seed_gen_data = 11

if Dataset ==1:
    # Dataset 1 parameters
    mean_pos = np.random.randint(10,20,n)
    mean_neg = np.random.randint(10,20,n)
    print("MEAN VALUES")
    print(mean_pos)
    print(mean_neg)
if Dataset ==2:
    # Dataset 2 parameters
    mean_pos = np.random.randint(10,20,n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)
if Dataset ==3:
    # Dataset 3 parameters
    mean_pos = np.random.randint(10,20,n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)
if Dataset ==4:
    # Dataset 4 parameters
    mean_pos = np.random.randint(40,100,n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)
if Dataset ==5:
    # Dataset 5 parameters, this dataset is supposed to be 50 dimensional
    mean_pos = np.random.randint(60,80,n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)
if Dataset ==6:
    # Dataset 6 parameters, this dataset is supposed to be 30 dimensional
    mean_pos = np.random.randint(2,8,n)
    mean_neg = np.random.randint(2,8,n)
    print("MEAN VALUES")
    print(mean_pos)
    print(mean_neg)
if Dataset ==8:
    # Dataset 8 parameters 20 dim
    mean_pos = np.random.randint(300,500,n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)

if Dataset == 7:
    # dataset number 7 parameters 5 dimension
    mean_pos = np.zeros(n)
    print("MEAN VALUES")
    print(mean_pos)
    mean_neg = mean_pos
    print(mean_neg)
    k=5

if Dataset in [1,2,4,5,6,8]:
    cov = Covaraince_matrix_factor_method(n,k)
elif Dataset == 7:
    cov = np.array([[1, 0.8, 0.9, 0.3, 0.9], [0.8, 1, 0.6, 0.4, 0.7], [0.9, 0.6, 1, 0.4, 0.8], [0.3, 0.4, 0.4, 1, 0.5],
                [0.9, 0.7, 0.8, 0.5, 1]])
else:
    cov = np.identity(n)
np.savetxt(str(datetime.date.today())+"Syn"+ str(Dataset) +"_Covariance_size_" +str(n) +"_original.csv",np.round(cov,2), delimiter=",")
print("Original covariance")
print(np.round(cov, 4))
for sample_num in range(num_various_seed_gen_data):
    dataset_pos = np.random.multivariate_normal(mean_pos,cov,m)
    dataset_neg = np.random.multivariate_normal(mean_neg,cov,m)

    # since right now we are generating a balanced dataset, taking a shortcut here by simply appending the positive and negative labels at the end of the dataframe
    df_pos = pd.DataFrame(dataset_pos)
    df_neg = pd.DataFrame(dataset_neg)

    df_pos = df_pos.assign(outcome=pd.Series(np.ones(len(df_pos))).values)
    df_neg = df_neg.assign(outcome=-pd.Series(np.ones(len(df_neg))).values)

    # combining the datframes and resetting the index to get a common index
    df_final = pd.concat([df_pos, df_neg])
    df_final.reset_index(drop = True, inplace=True)

    # renaming the column names
    column_name = ['Col'+ str(i) for i in range(1, n+2)]
    df_final.columns = column_name
    df_final.rename(columns={"Col"+str(n+1):"Y"}, inplace = True)

    # saving the dataset
    df_final.to_csv(str(datetime.date.today())+'Syn_Data_'+ str(Dataset) + "_Sample_no_" + str(sample_num+1) + "_size_" +str(n) +"_"+str(len(df_final))+"_for_AE_balanced.csv", index=False)

    # sanity check for the dataset generated
    u, s, v = np.linalg.svd(cov)
    est_cov = np.cov(np.transpose(df_final.values[:,:-1]))
    u1,s1, v1 = np.linalg.svd(est_cov)
    print("Estimated covariance for sample number ", sample_num+1)
    print(np.round(est_cov,2))
    print("Original eigen values \n", s)
    print("Estimated eigen values \n", s1)
    print("Estimated correlation for sample number ", sample_num + 1)
    cor = np.corrcoef(df_final.values[:,:-1].T)
    print(np.round(cor, 2))
print("End")




