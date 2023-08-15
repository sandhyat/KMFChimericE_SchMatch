"""
MIMIC dataset where the discharge summary embeddings are used as known mapped features and the chartevents are to be matched between two eras
list_lab_ids is kept in this code just for the sake of completenness otherwise it is not used in the computations for matching

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
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise,mutual_info_score,mean_squared_error
from sklearn.decomposition import PCA
from scipy import linalg, stats
import xgboost as xgb
#import shap
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
import pingouin as pg
import datetime
# from datetime import datetime
import json, sys, argparse


print(sys.getrecursionlimit())
sys.setrecursionlimit(3500)
print(sys.getrecursionlimit())

class JointLoss(torch.nn.Module):
    """
    Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
    """

    def __init__(self, options):
        super(JointLoss, self).__init__()
        # Assign options to self
        self.options = options
        # Batch size
        self.batch_size = options["batch_size"]
        # Temperature to use scale logits
        self.temperature = options["tau"]
        # Device to use: GPU or CPU
        self.device = options["device"]
        # initialize softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if options["cosine_similarity"] else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = torch.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(torch.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = torch.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def XNegloss(self, representation):
        # #breakpoint()
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        try:
            l_pos = torch.diag(similarity, self.batch_size)
        except RuntimeError:
            print("Error encountered. Debug.")
            #breakpoint()
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = torch.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = torch.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        closs = self.criterion(logits, labels)
        # # Loss per sample; this is being computed together in the main training loop
        # closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation):
        """
        Args:
            representation (torch.FloatTensor): representation is the projected latent value and latent is the output of the encoder
        """

        closs = self.XNegloss(representation)

        return closs

def generate_noisy_xbar(x, noise_type= "Zero-out", noise_level=0.1):
    """Generates noisy version of the samples x; Noise types: Zero-out, Gaussian, or Swap noise

    Args:
        x (np.ndarray): Input data to add noise to

    Returns:
        (np.ndarray): Corrupted version of input x

    """
    # Dimensions
    no, dim = x.shape
    # Initialize corruption array
    x_bar = torch.zeros_like(x)

    # Randomly (and column-wise) shuffle data
    if noise_type == "swap_noise":
        for i in range(dim):
            idx = torch.randperm(no)
            x_bar[:, i] = x[idx, i]
    # Elif, overwrite x_bar by adding Gaussian noise to x
    elif noise_type == "gaussian_noise":
        # #breakpoint()
        x_bar = x + torch.normal(0, noise_level, size = x.shape, device='cuda')
    else:
        x_bar = x_bar

    return x_bar


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
        self.hidden_dim = kwargs["hid_dim"]

        print("input_dimension_total", self.no_of_cont)
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=120)
        self.bn1 = nn.BatchNorm1d(num_features=120)
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_hidden_layer2 = nn.Linear(in_features=120, out_features=70)
        self.bn2 = nn.BatchNorm1d(num_features=70)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=70, out_features=self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(num_features=self.hidden_dim)
        self.decoder_hidden_layer1 = nn.Linear(in_features=hidden_dim, out_features=70)
        self.bn4 = nn.BatchNorm1d(num_features=70)
        self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_hidden_layer2 = nn.Linear(in_features=70, out_features=120)
        self.bn5 = nn.BatchNorm1d(num_features=120)
        self.drop_layer4 = nn.Dropout(p=self.drop_out_rate)
        self.decoder_output_layer = nn.Linear(in_features=120, out_features=kwargs["input_shape"])

    def forward(self, cont_data, cross=2):
        weights = torch.randn(1, device='cuda')
        if cross == 2:
            # this is basically to separate the encoder part of AE and use the Jacobian from encoder's output
            activation = self.encoder_hidden_layer1(cont_data)
            if self.batchnorm == 1:
                activation = self.bn1(activation)
            activation = self.encoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn2(activation)
            activation = torch.tanh(activation)
            # print(activation)
            # activation = F.prelu(activation, weight=weights)
            activation = self.drop_layer2(activation)
            code0 = self.encoder_output_layer(activation)
            return code0

        if  cross == 0:
            # print("inside the normal loop")
            activation = self.encoder_hidden_layer1(cont_data)
            if self.batchnorm == 1:
                activation = self.bn1(activation)
            activation = self.encoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn2(activation)
            activation = torch.tanh(activation)
            # print(activation)
            # activation = F.prelu(activation, weight=weights)
            activation = self.drop_layer2(activation)
            code0 = self.encoder_output_layer(activation)
            if self.batchnorm == 1:
                code0 = self.bn3(code0)
            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn4(activation)

            activation = torch.tanh(activation)
            # activation = F.prelu(activation, weight=weights)

            activation = self.drop_layer3(activation)
            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            reconstructed = self.decoder_output_layer(activation)
        if cross == 1:
            # print("inside the cross loop")
            code0 = cont_data
            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn4(activation)
            activation = torch.tanh(activation)
            # activation = F.prelu(activation, weight=weights)
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
            activation = torch.tanh(activation)
            activation = self.drop_layer2(activation)

            activation = self.encoder_hidden_layer3(activation)
            if self.batchnorm == 1:
                activation = self.bn3(activation)
            activation = torch.tanh(activation)
            activation = self.drop_layer3(activation)

            code0 = self.encoder_output_layer(activation)
            if self.batchnorm == 1:
                code0 = self.bn4(code0)

            activation = self.decoder_hidden_layer1(code0)
            if self.batchnorm == 1:
                activation = self.bn5(activation)
            activation = torch.tanh(activation)
            activation = self.drop_layer4(activation)

            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn6(activation)
            activation = torch.tanh(activation)
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
            activation = torch.tanh(activation)
            activation = self.drop_layer4(activation)

            activation = self.decoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn6(activation)
            activation = torch.tanh(activation)
            activation = self.drop_layer5(activation)

            activation = self.decoder_hidden_layer3(activation)
            if self.batchnorm == 1:
                activation = self.bn7(activation)
            reconstructed = self.decoder_output_layer(activation)

        return code0, reconstructed

class AE_2_hidden_layer_CL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.no_of_cont = kwargs["input_shape"]
        self.batchnorm = kwargs['batchnorm']
        self.drop_out_rate = kwargs["drop_out_p"]
        self.output_rep_dim = kwargs['repres_dim']  # dimension of the learnt representation

        # dropout at the start
        self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)
        print("input_dimension_total", self.no_of_cont)
        #*******
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=120)
        self.bn1 = nn.BatchNorm1d(num_features=120)
        self.encoder_hidden_layer2 = nn.Linear(in_features=120, out_features=70)
        self.bn2 = nn.BatchNorm1d(num_features=70)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=70, out_features=self.output_rep_dim)
        #*******
        # self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=200)
        # self.bn1 = nn.BatchNorm1d(num_features=200)
        # self.encoder_hidden_layer2 = nn.Linear(in_features=200, out_features=175)
        # self.bn2 = nn.BatchNorm1d(num_features=175)
        # self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        # self.encoder_hidden_layer3 = nn.Linear(in_features=175, out_features=150)
        # self.bn3 = nn.BatchNorm1d(num_features=150)
        # self.drop_layer3 = nn.Dropout(p=self.drop_out_rate)
        # self.encoder_hidden_layer4 = nn.Linear(in_features=150, out_features=125)
        # self.encoder_hidden_layer5 = nn.Linear(in_features=125, out_features=100)
        # self.encoder_output_layer = nn.Linear(in_features=100, out_features=self.output_rep_dim)

    def forward(self, data):

        data = self.drop_layer1(data)
        activation = self.encoder_hidden_layer1(data)
        if self.batchnorm == 1:
            activation = self.bn1(activation)
        activation = torch.tanh(self.encoder_hidden_layer2(activation))
        # if self.batchnorm == 1:
        #     activation = self.bn2(activation)
        # # activation = torch.sigmoid(activation)
        # activation = self.encoder_hidden_layer3(activation)
        # if self.batchnorm == 1:
        #     activation = self.bn3(activation)
        # activation = torch.sigmoid(activation)
        # activation = torch.sigmoid(self.encoder_hidden_layer4(activation))
        # activation = torch.sigmoid(self.encoder_hidden_layer5(activation))
        # activation = self.drop_layer2(activation)
        code0 = self.encoder_output_layer(activation)

        return code0

class AE_CL(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.hidden = kwargs["hidden_units"]
    self.hidden_final = kwargs["hidden_units_final"]
    self.hidden_depth = kwargs["hidden_depth"]
    self.input_size = kwargs["input_shape"]
    self.drop_out_rate = kwargs["drop_out_p"] # initial dropout rate

    self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)  # dropout layer just before the input layer to be applied to both views
    self.hidden_layers = torch.nn.ModuleList()
    ## always have at least 1 layer
    self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
    ## sizes for subsequent layers
    hiddensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype('int64')
    for thisindex in range(len(hiddensizes) - 1):
      self.hidden_layers.append(nn.Linear(in_features=hiddensizes[thisindex], out_features=hiddensizes[thisindex + 1]))

    self._reinitialize()

  def _reinitialize(self):
      """
      Tensorflow/Keras-like initialization
      """
      for name, p in self.named_parameters():
          if ('hidden' in name) or ('linear' in name):
              if 'weight' in name:
                  nn.init.xavier_uniform_(p.data)
              elif 'bias' in name:
                  p.data.fill_(0)

  def forward(self, data):
      data = self.drop_layer1(data)
      code0 = self.hidden_layers[0](data)
      if (len(self.hidden_layers) > 1):
          for thisindex in range(len(self.hidden_layers) - 1):
              code0 = torch.tanh(self.hidden_layers[thisindex + 1](code0))
      return code0

class AE_CL_withDec(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.hidden = kwargs["hidden_units"]
    self.hidden_final = kwargs["hidden_units_final"]
    self.hidden_depth = kwargs["hidden_depth"]
    self.input_size = kwargs["input_shape"]
    self.drop_out_rate = kwargs["drop_out_p"] # initial dropout rate

    self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)  # dropout layer just before the input layer to be applied to both views
    self.hidden_layers = torch.nn.ModuleList()
    ## always have at least 1 layer
    self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
    ## sizes for subsequent layers
    hiddensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype('int64')
    for thisindex in range(len(hiddensizes) - 1):
      self.hidden_layers.append(nn.Linear(in_features=hiddensizes[thisindex], out_features=hiddensizes[thisindex + 1]))

    self.hidden_layers_dec = torch.nn.ModuleList()
    hiddenSizes_dec  = np.ceil(np.linspace(start=self.hidden_final, stop=self.input_size, num= self.hidden_depth+1)).astype('int64')
    # breakpoint()
    for thisindex in range(1, len(hiddenSizes_dec) - 1):
        self.hidden_layers_dec.append(nn.Linear(in_features=hiddenSizes_dec[thisindex - 1], out_features=hiddenSizes_dec[thisindex]))
    self.hidden_layers_dec.append(nn.Linear(in_features=hiddenSizes_dec[-2], out_features=self.input_size))

    self._reinitialize()

  def _reinitialize(self):
      """
      Tensorflow/Keras-like initialization
      """
      for name, p in self.named_parameters():
          if ('hidden' in name) or ('linear' in name):
              if 'weight' in name:
                  nn.init.xavier_uniform_(p.data)
              elif 'bias' in name:
                  p.data.fill_(0)

  def forward(self, data, cross=0):
      if cross==0:
          data = self.drop_layer1(data)
          code0 = self.hidden_layers[0](data)
          if (len(self.hidden_layers) > 1):
              for thisindex in range(len(self.hidden_layers) - 1):
                  code0 = torch.tanh(self.hidden_layers[thisindex + 1](code0))

          # breakpoint()
          code1 = code0
          # code1 = torch.tanh(self.hidden_layers_dec[0](code1))
          if (len(self.hidden_layers_dec) > 1):
              for thisindex in range(len(self.hidden_layers_dec)-1):
                  if thisindex == len(self.hidden_layers_dec)-2:
                      code1 = torch.tanh(self.hidden_layers_dec[thisindex](code1))
                  else:
                      code1 = torch.relu(self.hidden_layers_dec[thisindex](code1))
          code1 = self.hidden_layers_dec[-1](code1)
          return code0, code1
      if cross==1:
          code1 = data
          # code1 = torch.tanh(self.hidden_layers_dec[0](code1))
          if (len(self.hidden_layers_dec) > 1):
              for thisindex in range(len(self.hidden_layers_dec)-1):
                  if thisindex == len(self.hidden_layers_dec)-2:
                      code1 = torch.tanh(self.hidden_layers_dec[thisindex](code1))
                  else:
                      code1 = torch.relu(self.hidden_layers_dec[thisindex](code1))
          code1 = self.hidden_layers_dec[-1](code1)
          return code1

def NTXentLoss(embeddings_knw_o, embeddings_knw_r, embeddings_unknw_o, embeddings_unknw_r, temperature = 0.1):  # embeddings from known features of both databases followed by the unknown features
    # compute the cosine similarity bu first normalizing and then matrix multiplying the known and unknown tensors
    cos_sim_o = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o), torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0,1)),temperature)
    cos_sim_or = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
                                        torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
                           temperature)
    cos_sim_r = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r), torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r),0,1)),temperature)
    cos_sim_ro = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
                                        torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
                           temperature)
    # for numerical stability  ## TODO update this logit name
    logits_max_o, _ = torch.max(cos_sim_o, dim=1, keepdim=True)
    logits_o = cos_sim_o - logits_max_o.detach()
    logits_max_or, _ = torch.max(cos_sim_or, dim=1, keepdim=True)
    logits_or = cos_sim_or - logits_max_or.detach()
    logits_max_r, _ = torch.max(cos_sim_r, dim=1, keepdim=True)
    logits_r = cos_sim_r - logits_max_r.detach()
    logits_max_ro, _ = torch.max(cos_sim_ro, dim=1, keepdim=True)
    logits_ro = cos_sim_ro - logits_max_ro.detach()


    # #breakpoint()
    if True:
        # computing the exp logits
        exp_o = torch.exp(logits_o)
        exp_r = torch.exp(logits_r)
        batch_loss_o =  - torch.log(exp_o.diag()/exp_o.sum(dim=0)).sum() - torch.log(exp_o.diag()/exp_o.sum(dim=1)).sum()
        batch_loss_r =  - torch.log(exp_r.diag()/exp_r.sum(dim=0)).sum() - torch.log(exp_r.diag()/exp_r.sum(dim=1)).sum()
        # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
        # since we are computing the rank on the similarity so higher the better
        avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_o.cpu().detach().numpy(), axis=1)) / len(cos_sim_o)
        avg_rank_cos_sim_r = np.trace(stats.rankdata(cos_sim_r.cpu().detach().numpy(), axis=1)) / len(cos_sim_r)

    # alternative way of computing the loss where the unknown feature part of the examples from the other database are treated as negative examples
    if False:
        cos_sim_combined = torch.concat([torch.concat([logits_o, logits_or], dim=1), torch.concat([logits_ro, logits_r], dim=1)], dim=0)
        exp_comb = torch.exp(cos_sim_combined)
        batch_loss = - torch.log(exp_comb.diag()/exp_comb.sum(dim=0)).sum() - torch.log(exp_comb.diag()/exp_comb.sum(dim=1)).sum()
        # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
        # since we are computing the rank on the similarity so higher the better
        # #breakpoint()
        avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_combined.cpu().detach().numpy(), axis=1))/len(cos_sim_combined)
        avg_rank_cos_sim_r = avg_rank_cos_sim_o
        batch_loss_o = batch_loss
        batch_loss_r = batch_loss

    # print("This batch's loss and avg rank ", batch_loss_o.item(), batch_loss_r.item(), avg_rank_cos_sim_o, avg_rank_cos_sim_r)
    return batch_loss_o, batch_loss_r, avg_rank_cos_sim_o, avg_rank_cos_sim_r
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    # print(c_xy)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def Train_CL(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,  Df_holdout_orig, DF_holdout_r, partition, trial):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig)

    num_features = len(reordered_column_names_r)
    num_NonCat_features_r = len(reordered_column_names_r)

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures

    dataset_orig = TabularDataset(data=df_train_preproc)
    train_loader_orig = DataLoader(dataset_orig, batchSize, shuffle=True, num_workers=1)
    dataset_orig_val = TabularDataset(data=Df_holdout_orig)
    val_loader_orig = DataLoader(dataset_orig_val, batchSize, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc)
    train_loader_r = DataLoader(dataset_r, batchSize, shuffle=True, num_workers=1)
    dataset_r_val = TabularDataset(data=DF_holdout_r)
    val_loader_r = DataLoader(dataset_r_val, batchSize, shuffle=True, num_workers=1)

    known_features_encoder = AE_CL(input_shape=mpfeatures, hidden_units_final=encKnwMapWidthFinal,
                                                  hidden_depth=encKnwMapDepth,
                                                  hidden_units=encKnwMapWidth, drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_orig = AE_CL(input_shape=num_NonCat_features_orig - mpfeatures, hidden_units_final=encUknwD1OrigWidthFinal,
                                                  hidden_depth=encUknwD1OrigDepth,
                                                  hidden_units=encUknwD1OrigWidth, drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_r = AE_CL(input_shape=num_NonCat_features_r - mpfeatures, hidden_units_final=encUknwD2ReWidthFinal,
                                                  hidden_depth=encUknwD2ReDepth,
                                                  hidden_units=encUknwD2ReWidth, drop_out_p=dropout_rate_CL).to(device)

    # #breakpoint()

    withinCL_options = {'batch_size': batchSize, "tau": tau, "device": device, "cosine_similarity": True}
    aug_loss = JointLoss(withinCL_options)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_known = optim.Adam(known_features_encoder.parameters(), lr=learningRate, weight_decay=1e-5)
    optimizer_unk_orig = optim.Adam(unknown_features_encoder_orig.parameters(), lr=learningRate, weight_decay=1e-5)
    optimizer_unk_r = optim.Adam(unknown_features_encoder_r.parameters(), lr=learningRate, weight_decay=1e-5)

    # lr scheduler
    scheduler_known = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_known, patience=LRPatience, verbose=True)
    scheduler_unk_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_orig, patience=LRPatience, verbose=True)
    scheduler_unk_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_r, patience=LRPatience, verbose=True)

    # lr scheduler
    # scheduler_known = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_known, mode='max',  patience=2, verbose=True) # mode= max because higher F1 score is better
    # scheduler_unk_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_orig, mode='max',patience=2, verbose=True)
    # scheduler_unk_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_r, mode='max', patience=2, verbose=True)

    if False:
        # Debugging to see where the embedding are positioned in the latent space
        rep_known_val_o_b = known_features_encoder(
            torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_o_b = unknown_features_encoder_orig(
            torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))

        rep_known_val_r_b = known_features_encoder(torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_r_b = unknown_features_encoder_r(
            torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device))

        # checking the cone effect
        # pair_wise_similarity(rep_known_val_o_b.cpu().detach().numpy(), 'known_o', 'before')
        # pair_wise_similarity(rep_known_val_r_b.cpu().detach().numpy(), 'known_r', 'before')
        # pair_wise_similarity(rep_unknown_val_o_b.cpu().detach().numpy(), 'unknown_o', 'before')
        # pair_wise_similarity(rep_unknown_val_r_b.cpu().detach().numpy(), 'unknown_r', 'before')

        # before training
        points_to_plot = 200  # len(rep_known_val_o.cpu().detach().numpy())
        features_2d_o = svd(np.concatenate([rep_known_val_o_b.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_o_b.cpu().detach().numpy()[:points_to_plot]], 0))
        features_2d_r = svd(np.concatenate([rep_known_val_r_b.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_r_b.cpu().detach().numpy()[:points_to_plot]], 0))

        plt.figure(figsize=(5, 5))
        plt.scatter(features_2d_o[:-points_to_plot, 0], features_2d_o[:-points_to_plot, 1], c='red')  # known mapped
        plt.scatter(features_2d_o[-points_to_plot:, 0], features_2d_o[-points_to_plot:, 1], c='blue')  # unknown
        plt.scatter(features_2d_r[:-points_to_plot, 0], features_2d_r[:-points_to_plot, 1], c='magenta')  # known mapped
        plt.scatter(features_2d_r[-points_to_plot:, 0], features_2d_r[-points_to_plot:, 1], c='green')  # unknown
        # connect the dots
        for i in range(points_to_plot):
            plt.plot([features_2d_o[i, 0], features_2d_o[points_to_plot + i, 0]],
                     [features_2d_o[i, 1], features_2d_o[points_to_plot + i, 1]], c='black', alpha=0.1)
            plt.plot([features_2d_r[i, 0], features_2d_r[points_to_plot + i, 0]],
                     [features_2d_r[i, 1], features_2d_r[points_to_plot + i, 1]], c='black', alpha=0.1)
        plt.xlabel(" SVD1 ")
        plt.ylabel(" SVD2 ")
        plt.title("Before training")
        plt.savefig(
            saving_dir + "/Before_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".png")
        plt.savefig(
            saving_dir + "/Before_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".pdf")
        plt.close()

    total_loss = []
    for epoch in range(epochs):
        loss_tr_o = 0
        loss_tr_r = 0
        rank_tr_o = 0
        rank_tr_r = 0
        counting_flag_for_rank = 0

        known_features_encoder.train()
        unknown_features_encoder_orig.train()
        unknown_features_encoder_r.train()
        for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
            if len(data[0][1]) == len(data[1][1]):
                # seperate the known and unknown features here  for passing to their corresponding encoders
                x_kn_o = data[0][1][:, :mpfeatures].to(device)
                x_kn_r = data[1][1][:, :mpfeatures].to(device)
                x_unkn_o = data[0][1][:, mpfeatures:].to(device)
                x_unkn_r = data[1][1][:, mpfeatures:].to(device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer_known.zero_grad()
                optimizer_unk_orig.zero_grad()
                optimizer_unk_r.zero_grad()

                datalist = [x_kn_o, x_kn_r, x_unkn_o, x_unkn_r]  # this step could be avoided

                if True: # adding self augmentation
                    for dt in range(4):
                        x_bar = datalist[dt]
                        x_bar_noisy = generate_noisy_xbar(x_bar)
                        # Generate binary mask
                        mask = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                        mask1 = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                        # #breakpoint()
                        # Replace selected x_bar features with the noisy ones
                        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                        x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                        # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                        datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)


                # #breakpoint()

                # computing the encoded values
                known_rep_o = known_features_encoder(datalist[0])
                known_rep_r = known_features_encoder(datalist[1])
                unknown_rep_o = unknown_features_encoder_orig(datalist[2])
                unknown_rep_r = unknown_features_encoder_r(datalist[3])

                # embeddings from known features of both databases followed by the unknown features
                contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r = NTXentLoss(known_rep_o, known_rep_r,
                                                                                            unknown_rep_o,
                                                                                            unknown_rep_r, tau)



                contrastive_loss = contrastive_loss_o + contrastive_loss_r
                if True: # adding self augmentation
                    self_aug_loss_un = aug_loss(unknown_rep_o) + aug_loss(unknown_rep_r) + aug_loss(
                        known_rep_o) + aug_loss(known_rep_r)
                    contrastive_loss = contrastive_loss + self_aug_loss_un

                # compute accumulated gradients
                contrastive_loss.backward()

                # perform parameter update based on current gradients
                optimizer_known.step()
                optimizer_unk_orig.step()
                optimizer_unk_r.step()

                # add the mini-batch training loss to epoch loss
                loss_tr_o += contrastive_loss_o.item()
                loss_tr_r += contrastive_loss_r.item()
                if (data[0][1].shape[0] == batchSize):
                    rank_tr_o += avg_Rank_o.item()
                    rank_tr_r += avg_Rank_r.item()
                    counting_flag_for_rank = counting_flag_for_rank + 1

        # compute the epoch training loss
        loss_tr_o = loss_tr_o / (len(train_loader_orig))
        loss_tr_r = loss_tr_r / (len(train_loader_r))
        rank_tr_o = rank_tr_o / (
            counting_flag_for_rank)  # dividing by counting_flag_for_rank because the avg rank from all batches in not included
        rank_tr_r = rank_tr_r / (counting_flag_for_rank)

        # #breakpoint()

        with torch.no_grad():
            # computing the representations from the trained encoders
            known_features_encoder.eval()
            unknown_features_encoder_orig.eval()
            unknown_features_encoder_r.eval()

            if True:
                loss_val = 0
                loss_val_o = 0
                loss_val_r = 0
                rank_val_o = 0
                rank_val_r = 0
                counting_flag_for_rank_val = 0
                within_unkn_CL_loss_val = 0

                for i, data in enumerate(zip(val_loader_orig, val_loader_r)):
                    if len(data[0][1]) == len(data[1][1]):
                        # seperate the known and unknown features here  for passing to their corresponding encoders
                        x_kn_o = data[0][1][:, :mpfeatures].to(device)
                        x_kn_r = data[1][1][:, :mpfeatures].to(device)
                        x_unkn_o = data[0][1][:, mpfeatures:].to(device)
                        x_unkn_r = data[1][1][:, mpfeatures:].to(device)

                        datalist = [x_kn_o, x_kn_r, x_unkn_o, x_unkn_r]  # this step could be avoided

                        if True:  # adding self augmentation
                            for dt in range(4):
                                x_bar = datalist[dt]
                                x_bar_noisy = generate_noisy_xbar(x_bar)
                                # Generate binary mask
                                mask = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                                mask1 = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                                # #breakpoint()
                                # Replace selected x_bar features with the noisy ones
                                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                                x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                                # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                                datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)

                        # computing the encoded values
                        known_rep_o = known_features_encoder(datalist[0])
                        known_rep_r = known_features_encoder(datalist[1])
                        unknown_rep_o = unknown_features_encoder_orig(datalist[2])
                        unknown_rep_r = unknown_features_encoder_r(datalist[3])

                        # #breakpoint()
                        # embeddings from known features of both databases followed by the unknown features
                        contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r  = NTXentLoss(known_rep_o, known_rep_r, unknown_rep_o, unknown_rep_r, tau)

                        contrastive_loss_val = contrastive_loss_o + contrastive_loss_r

                        if True:  # adding self augmentation
                            self_aug_loss_un = aug_loss(unknown_rep_o) + aug_loss(unknown_rep_r) + aug_loss(
                                known_rep_o) + aug_loss(known_rep_r)
                            contrastive_loss_val = contrastive_loss_val + self_aug_loss_un
                            within_unkn_CL_loss_val += self_aug_loss_un.item()

                        # add the mini-batch training loss to epoch loss
                        loss_val += contrastive_loss_val.item()

                        # add the mini-batch training loss to epoch loss
                        loss_val_o += contrastive_loss_o.item()
                        loss_val_r += contrastive_loss_r.item()
                        if (data[0][1].shape[0] == batchSize):
                            rank_val_o += avg_Rank_o.item()
                            rank_val_r += avg_Rank_r.item()
                            counting_flag_for_rank_val = counting_flag_for_rank_val + 1

                loss_val = loss_val / (len(val_loader_orig) + len(val_loader_r))
                within_unkn_CL_loss_val = within_unkn_CL_loss_val / (len(val_loader_orig) + len(val_loader_r))

                # compute the epoch training loss
                loss_val_o = loss_val_o / (len(val_loader_orig))
                loss_val_r = loss_val_r / (len(val_loader_r))
                rank_val_o = rank_val_o / (
                    counting_flag_for_rank_val)  # dividing by counting_flag_for_rank because the avg rank from all batches in not included
                rank_val_r = rank_val_r / (counting_flag_for_rank_val)

                # # display the epoch training loss
                # print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, epochs, loss_tr,
                #                                                                                loss_val))

                # display the epoch training loss
                print(
                    "Validation performance epoch : {}/{}, loss_o = {:.5f}, loss_r = {:.5f}, within cont loss = {:.5f},  avgbatchwise_rank_o = {:.5f}, avgbatchwise_rank_r = {:.5f}".format(
                        epoch + 1, epochs, loss_val_o, loss_val_r, within_unkn_CL_loss_val, rank_val_o, rank_val_r))
            else:
                for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
                temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)
                grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
                for i in range(temp_input.shape[0]):
                    grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])

                grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

                for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
                temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device)
                grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
                for i in range(temp_input.shape[0]):
                    grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])

                grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

                o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                                        np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                                        dense_output=True)
                r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                                        np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                                        dense_output=True)

                # Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(
                #     o_to_r_sim,
                #     r_to_o_sim,
                #     index_for_mapping_orig_to_rename[len(mapped_features):],
                #     index_for_mapping_rename_to_orig[len(mapped_features):],
                #     len(mapped_features))

                correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
                    o_to_r_sim,
                    r_to_o_sim,
                    P_x1, len(mapped_features))

                print(" CL  X1_train mistakes number on holdout set",
                      unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
                      unmapped_features_orig - num_xtra_feat_inX1)

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

                F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)

                # display the epoch training loss
                print("Training loss epoch : {}/{}, loss_o = {:.5f}, loss_r = {:,.5f}  F1 on validation = {:.5f}".format(
                        epoch + 1, epochs, loss_tr_o, loss_tr_r, F1_fromx1))

        scheduler_known.step(loss_val)
        scheduler_unk_orig.step(loss_val)
        scheduler_unk_r.step(loss_val)


    if False:
        # Debugging to see where the embedding are positioned in the latent space
        rep_known_val_o = known_features_encoder(torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_o = unknown_features_encoder_orig(
            torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))
        # rep_unknown_val_o_from_r = unknown_features_encoder_r(
        #     torch.Tensor(Df_holdout_orig0.reindex(columns=reordered_column_names_r).iloc[:, mpfeatures:-1].values).to(
        #         device))

        rep_known_val_r = known_features_encoder(torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_r = unknown_features_encoder_r(
            torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device))
        # rep_unknown_val_r_from_orig = unknown_features_encoder_orig(
        #     torch.Tensor(DF_holdout_r0.iloc[:, mpfeatures:-1].values).to(device))  # r0 is the one that was not permuted

        # checking the cone effect
        # pair_wise_similarity(rep_known_val_o.cpu().detach().numpy(), 'known_o', 'after')
        # pair_wise_similarity(rep_known_val_r.cpu().detach().numpy(), 'known_r', 'after')
        # pair_wise_similarity(rep_unknown_val_o.cpu().detach().numpy(), 'unknown_o', 'after')
        # pair_wise_similarity(rep_unknown_val_r.cpu().detach().numpy(), 'unknown_r', 'after')

        ## checking for the modality gap of two databases on the same plot
        # #breakpoint()
        points_to_plot = 200  # len(rep_known_val_o.cpu().detach().numpy())
        features_2d_o = svd(np.concatenate([rep_known_val_o.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_o.cpu().detach().numpy()[:points_to_plot]], 0))
        features_2d_r = svd(np.concatenate([rep_known_val_r.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_r.cpu().detach().numpy()[:points_to_plot]], 0))

        plt.figure(figsize=(5, 5))
        plt.scatter(features_2d_o[:-points_to_plot, 0], features_2d_o[:-points_to_plot, 1], c='red')  # known mapped
        plt.scatter(features_2d_o[-points_to_plot:, 0], features_2d_o[-points_to_plot:, 1], c='blue')  # unknown
        plt.scatter(features_2d_r[:-points_to_plot, 0], features_2d_r[:-points_to_plot, 1], c='magenta')  # known mapped
        plt.scatter(features_2d_r[-points_to_plot:, 0], features_2d_r[-points_to_plot:, 1], c='green')  # unknown
        # connect the dots
        for i in range(points_to_plot):
            plt.plot([features_2d_o[i, 0], features_2d_o[points_to_plot + i, 0]],
                     [features_2d_o[i, 1], features_2d_o[points_to_plot + i, 1]], c='black', alpha=0.1)
            plt.plot([features_2d_r[i, 0], features_2d_r[points_to_plot + i, 0]],
                     [features_2d_r[i, 1], features_2d_r[points_to_plot + i, 1]], c='black', alpha=0.1)
        plt.xlabel(" SVD1 ")
        plt.ylabel(" SVD2 ")
        plt.title("After training")

        plt.savefig(
            saving_dir + "/After_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) +".png")
        plt.savefig(
            saving_dir + "/After_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".pdf")
        plt.close()

        breakpoint()

        """ plotting the summary importance """

        background_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[:10, :mpfeatures].values),
                                         torch.Tensor(DF_holdout_r.iloc[:10, :mpfeatures].values)], dim=0).to(device)
        shap_to_use_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[10:100, :mpfeatures].values),
                                          torch.Tensor(DF_holdout_r.iloc[10:100, :mpfeatures].values)], dim=0).to(device)
        en_known = shap.DeepExplainer(known_features_encoder, background_known)
        shap_vals_knwn_comb = en_known.shap_values(shap_to_use_known)

        # array_shap_dim = np.zeros((len(shap_vals_knwn_comb), mpfeatures))
        # for i in range(len(shap_vals_knwn_comb)): array_shap_dim[i] = np.mean(
        #     np.absolute(np.transpose(shap_vals_knwn_comb[i])), 1)
        # #breakpoint()
        shap.summary_plot(shap_vals_knwn_comb, feature_names=[itemid_label_dict[int(i)] for i in mapped_features],
                          show=False, max_display=len(mapped_features))
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Known Encoder using both data sources ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) +  "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

        en_unkn_orig = shap.DeepExplainer(unknown_features_encoder_orig,
                                          torch.Tensor(Df_holdout_orig.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_orig = en_unkn_orig.shap_values(
            torch.Tensor(Df_holdout_orig.iloc[10:100, mpfeatures:].values).to(device))

        array_shap_dimUn_orig = np.zeros((len(shap_vals_unkn_orig), len(reordered_column_names_orig[mpfeatures:])))
        for i in range(len(shap_vals_unkn_orig)): array_shap_dimUn_orig[i] = np.mean(np.absolute(np.transpose(shap_vals_unkn_orig[i])), 1)

        shap.summary_plot(shap_vals_unkn_orig,
                          feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_orig[mpfeatures:]], show=False,
                          max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder Original ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

        en_unkn_r = shap.DeepExplainer(unknown_features_encoder_r,
                                       torch.Tensor(DF_holdout_r.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_r = en_unkn_r.shap_values(torch.Tensor(DF_holdout_r.iloc[10:100, mpfeatures:].values).to(device))

        array_shap_dimUn_r = np.zeros((len(shap_vals_unkn_r), len(reordered_column_names_r[mpfeatures:])))
        for i in range(len(shap_vals_unkn_r)): array_shap_dimUn_r[i] = np.mean(np.absolute(np.transpose(shap_vals_unkn_r[i])), 1)

        shap.summary_plot(shap_vals_unkn_r, feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_r[mpfeatures:]],
                          show=False, max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder R ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

    # #breakpoint()
    # computing the gradient of the output wrt the input data
    # #breakpoint()


    for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)
    grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]):
        grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device)
    grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]):
        grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    # np.savetxt(output_dir+"Grad_unknwn_Orig_" + str(randomSeed) + "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_o.cpu().detach().numpy(), delimiter=",")
    # np.savetxt(output_dir+"Grad_unknwn_R_" + str(randomSeed) +  "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_r.cpu().detach().numpy(), delimiter=",")
    # exit()

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)

    # Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(
    #     o_to_r_sim,
    #     r_to_o_sim,
    #     index_for_mapping_orig_to_rename[len(mapped_features):],
    #     index_for_mapping_rename_to_orig[len(mapped_features):],
    #     len(mapped_features))

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
        o_to_r_sim,
        r_to_o_sim,
        P_x1, len(mapped_features))
    # #breakpoint()
    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match" , 'Shap_Contr', 'Shap_rank', 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'Shap_Contr', 'Shap_rank', 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    # #breakpoint()
    for i in range(x1_match_matrix_test.shape[0]):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if
                         x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        # temp_inf_x1.loc[i, 'Shap_Contr'] =
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])
    # temp_inf_x1['Shap_Contr'] = np.sum(array_shap_dimUn_orig, 0)
    # temp_inf_x1['Shap_rank'] = temp_inf_x1['Shap_Contr'].rank(ascending=False)

    # #breakpoint()
    for i in range(x2_match_matrix_test.shape[0]):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if
                         x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])
    # temp_inf_x2['Shap_Cont'] = np.sum(array_shap_dimUn_r, 0)

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

    print(" \n Mistakes by the CL method on holdout data")
    print(" CL  X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" CL  X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])

    print(" -------- CL method training ends ------------- \n \n  ")


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


    # encoding to be used later
    known_features_encoder.eval()
    # training dataset
    rep_known_val_o = known_features_encoder(
        torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device)).cpu().detach().numpy()
    rep_known_val_r = known_features_encoder(
        torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device)).cpu().detach().numpy()

    del df_rename_preproc
    print("F1 from CL ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only)
    # #breakpoint()
    # exit()
    return grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test, correct_with_match_from_x2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r, overall_quality_error_matching_only

def Train_CL_withDec(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,  Df_holdout_orig, DF_holdout_r, partition, trial):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig)

    num_features = len(reordered_column_names_r)
    num_NonCat_features_r = len(reordered_column_names_r)

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures

    dataset_orig = TabularDataset(data=df_train_preproc)
    train_loader_orig = DataLoader(dataset_orig, batchSize, shuffle=True, num_workers=1)
    dataset_orig_val = TabularDataset(data=Df_holdout_orig)
    val_loader_orig = DataLoader(dataset_orig_val, batchSize, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc)
    train_loader_r = DataLoader(dataset_r, batchSize, shuffle=True, num_workers=1)
    dataset_r_val = TabularDataset(data=DF_holdout_r)
    val_loader_r = DataLoader(dataset_r_val, batchSize, shuffle=True, num_workers=1)

    known_features_encoder = AE_CL_withDec(input_shape=mpfeatures, hidden_units_final=encKnwMapWidthFinal,
                                                  hidden_depth=encKnwMapDepth,
                                                  hidden_units=encKnwMapWidth, drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_orig = AE_CL_withDec(input_shape=num_NonCat_features_orig - mpfeatures, hidden_units_final=encUknwD1OrigWidthFinal,
                                                  hidden_depth=encUknwD1OrigDepth,
                                                  hidden_units=encUknwD1OrigWidth, drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_r = AE_CL_withDec(input_shape=num_NonCat_features_r - mpfeatures, hidden_units_final=encUknwD2ReWidthFinal,
                                                  hidden_depth=encUknwD2ReDepth,
                                                  hidden_units=encUknwD2ReWidth, drop_out_p=dropout_rate_CL).to(device)

    # #breakpoint()

    withinCL_options = {'batch_size': batchSize, "tau": tau, "device": device, "cosine_similarity": True}
    aug_loss = JointLoss(withinCL_options)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_known = optim.Adam(known_features_encoder.parameters(), lr=learningRate, weight_decay=1e-5)
    optimizer_unk_orig = optim.Adam(unknown_features_encoder_orig.parameters(), lr=learningRate, weight_decay=1e-5)
    optimizer_unk_r = optim.Adam(unknown_features_encoder_r.parameters(), lr=learningRate, weight_decay=1e-5)

    # lr scheduler
    scheduler_known = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_known, patience=LRPatience, verbose=True)
    scheduler_unk_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_orig, patience=LRPatience, verbose=True)
    scheduler_unk_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_r, patience=LRPatience, verbose=True)

    rec_criterion = nn.MSELoss()
    # lr scheduler
    # scheduler_known = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_known, mode='max',  patience=2, verbose=True) # mode= max because higher F1 score is better
    # scheduler_unk_orig = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_orig, mode='max',patience=2, verbose=True)
    # scheduler_unk_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unk_r, mode='max', patience=2, verbose=True)

    if False:
        # Debugging to see where the embedding are positioned in the latent space
        rep_known_val_o_b = known_features_encoder(
            torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_o_b = unknown_features_encoder_orig(
            torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))

        rep_known_val_r_b = known_features_encoder(torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_r_b = unknown_features_encoder_r(
            torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device))

        # checking the cone effect
        # pair_wise_similarity(rep_known_val_o_b.cpu().detach().numpy(), 'known_o', 'before')
        # pair_wise_similarity(rep_known_val_r_b.cpu().detach().numpy(), 'known_r', 'before')
        # pair_wise_similarity(rep_unknown_val_o_b.cpu().detach().numpy(), 'unknown_o', 'before')
        # pair_wise_similarity(rep_unknown_val_r_b.cpu().detach().numpy(), 'unknown_r', 'before')

        # before training
        points_to_plot = 200  # len(rep_known_val_o.cpu().detach().numpy())
        features_2d_o = svd(np.concatenate([rep_known_val_o_b.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_o_b.cpu().detach().numpy()[:points_to_plot]], 0))
        features_2d_r = svd(np.concatenate([rep_known_val_r_b.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_r_b.cpu().detach().numpy()[:points_to_plot]], 0))

        plt.figure(figsize=(5, 5))
        plt.scatter(features_2d_o[:-points_to_plot, 0], features_2d_o[:-points_to_plot, 1], c='red')  # known mapped
        plt.scatter(features_2d_o[-points_to_plot:, 0], features_2d_o[-points_to_plot:, 1], c='blue')  # unknown
        plt.scatter(features_2d_r[:-points_to_plot, 0], features_2d_r[:-points_to_plot, 1], c='magenta')  # known mapped
        plt.scatter(features_2d_r[-points_to_plot:, 0], features_2d_r[-points_to_plot:, 1], c='green')  # unknown
        # connect the dots
        for i in range(points_to_plot):
            plt.plot([features_2d_o[i, 0], features_2d_o[points_to_plot + i, 0]],
                     [features_2d_o[i, 1], features_2d_o[points_to_plot + i, 1]], c='black', alpha=0.1)
            plt.plot([features_2d_r[i, 0], features_2d_r[points_to_plot + i, 0]],
                     [features_2d_r[i, 1], features_2d_r[points_to_plot + i, 1]], c='black', alpha=0.1)
        plt.xlabel(" SVD1 ")
        plt.ylabel(" SVD2 ")
        plt.title("Before training")
        plt.savefig(
            saving_dir + "/Before_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".png")
        plt.savefig(
            saving_dir + "/Before_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".pdf")
        plt.close()

    total_loss = []
    for epoch in range(epochs):
        loss_tr_o = 0
        loss_tr_r = 0
        rank_tr_o = 0
        rank_tr_r = 0
        counting_flag_for_rank = 0

        known_features_encoder.train()
        unknown_features_encoder_orig.train()
        unknown_features_encoder_r.train()
        for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
            if len(data[0][1]) == len(data[1][1]):
                # seperate the known and unknown features here  for passing to their corresponding encoders
                x_kn_o = data[0][1][:, :mpfeatures].to(device)
                x_kn_r = data[1][1][:, :mpfeatures].to(device)
                x_unkn_o = data[0][1][:, mpfeatures:].to(device)
                x_unkn_r = data[1][1][:, mpfeatures:].to(device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer_known.zero_grad()
                optimizer_unk_orig.zero_grad()
                optimizer_unk_r.zero_grad()

                datalist = [x_kn_o, x_kn_r, x_unkn_o, x_unkn_r]  # this step could be avoided

                if True: # adding self augmentation
                    for dt in range(4):
                        x_bar = datalist[dt]
                        x_bar_noisy = generate_noisy_xbar(x_bar)
                        # Generate binary mask
                        mask = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                        mask1 = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                        # #breakpoint()
                        # Replace selected x_bar features with the noisy ones
                        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                        x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                        # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                        datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)


                # #breakpoint()

                # computing the encoded values
                known_rep_o, known_dec_o = known_features_encoder(datalist[0])
                known_rep_r, known_dec_r = known_features_encoder(datalist[1])
                unknown_rep_o, unknown_dec_o = unknown_features_encoder_orig(datalist[2])
                unknown_rep_r, unknown_dec_r = unknown_features_encoder_r(datalist[3])

                # breakpoint()
                # embeddings from known features of both databases followed by the unknown features
                contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r = NTXentLoss(known_rep_o, known_rep_r,
                                                                                            unknown_rep_o,
                                                                                            unknown_rep_r, tau)
                contrastive_loss = contrastive_loss_o + contrastive_loss_r

                # Direct reconstruction
                direct_rec_loss = rec_criterion(known_dec_o, datalist[0]) + rec_criterion(known_dec_r, datalist[1]) + rec_criterion(unknown_dec_o, datalist[2]) + rec_criterion(unknown_dec_r, datalist[3])


                # Cross reconstruction (unknown to known, combined rep to known, combined rep to unknown)
                known_cross_dec_o = known_features_encoder(unknown_rep_o,1)
                known_cross_dec_r = known_features_encoder(unknown_rep_r,1)

                supervised_loss_Known = rec_criterion(known_cross_dec_o,datalist[0]) + rec_criterion(known_cross_dec_r,datalist[1])

                known_comb_dec_o = known_features_encoder(0.5*unknown_rep_o + 0.5*known_rep_o,1)
                known_comb_dec_r = known_features_encoder(0.5*unknown_rep_r + 0.5*known_rep_r,1)
                unknown_comb_dec_o = unknown_features_encoder_orig(0.5*unknown_rep_o + 0.5*known_rep_o,1)
                unknown_comb_dec_r = unknown_features_encoder_r(0.5*unknown_rep_r + 0.5*known_rep_r,1)

                Comb_supervised_loss = rec_criterion(known_comb_dec_o, datalist[0] ) + rec_criterion(known_comb_dec_r, datalist[1]) \
                                       + rec_criterion(unknown_comb_dec_o, datalist[2]) + rec_criterion(unknown_comb_dec_r, datalist[3])




                if True: # adding self augmentation
                    self_aug_loss_un = aug_loss(unknown_rep_o) + aug_loss(unknown_rep_r) + aug_loss(
                        known_rep_o) + aug_loss(known_rep_r)
                    contrastive_loss = contrastive_loss + self_aug_loss_un

                # combining the contrastive and decoder losses # TODO: multipliers of the various losses
                contrastive_loss = contrastive_loss + 100 * (weightDirDecoder*direct_rec_loss + weightCrossDecoder*supervised_loss_Known + weightCombDecoder*Comb_supervised_loss)

                # compute accumulated gradients
                contrastive_loss.backward()

                # perform parameter update based on current gradients
                optimizer_known.step()
                optimizer_unk_orig.step()
                optimizer_unk_r.step()

                # add the mini-batch training loss to epoch loss
                loss_tr_o += contrastive_loss_o.item()
                loss_tr_r += contrastive_loss_r.item()
                if (data[0][1].shape[0] == batchSize):
                    rank_tr_o += avg_Rank_o.item()
                    rank_tr_r += avg_Rank_r.item()
                    counting_flag_for_rank = counting_flag_for_rank + 1

        # compute the epoch training loss
        loss_tr_o = loss_tr_o / (len(train_loader_orig))
        loss_tr_r = loss_tr_r / (len(train_loader_r))
        rank_tr_o = rank_tr_o / (
            counting_flag_for_rank)  # dividing by counting_flag_for_rank because the avg rank from all batches in not included
        rank_tr_r = rank_tr_r / (counting_flag_for_rank)

        # #breakpoint()

        with torch.no_grad():
            # computing the representations from the trained encoders
            known_features_encoder.eval()
            unknown_features_encoder_orig.eval()
            unknown_features_encoder_r.eval()

            if True:
                loss_val = 0
                loss_val_o = 0
                loss_val_r = 0
                rank_val_o = 0
                rank_val_r = 0
                counting_flag_for_rank_val = 0
                within_unkn_CL_loss_val = 0

                for i, data in enumerate(zip(val_loader_orig, val_loader_r)):
                    if len(data[0][1]) == len(data[1][1]):
                        # seperate the known and unknown features here  for passing to their corresponding encoders
                        x_kn_o = data[0][1][:, :mpfeatures].to(device)
                        x_kn_r = data[1][1][:, :mpfeatures].to(device)
                        x_unkn_o = data[0][1][:, mpfeatures:].to(device)
                        x_unkn_r = data[1][1][:, mpfeatures:].to(device)

                        datalist = [x_kn_o, x_kn_r, x_unkn_o, x_unkn_r]  # this step could be avoided

                        if True:  # adding self augmentation
                            for dt in range(4):
                                x_bar = datalist[dt]
                                x_bar_noisy = generate_noisy_xbar(x_bar)
                                # Generate binary mask
                                mask = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                                mask1 = torch.tensor(np.random.binomial(1, masking_ratio, x_bar.shape)).to(device)
                                # #breakpoint()
                                # Replace selected x_bar features with the noisy ones
                                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                                x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                                # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                                datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)

                        # computing the encoded values
                        known_rep_o, known_dec_o = known_features_encoder(datalist[0])
                        known_rep_r, known_dec_r = known_features_encoder(datalist[1])
                        unknown_rep_o, unknown_dec_o = unknown_features_encoder_orig(datalist[2])
                        unknown_rep_r, unknown_dec_r = unknown_features_encoder_r(datalist[3])

                        # breakpoint()
                        # embeddings from known features of both databases followed by the unknown features
                        contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r  = NTXentLoss(known_rep_o, known_rep_r, unknown_rep_o, unknown_rep_r, tau)

                        contrastive_loss_val = contrastive_loss_o + contrastive_loss_r

                        # Direct reconstruction
                        direct_rec_loss_val = rec_criterion(known_dec_o, datalist[0]) + rec_criterion(known_dec_r, datalist[
                            1]) + rec_criterion(unknown_dec_o, datalist[2]) + rec_criterion(unknown_dec_r, datalist[3])

                        # breakpoint()
                        # Cross reconstruction (unknown to known, combined rep to known, combined rep to unknown)
                        known_cross_dec_o = known_features_encoder(unknown_rep_o, 1)
                        known_cross_dec_r = known_features_encoder(unknown_rep_r, 1)

                        supervised_loss_Known_val = rec_criterion(known_cross_dec_o, datalist[0]) + rec_criterion(
                            known_cross_dec_r, datalist[1])

                        known_comb_dec_o = known_features_encoder(0.5 * unknown_rep_o + 0.5 * known_rep_o, 1)
                        known_comb_dec_r = known_features_encoder(0.5 * unknown_rep_r + 0.5 * known_rep_r, 1)
                        unknown_comb_dec_o = unknown_features_encoder_orig(0.5 * unknown_rep_o + 0.5 * known_rep_o, 1)
                        unknown_comb_dec_r = unknown_features_encoder_r(0.5 * unknown_rep_r + 0.5 * known_rep_r, 1)

                        Comb_supervised_loss_val = rec_criterion(known_comb_dec_o, datalist[0]) + rec_criterion(
                            known_comb_dec_r, datalist[1]) \
                                               + rec_criterion(unknown_comb_dec_o, datalist[2]) + rec_criterion(
                            unknown_comb_dec_r, datalist[3])

                        if True:  # adding self augmentation
                            self_aug_loss_un = aug_loss(unknown_rep_o) + aug_loss(unknown_rep_r) + aug_loss(
                                known_rep_o) + aug_loss(known_rep_r)
                            contrastive_loss_val = contrastive_loss_val + self_aug_loss_un
                            within_unkn_CL_loss_val += self_aug_loss_un.item()

                        # combining the contrastive and decoder losses # TODO: multipliers of the various losses
                        contrastive_loss_val = contrastive_loss_val + 100 * (
                                    weightDirDecoder * direct_rec_loss_val + weightCrossDecoder * supervised_loss_Known_val + weightCombDecoder * Comb_supervised_loss_val)

                        # add the mini-batch training loss to epoch loss
                        loss_val += contrastive_loss_val.item()

                        # add the mini-batch training loss to epoch loss
                        loss_val_o += contrastive_loss_o.item()
                        loss_val_r += contrastive_loss_r.item()
                        if (data[0][1].shape[0] == batchSize):
                            rank_val_o += avg_Rank_o.item()
                            rank_val_r += avg_Rank_r.item()
                            counting_flag_for_rank_val = counting_flag_for_rank_val + 1

                loss_val = loss_val / (len(val_loader_orig) + len(val_loader_r))
                within_unkn_CL_loss_val = within_unkn_CL_loss_val / (len(val_loader_orig) + len(val_loader_r))

                # compute the epoch training loss
                loss_val_o = loss_val_o / (len(val_loader_orig))
                loss_val_r = loss_val_r / (len(val_loader_r))
                rank_val_o = rank_val_o / (
                    counting_flag_for_rank_val)  # dividing by counting_flag_for_rank because the avg rank from all batches in not included
                rank_val_r = rank_val_r / (counting_flag_for_rank_val)

                # # display the epoch training loss
                # print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, epochs, loss_tr,
                #                                                                                loss_val))

                # display the epoch training loss
                print(
                    "Validation performance epoch : {}/{}, loss_o = {:.5f}, loss_r = {:.5f}, within cont loss = {:.5f},  avgbatchwise_rank_o = {:.5f}, avgbatchwise_rank_r = {:.5f}".format(
                        epoch + 1, epochs, loss_val_o, loss_val_r, within_unkn_CL_loss_val, rank_val_o, rank_val_r))
            else:
                for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
                temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)
                grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
                for i in range(temp_input.shape[0]):
                    grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])

                grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

                for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
                temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device)
                grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
                for i in range(temp_input.shape[0]):
                    grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])

                grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

                o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                                        np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                                        dense_output=True)
                r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                                        np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                                        dense_output=True)

                # Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(
                #     o_to_r_sim,
                #     r_to_o_sim,
                #     index_for_mapping_orig_to_rename[len(mapped_features):],
                #     index_for_mapping_rename_to_orig[len(mapped_features):],
                #     len(mapped_features))

                correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
                    o_to_r_sim,
                    r_to_o_sim,
                    P_x1, len(mapped_features))

                print(" CL  X1_train mistakes number on holdout set",
                      unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
                      unmapped_features_orig - num_xtra_feat_inX1)

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

                F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)

                # display the epoch training loss
                print("Training loss epoch : {}/{}, loss_o = {:.5f}, loss_r = {:,.5f}  F1 on validation = {:.5f}".format(
                        epoch + 1, epochs, loss_tr_o, loss_tr_r, F1_fromx1))

        scheduler_known.step(loss_val)
        scheduler_unk_orig.step(loss_val)
        scheduler_unk_r.step(loss_val)


    if False:
        # Debugging to see where the embedding are positioned in the latent space
        rep_known_val_o = known_features_encoder(torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_o = unknown_features_encoder_orig(
            torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))
        # rep_unknown_val_o_from_r = unknown_features_encoder_r(
        #     torch.Tensor(Df_holdout_orig0.reindex(columns=reordered_column_names_r).iloc[:, mpfeatures:-1].values).to(
        #         device))

        rep_known_val_r = known_features_encoder(torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))
        rep_unknown_val_r = unknown_features_encoder_r(
            torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device))
        # rep_unknown_val_r_from_orig = unknown_features_encoder_orig(
        #     torch.Tensor(DF_holdout_r0.iloc[:, mpfeatures:-1].values).to(device))  # r0 is the one that was not permuted

        # checking the cone effect
        # pair_wise_similarity(rep_known_val_o.cpu().detach().numpy(), 'known_o', 'after')
        # pair_wise_similarity(rep_known_val_r.cpu().detach().numpy(), 'known_r', 'after')
        # pair_wise_similarity(rep_unknown_val_o.cpu().detach().numpy(), 'unknown_o', 'after')
        # pair_wise_similarity(rep_unknown_val_r.cpu().detach().numpy(), 'unknown_r', 'after')

        ## checking for the modality gap of two databases on the same plot
        # #breakpoint()
        points_to_plot = 200  # len(rep_known_val_o.cpu().detach().numpy())
        features_2d_o = svd(np.concatenate([rep_known_val_o.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_o.cpu().detach().numpy()[:points_to_plot]], 0))
        features_2d_r = svd(np.concatenate([rep_known_val_r.cpu().detach().numpy()[:points_to_plot],
                                            rep_unknown_val_r.cpu().detach().numpy()[:points_to_plot]], 0))

        plt.figure(figsize=(5, 5))
        plt.scatter(features_2d_o[:-points_to_plot, 0], features_2d_o[:-points_to_plot, 1], c='red')  # known mapped
        plt.scatter(features_2d_o[-points_to_plot:, 0], features_2d_o[-points_to_plot:, 1], c='blue')  # unknown
        plt.scatter(features_2d_r[:-points_to_plot, 0], features_2d_r[:-points_to_plot, 1], c='magenta')  # known mapped
        plt.scatter(features_2d_r[-points_to_plot:, 0], features_2d_r[-points_to_plot:, 1], c='green')  # unknown
        # connect the dots
        for i in range(points_to_plot):
            plt.plot([features_2d_o[i, 0], features_2d_o[points_to_plot + i, 0]],
                     [features_2d_o[i, 1], features_2d_o[points_to_plot + i, 1]], c='black', alpha=0.1)
            plt.plot([features_2d_r[i, 0], features_2d_r[points_to_plot + i, 0]],
                     [features_2d_r[i, 1], features_2d_r[points_to_plot + i, 1]], c='black', alpha=0.1)
        plt.xlabel(" SVD1 ")
        plt.ylabel(" SVD2 ")
        plt.title("After training")

        plt.savefig(
            saving_dir + "/After_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) +".png")
        plt.savefig(
            saving_dir + "/After_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
                points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + ".pdf")
        plt.close()

        # breakpoint()

        """ plotting the summary importance """

        background_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[:10, :mpfeatures].values),
                                         torch.Tensor(DF_holdout_r.iloc[:10, :mpfeatures].values)], dim=0).to(device)
        shap_to_use_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[10:100, :mpfeatures].values),
                                          torch.Tensor(DF_holdout_r.iloc[10:100, :mpfeatures].values)], dim=0).to(device)
        en_known = shap.DeepExplainer(known_features_encoder, background_known)
        shap_vals_knwn_comb = en_known.shap_values(shap_to_use_known)

        # array_shap_dim = np.zeros((len(shap_vals_knwn_comb), mpfeatures))
        # for i in range(len(shap_vals_knwn_comb)): array_shap_dim[i] = np.mean(
        #     np.absolute(np.transpose(shap_vals_knwn_comb[i])), 1)
        # #breakpoint()
        shap.summary_plot(shap_vals_knwn_comb, feature_names=[itemid_label_dict[int(i)] for i in mapped_features],
                          show=False, max_display=len(mapped_features))
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Known Encoder using both data sources ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) +  "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

        en_unkn_orig = shap.DeepExplainer(unknown_features_encoder_orig,
                                          torch.Tensor(Df_holdout_orig.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_orig = en_unkn_orig.shap_values(
            torch.Tensor(Df_holdout_orig.iloc[10:100, mpfeatures:].values).to(device))

        array_shap_dimUn_orig = np.zeros((len(shap_vals_unkn_orig), len(reordered_column_names_orig[mpfeatures:])))
        for i in range(len(shap_vals_unkn_orig)): array_shap_dimUn_orig[i] = np.mean(np.absolute(np.transpose(shap_vals_unkn_orig[i])), 1)

        shap.summary_plot(shap_vals_unkn_orig,
                          feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_orig[mpfeatures:]], show=False,
                          max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder Original ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

        en_unkn_r = shap.DeepExplainer(unknown_features_encoder_r,
                                       torch.Tensor(DF_holdout_r.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_r = en_unkn_r.shap_values(torch.Tensor(DF_holdout_r.iloc[10:100, mpfeatures:].values).to(device))

        array_shap_dimUn_r = np.zeros((len(shap_vals_unkn_r), len(reordered_column_names_r[mpfeatures:])))
        for i in range(len(shap_vals_unkn_r)): array_shap_dimUn_r[i] = np.mean(np.absolute(np.transpose(shap_vals_unkn_r[i])), 1)

        shap.summary_plot(shap_vals_unkn_r, feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_r[mpfeatures:]],
                          show=False, max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder R ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
                encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(epochs) + "_representation_dim_" +str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" +str(trial) + ".pdf")
        plt.close()

    # #breakpoint()
    # computing the gradient of the output wrt the input data
    # breakpoint()
    #

    for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)
    grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])[0]

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:].values).to(device)
    grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])[0]

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    # np.savetxt(output_dir+"Grad_unknwn_Orig_" + str(randomSeed) + "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_o.cpu().detach().numpy(), delimiter=",")
    # np.savetxt(output_dir+"Grad_unknwn_R_" + str(randomSeed) +  "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_r.cpu().detach().numpy(), delimiter=",")
    # exit()

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)



    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
        o_to_r_sim,
        r_to_o_sim,
        P_x1, len(mapped_features))

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match" , 'Shap_Contr', 'Shap_rank', 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'Shap_Contr', 'Shap_rank', 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    # #breakpoint()
    for i in range(x1_match_matrix_test.shape[0]):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if
                         x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        # temp_inf_x1.loc[i, 'Shap_Contr'] =
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])
    # temp_inf_x1['Shap_Contr'] = np.sum(array_shap_dimUn_orig, 0)
    # temp_inf_x1['Shap_rank'] = temp_inf_x1['Shap_Contr'].rank(ascending=False)

    # #breakpoint()
    for i in range(x2_match_matrix_test.shape[0]):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if
                         x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])
    # temp_inf_x2['Shap_Cont'] = np.sum(array_shap_dimUn_r, 0)

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

    print(" \n Mistakes by the CL method on holdout data")
    print(" CL + Dec   X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" CL + Dec X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)


    print(" -------- CL method training ends ------------- \n \n  ")

    # breakpoint()
    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(list(DF_holdout_r.columns[mpfeatures:]).index(i))
    # breakpoint()

    # test to check if the above lists are correct or  not
    # DF_holdout_r.columns[mpfeatures:][correct_match_idx_r_from_x1]
    # Df_holdout_orig.columns[mpfeatures:][correct_match_idx_orig_from_x1]

    # to compare
    if False:
        # reconstrOrig_from_r_correct_matches = unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[:,correct_match_idx_r_from_x1] and
        Orig_correct_matches = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,correct_match_idx_orig_from_x1]

        # plotting the reconstruction after matching
        row_index_no_orig = np.random.choice(len(Orig_correct_matches), 500, replace=False)
        al = Df_holdout_orig.columns[mpfeatures:]
        for i in range(len(correct_match_idx_orig_from_x1)): # plotting on the correctly mapped features
            x_axis = Df_holdout_orig.iloc[:, mpfeatures:].values[row_index_no_orig,correct_match_idx_orig_from_x1[i]]
            y_axis = unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[row_index_no_orig,correct_match_idx_r_from_x1[i]].cpu().detach().numpy()
            plt.scatter(x_axis, y_axis, color='blue')
            plt.xlabel("true X1 feature value")
            plt.ylabel("cross reconstructed feature value ")
            temp = stats.pearsonr(x_axis, y_axis)[0]
            plt.figtext(0.6, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
            plt.title(" number of mapped feature  " + str(mpfeatures) + " & " + str(itemid_label_dict[int(al[correct_match_idx_orig_from_x1[i]])]) + " correctly mapped ", fontsize=8)
            plt.savefig(saving_dir + "/Cross_recon_qua_"+str(itemid_label_dict[int(al[correct_match_idx_orig_from_x1[i]])]) + "_Cor_Map" + ".png", bbox='tight')
            plt.close()
        # Rec_error_onCorrect = rec_criterion(unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[:,correct_match_idx_r_from_x1],torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,correct_match_idx_orig_from_x1] )
        # Rec_error_onIncorrect = rec_criterion(unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[:,incorrect_match_idx_r_from_x1],torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,incorrect_match_idx_orig_from_x1] )

    # breakpoint()
    predicted_match_dic_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['no_match_or_not'] == 0]['ump_feature_in_X1']),
                                      list(temp_inf_x1[temp_inf_x1['no_match_or_not'] == 0]['match_byGS'])))

    final_dic_for_compar_matching = {}
    for key, val in match_dic.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]


    # Matching metric error
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values, Df_holdout_orig[final_dic_for_compar_matching.values()])

    # oracle combination metric
    overall_quality_oracle_comb = rec_criterion(unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[:,incorrect_match_idx_r_from_x1],torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,incorrect_match_idx_orig_from_x1] )

    # final_dic_for_compar_incorrrect = {}
    # for key, val in match_dic.items():
    #     if val in incorrect_match_dict_x1.values():
    #         final_dic_for_compar_matching[key] = list(incorrect_match_dict_x1.keys())[list(incorrect_match_dict_x1.values()).index(val)]

    # breakpoint()

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


    # encoding to be used later
    known_features_encoder.eval()
    # training dataset
    rep_known_val_o = known_features_encoder(
        torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))[0].cpu().detach().numpy()
    rep_known_val_r = known_features_encoder(
        torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))[0].cpu().detach().numpy()

    del df_rename_preproc
    print("F1 + Dec from CL ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only,  'Oracle combo metric ', overall_quality_oracle_comb)
    # breakpoint()
    # exit()
    return grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test, correct_with_match_from_x2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r, overall_quality_error_matching_only, overall_quality_oracle_comb.item()


def Train_KMF(df_train_preproc, df_rename_preproc,
                   P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures
    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_unmap_mapped = np.zeros((unmapped_features_orig, mpfeatures))
    CorMatrix_X2_unmap_mapped = np.zeros((unmapped_features_r, mpfeatures))
    CorMatrix_X1_unmap_mapped_P_value = np.zeros((unmapped_features_orig, mpfeatures))
    CorMatrix_X2_unmap_mapped_P_value = np.zeros((unmapped_features_r, mpfeatures))

    for i in range(unmapped_features_orig):
        for j in range(mpfeatures):
            temp = stats.pearsonr(Df_holdout_orig.values[:, mpfeatures + i], Df_holdout_orig.values[:, j])
            CorMatrix_X1_unmap_mapped[i, j] = temp[0]
            CorMatrix_X1_unmap_mapped_P_value[i, j] = temp[1]

    for i in range(unmapped_features_r):
        for j in range(mpfeatures):
            temp = stats.pearsonr(DF_holdout_r.values[:, mpfeatures + i], DF_holdout_r.values[:, j])
            CorMatrix_X2_unmap_mapped[i, j] = temp[0]
            CorMatrix_X2_unmap_mapped_P_value[i, j] = temp[1]

    # similarity between the correlation matrices
    sim_cor_norm_X1_to_X2 = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,
                                                       dense_output=True)
    sim_cor_norm_X2_to_X1 = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped, CorMatrix_X1_unmap_mapped,
                                                       dense_output=True)

    """ Calling the stable marriage algorithm for mappings  """
    # #breakpoint()
    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test_KMF, x2_match_matrix_test_KMF = Matching_via_HRM(
        sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,P_x1,
        len(mapped_features))


    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])


    temp_inf_x1.estimated_cross_corr = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test_KMF.shape[0]) for
                                        j in range(x1_match_matrix_test_KMF.shape[1]) if x1_match_matrix_test_KMF[i, j] == 1]
    temp_inf_x2.estimated_cross_corr = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test_KMF.shape[0]) for
                                        j in range(x2_match_matrix_test_KMF.shape[1]) if x2_match_matrix_test_KMF[
                                            i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small


    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test_KMF.shape[1]) if x1_match_matrix_test_KMF[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test_KMF.shape[1]) if x2_match_matrix_test_KMF[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])


    print(" \n Mistakes by the simple correlation method on holdout data")
    print(" Sim_Correlation  X1_train mistakes number", unmapped_features_orig-correct_with_match_from_x1_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number", unmapped_features_orig-correct_with_match_from_x2_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)


    print(" -------- KMF-l methods  ends ------------- \n \n  ")

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])

    del df_rename_preproc

    # Computation of F1 scores

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test_KMF[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test_KMF[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_KMF[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_KMF[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test_KMF[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test_KMF[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_KMF[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_KMF[i, j] == 1):
                FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    print("F1 from KMF ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only)

    return CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped, correct_with_match_from_x1_test, correct_with_match_from_x2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, overall_quality_error_matching_only

def CL_with_KMF_linear(grad_sum_unkn_o, grad_sum_unkn_r, CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')

    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures

    if block_stand_comb == 1:
        grad_sum_unkn_o = (grad_sum_unkn_o / np.std(grad_sum_unkn_o.cpu().detach().numpy()))
        grad_sum_unkn_r = (grad_sum_unkn_r / np.std(grad_sum_unkn_r.cpu().detach().numpy()))
        CorMatrix_X1_unmap_mapped = (CorMatrix_X1_unmap_mapped / np.std(CorMatrix_X1_unmap_mapped))
        CorMatrix_X2_unmap_mapped = (CorMatrix_X2_unmap_mapped / np.std(CorMatrix_X2_unmap_mapped))

    # concat the grad and cor together
    grad_sum_unkn_o_concat = np.concatenate(
        [np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), CorMatrix_X1_unmap_mapped], axis=1)
    grad_sum_unkn_r_concat = np.concatenate(
        [np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), CorMatrix_X2_unmap_mapped], axis=1)

    o_to_r_sim_CL_KMFl = pairwise.cosine_similarity(grad_sum_unkn_o_concat, grad_sum_unkn_r_concat, dense_output=True)
    r_to_o_sim_CL_KMFl = pairwise.cosine_similarity(grad_sum_unkn_r_concat, grad_sum_unkn_o_concat, dense_output=True)



    correct_with_match_from_x1_te_CL_KMFl, correct_with_match_from_x2_te_CL_KMFl, x1_match_matrix_test_CL_KMFl, x2_match_matrix_test_CL_KMFl = Matching_via_HRM(
        o_to_r_sim_CL_KMFl, r_to_o_sim_CL_KMFl,P_x1,
        len(mapped_features))

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match" , 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match",'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])


    # temp_inf_x1.estimated_cross_corr = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test_KMF.shape[0]) for
    #                                     j in range(x1_match_matrix_test_KMF.shape[1]) if x1_match_matrix_test_KMF[i, j] == 1]
    # temp_inf_x2.estimated_cross_corr = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test_KMF.shape[0]) for
    #                                     j in range(x2_match_matrix_test_KMF.shape[1]) if x2_match_matrix_test_KMF[
    #                                         i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small


    for i in range(x1_match_matrix_test_CL_KMFl.shape[0]):
        matched_index = [j for j in range(x1_match_matrix_test_CL_KMFl.shape[1]) if x1_match_matrix_test_CL_KMFl[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    for i in range(x2_match_matrix_test_CL_KMFl.shape[0]):
        matched_index = [j for j in range(x2_match_matrix_test_CL_KMFl.shape[1]) if x2_match_matrix_test_CL_KMFl[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])


    print(" \n Mistakes by the CL KMFl method on holdout data")
    print(" CL KMFl X1_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x1_te_CL_KMFl-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" CL KMFl X2_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x2_te_CL_KMFl-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(reordered_column_names_orig[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(reordered_column_names_r[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(reordered_column_names_orig[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(reordered_column_names_r[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])



    print(" -------- CL + KMFl method training ends ------------- \n \n  ")

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test_CL_KMFl[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test_CL_KMFl[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_CL_KMFl[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_CL_KMFl[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test_CL_KMFl[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test_CL_KMFl[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_CL_KMFl[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_CL_KMFl[i, j] == 1):
                FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    print("CL + KMF linear F1 score ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only)

    return  correct_with_match_from_x1_te_CL_KMFl, correct_with_match_from_x2_te_CL_KMFl, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, overall_quality_error_matching_only


def CL_with_KMF_CLencoded(grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o, rep_known_val_r, Df_holdout_orig, DF_holdout_r,
                       P_x1
                       , reordered_column_names_orig, reordered_column_names_r,
                       mapped_features,Cor_from_df):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')

    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures

    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_unmap_mappedE = np.zeros((unmapped_features_orig, encKnwMapWidthFinal))
    CorMatrix_X2_unmap_mappedE = np.zeros((unmapped_features_r, encKnwMapWidthFinal))

    for i in range(unmapped_features_orig):
        for j in range(encKnwMapWidthFinal):
            temp = stats.pearsonr(Df_holdout_orig.values[:, mpfeatures + i], rep_known_val_o[:, j])
            CorMatrix_X1_unmap_mappedE[i, j] = temp[0]

    for i in range(unmapped_features_r):
        for j in range(encKnwMapWidthFinal):
            temp = stats.pearsonr(DF_holdout_r.values[:, mpfeatures + i], rep_known_val_r[:, j])
            CorMatrix_X2_unmap_mappedE[i, j] = temp[0]

    if block_stand_comb == 1:
        grad_sum_unkn_o = (grad_sum_unkn_o / np.std(grad_sum_unkn_o.cpu().detach().numpy()))
        grad_sum_unkn_r = (grad_sum_unkn_r / np.std(grad_sum_unkn_r.cpu().detach().numpy()))
        CorMatrix_X1_unmap_mappedE = (CorMatrix_X1_unmap_mappedE / np.std(CorMatrix_X1_unmap_mappedE))
        CorMatrix_X2_unmap_mappedE = (CorMatrix_X2_unmap_mappedE / np.std(CorMatrix_X2_unmap_mappedE))

    # concat the grad and cor (unknown, embeddings) together
    grad_sum_unkn_o_concat_en = np.concatenate(
        [np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), CorMatrix_X1_unmap_mappedE], axis=1)
    grad_sum_unkn_r_concat_en = np.concatenate(
        [np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), CorMatrix_X2_unmap_mappedE], axis=1)

    o_to_r_sim_CL_KMFen = pairwise.cosine_similarity(grad_sum_unkn_o_concat_en, grad_sum_unkn_r_concat_en,
                                                     dense_output=True)
    r_to_o_sim_CL_KMFen = pairwise.cosine_similarity(grad_sum_unkn_r_concat_en, grad_sum_unkn_o_concat_en,
                                                     dense_output=True)


    correct_with_match_from_x1_te_CL_KMFen, correct_with_match_from_x2_te_CL_KMFen, x1_match_matrix_test_CL_KMFen, x2_match_matrix_test_CL_KMFen = Matching_via_HRM(
        o_to_r_sim_CL_KMFen, r_to_o_sim_CL_KMFen,P_x1,
        len(mapped_features))

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])



    for i in range(x1_match_matrix_test_CL_KMFen.shape[0]):
        matched_index = [j for j in range(x1_match_matrix_test_CL_KMFen.shape[1]) if x1_match_matrix_test_CL_KMFen[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    for i in range(x2_match_matrix_test_CL_KMFen.shape[0]):
        matched_index = [j for j in range(x2_match_matrix_test_CL_KMFen.shape[1]) if x2_match_matrix_test_CL_KMFen[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    print(" \n Mistakes by the CL KMF-encoded method on holdout data")
    print(" CL KMFl X1_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x1_te_CL_KMFen-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" CL KMFl X2_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x2_te_CL_KMFen-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])


    print(" -------- CL + KMF-encoded method training ends ------------- \n \n  ")

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test_CL_KMFen[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test_CL_KMFen[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_CL_KMFen[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test_CL_KMFen[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test_CL_KMFen[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test_CL_KMFen[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_CL_KMFen[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test_CL_KMFen[i, j] == 1):
                FP_x2 = FP_x2 + 1

    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)
    print("CL + KMF encoded F1 score ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only)

    return correct_with_match_from_x1_te_CL_KMFen, correct_with_match_from_x2_te_CL_KMFen, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, overall_quality_error_matching_only

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

    num_iter = 2000
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


    # true_permutation = list(np.arange(mpfeatures)) +  [np.where(P_x1[a,:]==1)[0] + mpfeatures for a in range(len(P_x1))]
    true_permutation = [np.where(P_x1[a,:]==1)[0] for a in range(len(P_x1))]
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_permutation[i]==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1

    print(" \n Mistakes by the KANG method on training data")

    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    x1_match_matrix_test = np.zeros(P_x1.shape)
    for i in range(x1_match_matrix_test.shape[0]):
        # print(i, x1_train_y[i]-1)
        x1_match_matrix_test[i, initial_perm[i]] = 1
    x1_match_matrix_test_updated = x1_match_matrix_test[mpfeatures:,mpfeatures:] # update to make sure that the mapped features are not  included when computing the mistakes

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", "Correct_Match", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    # breakpoint()

    for i in range(x1_match_matrix_test_updated.shape[0]):
        matched_index = [j for j in range(x1_match_matrix_test_updated.shape[1]) if x1_match_matrix_test_updated[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    # correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
    #                                  list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    # incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
    #                                    list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])

    print(" -------- KANG  methods  ends ------------- \n \n  ")
    # breakpoint()

    del DF_holdout_r

    # this update is being done to make sure that the corrects matches of the mapped variables do not count towards the F! score
    P_x1_updated = P_x1[mpfeatures:,mpfeatures:]
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


    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    # F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)

    print("Kang F1 score ", F1_fromx1)
    print('Matching metric ', overall_quality_error_matching_only)
    # breakpoint()

    return correct_total_fromKANG, F1_fromx1, overall_quality_error_matching_only

def Train_cross_AE(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r,
                   mapped_features, Cor_from_df,
                   Df_holdout_orig, DF_holdout_r):
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
            model_orig = AE_2_hidden_layer(input_shape=num_NonCat_features_orig, batchnorm=batchnorm, drop_out_p=dropout_rate,hid_dim = hidden_dim).to(
                        device)
            model_r = AE_2_hidden_layer(input_shape=num_NonCat_features_r, batchnorm=batchnorm, drop_out_p=dropout_rate, hid_dim = hidden_dim).to(device)
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
    # optimizer_orig = optim.Adam(model_orig.parameters(), lr=learning_rate)
    # optimizer_r = optim.Adam(model_r.parameters(), lr=learning_rate)

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
                train_loss_cross_r = criterion(output_cross_r[:, :len(mapped_features)],
                                               x_r[:, :len(mapped_features)])
                train_loss_cross_orig = criterion(output_cross_orig[:, :len(mapped_features)],
                                                  x_o[:, :len(mapped_features)])

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
                    orthog_loss_orig = torch.zeros(1).to(device)
                    orthog_loss_r = torch.zeros(1).to(device)

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

    # plotting loss versus epochs
    x_time = np.arange(epochs)
    plt.plot(x_time, np.array(total_loss), color='black', label='total recon loss')
    plt.plot(x_time, np.array(ae_orig_error_list), color='blue', label='ae_orig_error')
    plt.plot(x_time, np.array(ae_r_error_list), color='green', label='ae_r_error')
    plt.plot(x_time, np.array(ae_orig_on_r_cross_list), color='yellow', label='ae_orig_on_r_cross')
    plt.plot(x_time, np.array(ae_r_on_orig_cross_list), color='red', label='ae_r_on_orig_cross')
    plt.xlabel("epoch no")
    plt.ylabel("reconstruction error")
    plt.title(' ae error from different parts', fontsize=8)
    plt.legend()
    plt.savefig('MIMIC_cont_ae_error_Orthtype' +str(orthogonalization_type)+ "_#layers_" +str(num_of_hidden_layers) + "_hid_" +str(hidden_dim)+'.pdf', bbox='tight')
    plt.close()

    """ AE part preprocessing  ends   """

    # switching to eval mode so that drop out is off when evaluating
    model_orig.eval()
    model_r.eval()

    """ JACOBIAN CALCULATION  """
    # #breakpoint()
    for param in model_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, :].values).to(device)
    grad_sum_unkn_o = torch.zeros((hidden_dim, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_o += torch.autograd.functional.jacobian(model_orig, temp_input[i])

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in model_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, :].values).to(device)
    grad_sum_unkn_r = torch.zeros((hidden_dim, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_r += torch.autograd.functional.jacobian(model_r, temp_input[i])

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)


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
        columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)" , 'true_correlation', 'estimated_cross_corr', 'corr_p_value',
                 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', 'CV_label','match_byGS', "match_byGS(MV_label)", 'true_correlation', 'estimated_cross_corr', 'corr_p_value',
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


    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
            temp_inf_x1.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0
            temp_inf_x1.loc[i, "Correct_Match"] = sum(temp_inf_x1.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x1.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
            int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[
            reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[
                len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
            temp_inf_x2.loc[i, "Correct_Match"] = "NA"
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0
            temp_inf_x2.loc[i, "Correct_Match"] = sum(temp_inf_x2.loc[i, 'match_byGS'] ==
                                              match_df[
                                                  match_df['CV_itemids'] == temp_inf_x2.loc[i, 'ump_feature_in_X1']][
                                                  'MV_itemids'])

    # correct_with_no_match_from_CCx1_test = 0
    # correct_with_no_match_from_CCx2_test = 0
    # for i in range(len(temp_inf_x1.SD_rejects_H0)):
    #     if temp_inf_x1.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
    #         correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
    #     if temp_inf_x2.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
    #         correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the 2stage Chimeric method on holdout data")
    print(" Chimeric  X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" Chimeric  X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)

    # print("Mistakes by the significance testing algorithm on holdout data (2stage chimeric)")
    # print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    # print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)

    # print(" DF for post-hoc analysis from x1")
    # print(temp_inf_x1)
    # print(" DF for post-hoc analysis from x2")
    # print(temp_inf_x2)

    # evaluating the reconstruction quality (on correct matches) and computing the recontruction error (on incorrect matches)
    correct_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 1]['match_byGS'])))
    correct_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['ump_feature_in_X1']),
                                     list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 1]['match_byGS'])))
    incorrect_match_dict_x1 = dict(zip(list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x1[temp_inf_x1['Correct_Match'] == 0]['match_byGS'])))
    incorrect_match_dict_x2 = dict(zip(list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['ump_feature_in_X1']),
                                       list(temp_inf_x2[temp_inf_x2['Correct_Match'] == 0]['match_byGS'])))

    # obtaining the index to compute the loss on incorrect or correct matches
    correct_match_idx_orig_from_x1 = []
    for i in list(correct_match_dict_x1.keys()): correct_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    correct_match_idx_r_from_x1 = []
    for i in list(correct_match_dict_x1.values()): correct_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
        list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
        list(DF_holdout_r.columns[mpfeatures:]).index(i))

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
        Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
        Df_holdout_orig[final_dic_for_compar_matching.values()])

    # oracle combination metric
    overall_quality_oracle_comb = criterion(recons_orig_Test_frommodelR[:,
                                                incorrect_match_idx_r_from_x1],
                                                torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,
                                                incorrect_match_idx_orig_from_x1])

    print(" -------- Chimeric AE method training ends ------------- \n \n  ")

    # print("Mapped feature_set after KMF ")
    # print(" X1 ", mapped_features_updated_orig)
    # print(" X2 ", mapped_features_updated_r)
    # exit()
    # plot for partial mapping evaluation
    # features_true_orig_test_not_inorig = DF_holdout_orig0_not_includedwhiletraining.values
    # row_index_no_orig = np.random.choice(len(features_true_orig_test_not_inorig), 500, replace=False)
    # for feature_name in unmapped_features_extra_orig:
    #     print('\n ---------- feature name --------- \n ', feature_name)
    #     col_idxno_orig = list(DF_holdout_orig0_not_includedwhiletraining.columns).index(feature_name)
    #     col_idxno_renamed = reordered_column_names_r.index(feature_name)
    #     x_axis = features_true_orig_test_not_inorig[row_index_no_orig, col_idxno_orig]
    #     y_axis = features_reconst_from_crossR_test[row_index_no_orig, col_idxno_renamed]
    #
    #     plt.scatter(x_axis, y_axis, color='blue')
    #     plt.xlabel("true feature value not present in x1 while training")
    #     plt.ylabel("reconstructed feature value from x2")
    #     temp = stats.pearsonr(x_axis, y_axis)[0]
    #     plt.figtext(0.5, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
    #     plt.figtext(0.5, 0.75, "True frac value 1 = " + str(np.round(np.average(x_axis), decimals=3)))
    #     plt.title(" number of mapped feature  " + str(
    #         mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     if feature_name in mapped_features_updated_orig:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".pdf",
    #             bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".png",
    #             bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    #
    #     Temp_df = pd.DataFrame(columns=['True values', 'Reconstructed values'])
    #     Temp_df['True values'] = x_axis
    #     Temp_df['Reconstructed values'] = y_axis
    #     from sklearn.calibration import calibration_curve
    #     from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
    #     precision, recall, _ = precision_recall_curve(x_axis, y_axis)
    #     disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    #     disp.plot()
    #     auprc = average_precision_score(x_axis, y_axis)
    #     plt.figtext(0.6, 0.8, "AUPRC value = " + str(np.round(auprc, decimals=3)))
    #     plt.title(" number of mapped feature  " + str(
    #         mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     if feature_name in mapped_features_updated_orig:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_PR_curve_GotMappedafter_KMF" + "_NotInOrig_" + str(
    #                 feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_PR_curve_GotMappedafter_KMF" + "_NotInOrig_" + str(
    #                 feature_name) + ".png", bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_PR_curve_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_PR_curve_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    #
    #     # # CDF
    #     plt.plot(np.sort(Temp_df[Temp_df['True values'] == 0]['Reconstructed values']),
    #              np.linspace(0, 1, len(Temp_df[Temp_df['True values'] == 0]['Reconstructed values']), endpoint=False),
    #              color='blue', label='x = 0')
    #     plt.plot(np.sort(Temp_df[Temp_df['True values'] == 1]['Reconstructed values']),
    #              np.linspace(0, 1, len(Temp_df[Temp_df['True values'] == 1]['Reconstructed values']), endpoint=False),
    #              color='green', label='x = 1')
    #     # plt.plot(np.sort(y_axis), np.linspace(0, 1, len(y_axis), endpoint=False), color = 'green', label = 'Recons feature')
    #     # plt.xlim([0,  1])
    #     # plt.ylim([0,  1])
    #     plt.xlabel("feature values")
    #     plt.ylabel("Empirical CDF ")
    #     plt.legend()
    #     plt.title(" CDF when number of mapped feature  " + str(
    #         mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     if feature_name in mapped_features_updated_orig:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_CDF_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".pdf",
    #             bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_CDF_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".png",
    #             bbox='tight')
    #     else:
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_CDF_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #         plt.savefig(
    #             filename_for_saving_PM_quality + "_CDF_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     plt.close()
    #     #
    #     # # Maatplotlib violin plot
    #     # plt.violinplot(dataset=[x_axis, y_axis])
    #     # plt.xlabel("(true, reconstructed)")
    #     # plt.ylabel("Distribution ")
    #     # plt.title(" Violinplot when number of mapped feature  " + str(
    #     #     mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     # if feature_name in mapped_features_updated_orig:
    #     #     plt.savefig(
    #     #         filename_for_saving_PM_quality + "_ViolinPlt_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #     #     plt.savefig(
    #     #         filename_for_saving_PM_quality + "_ViolinPlt_GotMappedafter_KMF" + "_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     # else:
    #     #     plt.savefig(
    #     #         filename_for_saving_PM_quality + "_ViolinPlt_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #     #     plt.savefig(
    #     #         filename_for_saving_PM_quality + "_ViolinPlt_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     # plt.close()
    #     #
    #     # # SNS violin plot
    #     # import seaborn as sns
    #     # # sns.violinplot(data=[x_axis, y_axis])
    #     # # sns.violinplot(data=[x_axis, y_axis])
    #     # sns.violinplot(x=".", y="Reconstructed values", hue="True values", data=Temp_df, palette="muted", split=True)
    #     # plt.title(" Violinplot when number of mapped feature  " + str(
    #     #     mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     # plt.savefig(
    #     #     filename_for_saving_PM_quality + "_SNSViolinPlt_NotInOrig_" + str(feature_name) + ".pdf", bbox='tight')
    #     # plt.savefig(
    #     #     filename_for_saving_PM_quality + "_SNSViolinPlt_NotInOrig_" + str(feature_name) + ".png", bbox='tight')
    #     # plt.close()
    #     # # Scaled SNS violin plot
    #     # sns.violinplot(x=".", y="Reconstructed values", hue="True values", data=Temp_df, palette="muted", split=True,
    #     #                scale='count')
    #     # plt.title("Scaled width (count based) Violinplot when number of mapped feature  " + str(
    #     #     mpfeatures) + " & " + str(feature_name) + " not in original data", fontsize=8)
    #     # plt.savefig(
    #     #     filename_for_saving_PM_quality + "_Scaled_SNSViolinPlt_NotInOrig_" + str(feature_name) + ".pdf",
    #     #     bbox='tight')
    #     # plt.savefig(
    #     #     filename_for_saving_PM_quality + "_Scaled_SNSViolinPlt_NotInOrig_" + str(feature_name) + ".png",
    #     #     bbox='tight')
    #     # plt.close()
    #
    # for feature_name in reordered_column_names_orig[:-1]:
    #     col_idxno_orig = list(Df_holdout_orig.columns).index(feature_name)
    #     x_axis = features_true_orig_test[row_index_no_orig,col_idxno_orig]
    #     y_axis = recons_orig_Test_from_orig.cpu().detach().numpy()[row_index_no_orig,col_idxno_orig]
    #     plt.scatter(x_axis, y_axis, color='blue')
    #     plt.xlabel("true feature value (x1) ")
    #     plt.ylabel("direct reconstructed feature value (x1)")
    #     temp = stats.pearsonr(x_axis,y_axis)[0]
    #     plt.figtext(0.5, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
    #     plt.figtext(0.5, 0.75, "True frac value 1 = " + str(np.round(np.average(x_axis), decimals=3)))
    #     # plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     # plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
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
    #     plt.figtext(0.5, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
    #     plt.figtext(0.5, 0.75, "True frac value 1 = " + str(np.round(np.average(x_axis), decimals=3)))
    #     # plt.xlim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
    #     # plt.ylim([min(min(y_axis), min(x_axis)) - 1, max(max(y_axis), max(x_axis)) + 1])
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

    print( "Chimeric F values ", F1_fromx1, F1_fromx2)
    print('Matching metric ', overall_quality_error_matching_only,  'Oracle combo metric ', overall_quality_oracle_comb)

    del df_rename_preproc

    return o_to_r_sim, r_to_o_sim, correct_with_match_from_x1_test, correct_with_match_from_x2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, overall_quality_error_matching_only, overall_quality_oracle_comb.item()


def main(dataset_no_sample):
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_num_threads(6)
    print("Number of threads being used are ", torch.get_num_threads())
    random.seed(100)
    np.random.seed(100)  # fixing the seed for reproducibility


    # output arrays

    AVG_MISMATCHES_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_MISMATCHES_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))


    AVG_MISMATCHES_X1_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_MISMATCHES_X2_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    F1_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    AVG_F1_X1_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t))
    AVG_F1_X2_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t))

    F1_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    # defining the match mse across trials; storing only for X1 side as that is what we will be using ultimately
    MatchMSE_across_trial_perm_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_KMF_en = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))

    ReconMSE_across_trial_perm_X1_tr = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))
    ReconMSE_across_trial_perm_X1_tr_CL_Dec = np.zeros((len(list_of_number_mapped_variables), n_t * n_p))


    no_match_inference_df_from_x1 = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)", "Correct_Match", 'Shap_Contr', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2 = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)",  "Correct_Match",'Shap_Contr','true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1','CV_label', 'match_byGS',"match_byGS(MV_label)" , "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)" , "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_CL = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)", "Correct_Match", 'Shap_Contr', 'Shap_rank','true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_CL = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)",  "Correct_Match",'Shap_Contr', 'Shap_rank','true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_CL_with_Dec = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)", "Correct_Match", 'Shap_Contr', 'Shap_rank','true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_CL_with_Dec = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)",  "Correct_Match",'Shap_Contr', 'Shap_rank','true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_CL_KMFl = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)", "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_CL_KMFl = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)",  "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    no_match_inference_df_from_x1_CL_KMFen = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)", "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    no_match_inference_df_from_x2_CL_KMFen = pd.DataFrame(columns=['ump_feature_in_X1', 'CV_label', 'match_byGS',"match_byGS(MV_label)",  "Correct_Match",'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])



    m = 0  # variables to keep track of the iterations over number of mapped features
    for mpfeatures in list_of_number_mapped_variables:
        run_num = 0  # variable to keep track of the run number out of n_t*n_p
        print("\n ********************************************************")
        print("Run when there are ", mpfeatures, " mapped features starts")
        print(" ******************************************************** \n")

        for trial in range(n_t):

            mapped_features = list(np.random.choice(known_columnslist, mpfeatures, replace=False))
            remaining_lab_ids = [i for i in known_columnslist if i not in mapped_features]

            # array for saving the frac of mistakes
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang = np.zeros(n_p)

            # array for saving F1 scores
            F1_for_fixed_trial_fixed_num_mapped_X1_tr = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang = np.zeros(n_p)


            # the copies are being made because of the multiple trials
            df_trainCV = df_train_CV_proc.copy()
            df_trainMV = df_train_MV_proc.copy()
            Df_holdout_CV0 = df_holdout_CV_proc.copy()
            Df_holdout_MV0 = df_holdout_MV_proc.copy()

            # maximum possible mistakes for this trial
            # max_mistakes = len(df_trainCV.columns) - len(mapped_features)
            max_mistakes = len(df_trainCV.columns) - len(known_columnslist)  # not moving the premappped labs to the chartevents set, does include the ones that do not have any matches but F1 score doesn't include them so its fine
            for partition in range(n_p):
                df_train_preproc = df_trainCV.copy()
                df_rename_preproc = df_trainMV.copy()  # a copy to keep the original ordering as a baseline when matching
                DF_holdout_orig = Df_holdout_CV0.copy()
                Df_holdout_r = Df_holdout_MV0.copy()

                print("\n ********************************************************")
                print(" Partition number ", partition + 1, "   starts for trail number ", trial + 1,
                      " when there are ",
                      mpfeatures, " mapped features")
                print(" ******************************************************** \n")
                # unmapped_CV = remaining_lab_ids + [i for i in df_train_preproc.columns if i not in known_columnslist]
                # unmapped_MV = remaining_lab_ids + [i for i in df_rename_preproc.columns if i not in known_columnslist]

                unmapped_CV = [i for i in df_train_preproc.columns if i not in known_columnslist] # not moving the premappped labs to the chartevents set
                unmapped_MV = [i for i in df_rename_preproc.columns if i not in known_columnslist] # not moving the premappped labs to the chartevents set

                # reshuffling the columns that have not been mapped
                random.shuffle(unmapped_CV)
                random.shuffle(unmapped_MV)

                # reordering the columns in the two datasets for a new permutation
                reorder_column_CV = mapped_features + unmapped_CV
                reorder_column_MV = mapped_features + unmapped_MV

                df_train_preproc = df_train_preproc.reindex(columns = reorder_column_CV)
                DF_holdout_orig = DF_holdout_orig.reindex(columns = reorder_column_CV)

                df_rename_preproc = df_rename_preproc.reindex(columns = reorder_column_MV)
                Df_holdout_r = Df_holdout_r.reindex(columns = reorder_column_MV)
                # #breakpoint()

                """ # true permutation matrix  """

                P_x1 = np.zeros((len(df_train_preproc.columns), len(df_rename_preproc.columns)))

                print("Shape of P_x1 ", P_x1.shape)
                # breakpoint()
                # exit()
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

                # calling the Kang function
                correct_with_match_from_x1_test_Kang, F1_fromx1_Kang, match_mse_kang = Kang_MI_HC_opt(
                                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1, reorder_column_CV, reorder_column_MV, mapped_features, Cor_df, DF_holdout_orig, Df_holdout_r)


                # calling the KMF function
                Cor_X1_map_unmap, Cor_X2_map_unmap,correct_with_match_from_x1_test_sim_cor, correct_with_match_from_x2_test_sim_cor, temp_infer_from_x1_sim_cor, temp_infer_from_x2_sim_cor, F1_fromx1_simcor, F1_fromx2_simcor, match_mse_kmf = Train_KMF(df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):,len(mapped_features):],
                                    reorder_column_CV, reorder_column_MV, mapped_features, Cor_df, DF_holdout_orig,
                                    Df_holdout_r)


                # calling the Chimeric function
                o_to_r_sim_chimeric, r_to_o_sim_chimeric,correct_with_match_from_x1_test, correct_with_match_from_x2_test, temp_infer_from_x1, temp_infer_from_x2, F1_fromx1, F1_fromx2, match_mse_chimeric, recon_mse_chimeric = \
                    Train_cross_AE(df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):,
                                                                                     len(mapped_features):], reorder_column_CV, reorder_column_MV, mapped_features, Cor_df,
                                   DF_holdout_orig, Df_holdout_r)


                # calling the CL function
                _, _, correct_with_match_from_x1_test_CL, correct_with_match_from_x2_test_CL, temp_infer_from_x1_CL, temp_infer_from_x2_CL, F1_x1_CL, F1_x2_CL, rep_known_val_o, rep_known_val_r, match_mse_cl = Train_CL(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):, len(mapped_features):], reorder_column_CV,  reorder_column_MV, mapped_features, Cor_df, DF_holdout_orig, Df_holdout_r, partition, trial)

                # calling the CL+dec function
                grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test_CL_withdec, correct_with_match_from_x2_test_CL_withdec, temp_infer_from_x1_CL_withdec, temp_infer_from_x2_CL_withdec, F1_x1_CL_withdec, F1_x2_CL_withdec, rep_known_val_o_withdec, rep_known_val_r_withdec, match_mse_cl_dec, recon_mse_cl_dec = Train_CL_withDec(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):, len(mapped_features):], reorder_column_CV,  reorder_column_MV, mapped_features, Cor_df, DF_holdout_orig, Df_holdout_r, partition, trial)

                # calling the CL + KMF-l function
                correct_with_match_from_x1_test_CL_KMFl, correct_with_match_from_x2_test_CL_KMFl, temp_infer_from_x1_CL_KMFl, temp_infer_from_x2_CL_KMFl, F1_x1_CL_CL_KMFl, F1_x2_CL_CL_KMFl, match_mse_cl_kmfl = CL_with_KMF_linear(grad_sum_unkn_o, grad_sum_unkn_r, Cor_X1_map_unmap,
                                   Cor_X2_map_unmap, P_x1[len(mapped_features):, len(mapped_features):], reorder_column_CV,  reorder_column_MV,
                                   mapped_features, Cor_df,DF_holdout_orig)

                # calling the CL + KMF-en function
                correct_with_match_from_x1_test_CL_KMFen, correct_with_match_from_x2_test_CL_KMFen, temp_infer_from_x1_CL_KMFen, temp_infer_from_x2_CL_KMFen, F1_x1_CL_CL_KMFen, F1_x2_CL_CL_KMFen, match_mse_cl_kmfen = CL_with_KMF_CLencoded(grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o_withdec, rep_known_val_r_withdec,
                                      DF_holdout_orig, Df_holdout_r,P_x1[len(mapped_features):, len(mapped_features):], reorder_column_CV,  reorder_column_MV,
                                      mapped_features, Cor_df)

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = 1-(correct_with_match_from_x1_test)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = 1-(correct_with_match_from_x2_test)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[partition]
                Frac_mismatches_across_trial_perm_X2_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[partition]

                no_match_inference_df_from_x1 = pd.concat([no_match_inference_df_from_x1, temp_infer_from_x1])
                no_match_inference_df_from_x2 = pd.concat([no_match_inference_df_from_x2, temp_infer_from_x2])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = 1-(correct_with_match_from_x1_test_sim_cor)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = 1-(correct_with_match_from_x2_test_sim_cor)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_sim_Cor[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition]
                Frac_mismatches_across_trial_perm_X2_tr_sim_Cor[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition]

                no_match_inference_df_from_x1_Sim_cor = pd.concat(
                    [no_match_inference_df_from_x1_Sim_cor, temp_infer_from_x1_sim_cor])
                no_match_inference_df_from_x2_Sim_cor = pd.concat(
                    [no_match_inference_df_from_x2_Sim_cor, temp_infer_from_x2_sim_cor])


                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = 1-(correct_with_match_from_x1_test_CL)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = 1-(correct_with_match_from_x2_test_CL)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition]


                no_match_inference_df_from_x1_CL = pd.concat([no_match_inference_df_from_x1_CL, temp_infer_from_x1_CL])
                no_match_inference_df_from_x2_CL = pd.concat([no_match_inference_df_from_x2_CL, temp_infer_from_x2_CL])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = 1-(correct_with_match_from_x1_test_CL_withdec)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = 1-(correct_with_match_from_x2_test_CL_withdec)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition]

                no_match_inference_df_from_x1_CL_with_Dec = pd.concat([no_match_inference_df_from_x1_CL_with_Dec, temp_infer_from_x1_CL_withdec])
                no_match_inference_df_from_x2_CL_with_Dec = pd.concat([no_match_inference_df_from_x2_CL_with_Dec, temp_infer_from_x2_CL_withdec])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition] = 1-(correct_with_match_from_x1_test_CL_KMFl)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition] = 1-(correct_with_match_from_x2_test_CL_KMFl)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_l[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_l[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition]

                no_match_inference_df_from_x1_CL_KMFl = pd.concat([no_match_inference_df_from_x1_CL_KMFl, temp_infer_from_x1_CL_KMFl])
                no_match_inference_df_from_x2_CL_KMFl = pd.concat([no_match_inference_df_from_x2_CL_KMFl, temp_infer_from_x2_CL_KMFl])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition] = 1-(correct_with_match_from_x1_test_CL_KMFen)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition] = 1-(correct_with_match_from_x2_test_CL_KMFen)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_enc[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_enc[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition]

                no_match_inference_df_from_x1_CL_KMFen = pd.concat([no_match_inference_df_from_x1_CL_KMFen, temp_infer_from_x1_CL_KMFen])
                no_match_inference_df_from_x2_CL_KMFen = pd.concat([no_match_inference_df_from_x2_CL_KMFen, temp_infer_from_x2_CL_KMFen])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = 1-(correct_with_match_from_x1_test_Kang)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_Kang[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]

                # Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG[partition] = len(Mistakes_X1_tr_RG)/max_mistakes
                # Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG[partition] = len(Mistakes_X2_tr_RG)/max_mistakes

                # Frac_mismatches_across_trial_perm_X1_tr_RG[m, run_num] = len(Mistakes_X1_tr_RG)/max_mistakes
                # Frac_mismatches_across_trial_perm_X2_tr_RG[m, run_num] = len(Mistakes_X2_tr_RG)/max_mistakes

                F1_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = F1_fromx1
                F1_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = F1_fromx2

                F1_across_trial_perm_X1_tr[m, run_num] = F1_fromx1
                F1_across_trial_perm_X2_tr[m, run_num] = F1_fromx2

                MatchMSE_across_trial_perm_X1_tr[m, run_num] = match_mse_chimeric
                ReconMSE_across_trial_perm_X1_tr[m, run_num] = recon_mse_chimeric

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = F1_fromx1_simcor
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = F1_fromx2_simcor

                F1_across_trial_perm_X1_tr_sim_Cor[m, run_num] = F1_fromx1_simcor
                F1_across_trial_perm_X2_tr_sim_Cor[m, run_num] = F1_fromx2_simcor

                MatchMSE_across_trial_perm_X1_tr_sim_Cor[m, run_num] = match_mse_kmf

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = F1_x1_CL
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = F1_x2_CL

                F1_across_trial_perm_X1_tr_CL[m, run_num] = F1_x1_CL
                F1_across_trial_perm_X2_tr_CL[m, run_num] = F1_x2_CL

                MatchMSE_across_trial_perm_X1_tr_CL[m,run_num] = match_mse_cl

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = F1_x1_CL_withdec
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = F1_x2_CL_withdec

                F1_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = F1_x1_CL_withdec
                F1_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = F1_x2_CL_withdec

                MatchMSE_across_trial_perm_X1_tr_CL_Dec[m,run_num] = match_mse_cl_dec
                ReconMSE_across_trial_perm_X1_tr_CL_Dec[m, run_num] = recon_mse_cl_dec

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition] = F1_x1_CL_CL_KMFl
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition] = F1_x2_CL_CL_KMFl

                F1_across_trial_perm_X1_tr_CL_KMF_l[m, run_num] = F1_x1_CL_CL_KMFl
                F1_across_trial_perm_X2_tr_CL_KMF_l[m, run_num] = F1_x2_CL_CL_KMFl

                MatchMSE_across_trial_perm_X1_tr_CL_KMF_l[m,run_num] = match_mse_cl_kmfl


                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition] = F1_x1_CL_CL_KMFen
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition] = F1_x2_CL_CL_KMFen

                F1_across_trial_perm_X1_tr_CL_KMF_enc[m, run_num] = F1_x1_CL_CL_KMFen
                F1_across_trial_perm_X2_tr_CL_KMF_enc[m, run_num] = F1_x2_CL_CL_KMFen
                MatchMSE_across_trial_perm_X1_tr_CL_KMF_en[m,run_num] = match_mse_cl_kmfen


                F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = F1_fromx1_Kang

                F1_across_trial_perm_X1_tr_Kang[m, run_num] = F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]

                MatchMSE_across_trial_perm_X1_tr_Kang[m,run_num] = match_mse_kang

                # F1_for_fixed_trial_fixed_num_mapped_X1_tr_RG[partition] = F1_x1_RG
                # F1_for_fixed_trial_fixed_num_mapped_X2_tr_RG[partition] = F1_x2_RG

                # F1_across_trial_perm_X1_tr_RG[m, run_num] = F1_x1_RG
                # F1_across_trial_perm_X2_tr_RG[m, run_num] = F1_x2_RG

                run_num = run_num + 1

                # Deleting the reshuffled as we have already made a copy earlier
                del df_rename_preproc, df_train_preproc, DF_holdout_orig, Df_holdout_r

                # storing the averaged mismatches across all paritition for a fixed trial and fixed number of mapped features
            print(" Chimeric AE when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr != []:
                AVG_MISMATCHES_X1_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr != []:
                AVG_MISMATCHES_X2_tr[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr != []:
                AVG_F1_X1_tr[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr != []:
                AVG_F1_X2_tr[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr)

            print(" Simple_correlation when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor != []:
                AVG_MISMATCHES_X1_tr_sim_Cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor != []:
                AVG_MISMATCHES_X2_tr_sim_Cor[m, trial] = np.average(
                    Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor != []:
                AVG_F1_X1_tr_sim_Cor[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor != []:
                AVG_F1_X2_tr_sim_Cor[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            print(" CL when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL != []:
                AVG_MISMATCHES_X1_tr_CL[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL != []:
                AVG_MISMATCHES_X2_tr_CL[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL != []:
                AVG_F1_X1_tr_CL[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL != []:
                AVG_F1_X2_tr_CL[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL)

            print(" CL + dec when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec != []:
                AVG_MISMATCHES_X1_tr_CL_with_Dec[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec != []:
                AVG_MISMATCHES_X2_tr_CL_with_Dec[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec != []:
                AVG_F1_X1_tr_CL_with_Dec[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec != []:
                AVG_F1_X2_tr_CL_with_Dec[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec)


            print(" CL + KMFl when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l != []:
                AVG_MISMATCHES_X1_tr_CL_KMF_l[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l != []:
                AVG_MISMATCHES_X2_tr_CL_KMF_l[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l != []:
                AVG_F1_X1_tr_CL_KMF_l[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l != []:
                AVG_F1_X2_tr_CL_KMF_l[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l)

            print(" CL + KMF-enc when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc != []:
                AVG_MISMATCHES_X1_tr_CL_KMF_enc[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc != []:
                AVG_MISMATCHES_X2_tr_CL_KMF_enc[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc != []:
                AVG_F1_X1_tr_CL_KMF_enc[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc)
            if F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc != []:
                AVG_F1_X2_tr_CL_KMF_enc[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc)

            print(" Kang et al's MI and HC based method when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)

            print("Value of m and trial is ", m, trial)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang != []:
                AVG_MISMATCHES_X1_tr_Kang[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)

            if F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang != []:
                AVG_F1_X1_tr_Kang[m, trial] = np.average(F1_for_fixed_trial_fixed_num_mapped_X1_tr_Kang)


            print("------")

        m = m + 1
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

    print(" ----  CL----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_CL)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_CL)

    print(" ----  CL + Dec----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_CL_with_Dec)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_CL_with_Dec)

    print(" ----  CL + KMFl ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_CL_KMF_l)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_CL_KMF_l)

    print(" ----  CL + KMF-enc----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_CL_KMF_enc)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_CL_KMF_enc)

    print(" ----  Kang et al's method ----")
    print("\n Array of mistakes for X1 tr")
    print(AVG_MISMATCHES_X1_tr_Kang)
    print("\n Array of mistakes for X2 tr")
    print(AVG_MISMATCHES_X2_tr_Kang)

    # print(" ----  RadialGAN ----")
    # print("\n Array of mistakes for X1 tr")
    # print(AVG_MISMATCHES_X1_tr_RG)
    # print("\n Array of mistakes for X2 tr")
    # print(AVG_MISMATCHES_X2_tr_RG)

    no_match_inference_df_from_x1.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_Chimeric.csv", index=False)
    no_match_inference_df_from_x2.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        hidden_dim) + "L_dim_from_hold_out_sample_Chimeric.csv", index=False)

    no_match_inference_df_from_x1_Sim_cor.to_csv(
        saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
            hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)
    no_match_inference_df_from_x2_Sim_cor.to_csv(
        saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
            hidden_dim) + "L_dim_from_hold_out_sample_Simple_correaltion.csv", index=False)

    no_match_inference_df_from_x1_CL.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL.csv", index=False)
    no_match_inference_df_from_x2_CL.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL.csv", index=False)

    no_match_inference_df_from_x1_CL_with_Dec.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_with_Dec.csv", index=False)
    no_match_inference_df_from_x2_CL_with_Dec.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_with_Dec.csv", index=False)

    no_match_inference_df_from_x1_CL_KMFl.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_KMFl.csv", index=False)
    no_match_inference_df_from_x2_CL_KMFl.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_KMFl.csv", index=False)

    no_match_inference_df_from_x1_CL_KMFen.to_csv(saving_dir + "/" + "Post-hoc_from_x1_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_KMFen.csv", index=False)
    no_match_inference_df_from_x2_CL_KMFen.to_csv(saving_dir + "/" + "Post-hoc_from_x2_" + str(n_t) + "_trials_" + str(
        encKnwMapWidthFinal) + "L_dim_from_hold_out_sample_CL_KMFen.csv", index=False)

    return AVG_F1_X1_tr, AVG_F1_X2_tr, np.average(AVG_F1_X1_tr, axis=1), np.average(AVG_F1_X2_tr, axis = 1), F1_across_trial_perm_X1_tr, F1_across_trial_perm_X2_tr, MatchMSE_across_trial_perm_X1_tr, ReconMSE_across_trial_perm_X1_tr, \
           AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, np.average(AVG_F1_X1_tr_sim_Cor, axis=1), np.average(AVG_F1_X2_tr_sim_Cor, axis = 1), F1_across_trial_perm_X1_tr_sim_Cor, F1_across_trial_perm_X2_tr_sim_Cor, MatchMSE_across_trial_perm_X1_tr_sim_Cor, \
           AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, np.average(AVG_F1_X1_tr_CL, axis=1), np.average(AVG_F1_X2_tr_CL, axis = 1), F1_across_trial_perm_X1_tr_CL, F1_across_trial_perm_X2_tr_CL, MatchMSE_across_trial_perm_X1_tr_CL,  \
           AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, np.average(AVG_F1_X1_tr_CL_with_Dec, axis=1), np.average(AVG_F1_X2_tr_CL_with_Dec, axis = 1), F1_across_trial_perm_X1_tr_CL_with_Dec, F1_across_trial_perm_X2_tr_CL_with_Dec, MatchMSE_across_trial_perm_X1_tr_CL_Dec, ReconMSE_across_trial_perm_X1_tr_CL_Dec, \
           AVG_F1_X1_tr_CL_KMF_l, AVG_F1_X2_tr_CL_KMF_l, np.average(AVG_F1_X1_tr_CL_KMF_l, axis=1), np.average(AVG_F1_X2_tr_CL_KMF_l, axis = 1), F1_across_trial_perm_X1_tr_CL_KMF_l, F1_across_trial_perm_X2_tr_CL_KMF_l, MatchMSE_across_trial_perm_X1_tr_CL_KMF_l, \
           AVG_F1_X1_tr_CL_KMF_enc, AVG_F1_X2_tr_CL_KMF_enc, np.average(AVG_F1_X1_tr_CL_KMF_enc, axis=1), np.average(AVG_F1_X2_tr_CL_KMF_enc, axis = 1), F1_across_trial_perm_X1_tr_CL_KMF_enc, F1_across_trial_perm_X2_tr_CL_KMF_enc, MatchMSE_across_trial_perm_X1_tr_CL_KMF_en,\
           AVG_F1_X1_tr_Kang, np.average(AVG_F1_X1_tr_Kang,axis=1), F1_across_trial_perm_X1_tr_Kang, MatchMSE_across_trial_perm_X1_tr_Kang


# presetting the number of threads to be used
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.cuda.set_per_process_memory_fraction(1.0, device=None)

# starting time of the script
# start_time = datetime.now()

parser = argparse.ArgumentParser(description='HP for CL optimization')

## for the Encoders width (Known mapped)
parser.add_argument("--encKnwMapDepth",  default=4, type=int) #
parser.add_argument("--encKnwMapWidth",  default=65, type=int) #
parser.add_argument("--encKnwMapWidthFinal",  default=25, type=int) # this should be same for all three encoders by design
parser.add_argument("--encKnwMapL2",  default=0.2, type=float)
parser.add_argument("--encKnwMapL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset1 (orig))
parser.add_argument("--encUknwD1OrigDepth",  default=6, type=int) #
parser.add_argument("--encUknwD1OrigWidth",  default=129, type=int) #
parser.add_argument("--encUknwD1OrigWidthFinal",  default=None, type=int) # setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD1OrigL2",  default=0.2, type=float)
parser.add_argument("--encUknwD1OrigL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset2 (r))
parser.add_argument("--encUknwD2ReDepth",  default=2, type=int) #
parser.add_argument("--encUknwD2ReWidth",  default=56, type=int) #
parser.add_argument("--encUknwD2ReWidthFinal",  default=None, type=int) ## setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD2ReL2",  default=0.2, type=float)
parser.add_argument("--encUknwD2ReL1",  default=0.1, type=float)

## learning parameters
parser.add_argument("--batchSize",  default=43, type=int) #
parser.add_argument("--learningRate",  default=0.0003, type=float) #
parser.add_argument("--learningRateFactor",  default=0.2557, type=float) #
parser.add_argument("--LRPatience",  default=4, type=int) #
parser.add_argument("--epochs",  default=40, type=int) #

## CL specific parameters
parser.add_argument("--tau", default=0.6073, type=float)  # temperature parameter in CL loss
parser.add_argument("--masking_ratio", default=0.1797, type=float)  # ratio for creating a new self augmented view
# parser.add_argument("--mp_features", default=10, type=int)  # number of mapped_features
parser.add_argument("--block_stand_comb", default=0, type=int)  # can set the type as bool too but for now int it is
parser.add_argument("--dropout_rate_CL", default=0.1278, type=float)
# not using the following features for now
# parser.add_argument("--mp_features_start", default=2, type=float)  # minimum number of mapped_features
# parser.add_argument("--mp_features_end", default=10, type=float)  # maximum number of mapped_features
# parser.add_argument("--mp_features_steps", default=5, type=float)  # number of mp_features to test between start and stop; check compatibility with abbove two
parser.add_argument("--weightDirDecoder", default =0.5984, type=float)
parser.add_argument("--weightCrossDecoder", default =0.7571, type=float)
parser.add_argument("--weightCombDecoder", default =0.338, type=float)

## dataset and setup parameters
parser.add_argument("--dataset_number",  default='MIMIC') # could be integer or string
parser.add_argument("--outcome",  default="Y")  # this is not very relevant but kep for the sake of completeness
parser.add_argument("--frac_renamed", default=0.5, type=float)
parser.add_argument("--randomSeed", default=8039, type=int )
parser.add_argument("--testDatasetSize", default=0.2, type=float) # this top fraction of data is not used by this code
# parser.add_argument("--dataset_no_sample", default=1, type=int) # this is used to decide which one to tune the HPs on in case of synthetic;
parser.add_argument("--num_of_dataset_samples", default=1, type=int) # number of dataset instances to be used from one  distribution
parser.add_argument("--n_p", default=5, type=int) # number of permutations
parser.add_argument("--n_t", default=3, type=int) # number of data partitioning trials
parser.add_argument("--datatype", default='c') # either continous or binary

## output parameters
parser.add_argument("--git",  default="") # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo",  default="") #
parser.add_argument("--outputcsv",  default="") # csv file where the HP setting with the results are saved


args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing


# enforcing representation size choices across the encoders
if encUknwD1OrigWidthFinal == None or encUknwD2ReWidthFinal ==None:
  encUknwD1OrigWidthFinal = encKnwMapWidthFinal
  encUknwD2ReWidthFinal = encKnwMapWidthFinal


# reproducibility settings
# random_seed = 1 # or any of your favorite number
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(100)


"""  for iterating and initial stuff  """

#input_dir = "/home/trips/"
input_dir = '/input/'

#output_dir = './'
output_dir = '/output/'

# Reading data files (Moose location)
CV_full = pd.read_csv(input_dir + 'MIMIC_feature_confusion/Final_MIMIC_lab_chart_CV.csv')
MV_full = pd.read_csv(input_dir + 'MIMIC_feature_confusion/Final_MIMIC_lab_chart_MV.csv')
# Getting list of all items along with the source and label
item_id_dbsource = pd.read_csv(input_dir + 'd_items_chartevents.csv')
itemid_labs = pd.read_csv(input_dir + 'd_items_labevents.csv')

# temporary patching; need fixing
item_id_dbsource =item_id_dbsource.drop_duplicates(subset=['label','dbsource'], keep='last')


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


#final matching dict
match_dic = dict(zip(CV_itemids_with_match, MV_itemids_with_match))


# itemids with no match
CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]
MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]

print( " CV_itemids_with match ", len(CV_itemids_with_match))
print( " MV_itemids_with match ", len(MV_itemids_with_match))

print( " CV_itemids_with NO match ", len(CV_itemids_withnomatch))
print( " MV_itemids_with NO match ", len(MV_itemids_withnomatch))


num_xtra_feat_inX1 = len(CV_itemids_withnomatch)


# reading the discharge summary embeddings
df_disch_emb_CV = pd.read_csv(input_dir + 'MIMIC_feature_confusion/MIMIC_DischargeSum_emb_CV.csv')
df_disch_emb_MV = pd.read_csv(input_dir + 'MIMIC_feature_confusion/MIMIC_DischargeSum_emb_MV.csv')

df_disch_emb_CV.set_index('hadm_id', inplace=True)
df_disch_emb_MV.set_index('hadm_id', inplace=True)

# breakpoint()
# """  # data details """
# # mpfeatures = len(list_lab_ids)
# n_p = 3  # number of permutations
# n_t = 4  # number of data partitioning trials
list_of_number_mapped_variables = [100]
# list_of_number_mapped_variables = [10,20,30,40,50]
num_of_dataset_samples = 1  # just a placeholder here

# num_xtra_feat_inX1 = len(CV_itemids_withnomatch)
# # num_xtra_feat_inX1 = 0
#
# datatype = 'c'  # b for the case when the data is binarized
#
alpha = 2  # used in KANG method, identified by tuning the value
#
# chimeri model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization
weight_direct = 0.5
weight_cross = 0.8  # 0 denotes no cross loss, 1 denotes cross loss
weight_cycle = 0.8
#
# Chimeric model architecture and parameter details
hidden_dim = 20
num_of_hidden_layers = 2   # 5 as a face value for the hidden data
batch_size = 64
learning_rate = 1e-2
dropout_rate = 0.5
#
# # CL details; moved as argument
# represFinal_dim = 80
# num_of_hidden_layers = 2
# batch_size_CL = 64
# learning_rate_CL = 0.001
# dropout_rate_CL = 0.1
# epochs_CL =10
# masking_ratio = 0.7
# tau = 0.1
# block_stand_comb = 0 # flag if the fingerprints are standardized during CL+KMF methods

known_columnslist = df_disch_emb_CV.columns

# breakpoint()

""" partitioning both the datasets into train and holdout """
full_data_CV0 = df_disch_emb_CV.join(CV_full[onlychart_cont_CV])
full_data_MV0 = df_disch_emb_MV.join(MV_full[onlychart_cont_MV])

if False:
    # plotting the correlation
    Cor_from_df_CV = full_data_CV0.corr()
    Cor_from_df_MV = full_data_MV0.corr()
    import seaborn as sns
    simple_CV_cor_plot = sns.heatmap(Cor_from_df_CV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    simple_CV_cor_plot.set_title("MIMIC-III Carevue (CV) era feature (Dis Sum vs Chart)  correlations")
    simple_CV_cor_plot.hlines([100],*simple_CV_cor_plot.get_xlim(), colors='black')
    simple_CV_cor_plot.vlines([100],*simple_CV_cor_plot.get_ylim(), colors='black')
    fig = simple_CV_cor_plot.get_figure()
    fig.savefig("CV_MIMIC_Block_correlation_DSEmb_vs_Chart.pdf", bbox='tight')
    fig.savefig("CV_MIMIC_Block_correlation_DSEmb_vs_Chart.png", bbox='tight')
    plt.close()

    simple_MV_cor_plot = sns.heatmap(Cor_from_df_MV,cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    simple_MV_cor_plot.set_title("MIMIC-III Metavision (MV) era feature (Dis Sum vs Chart) correlations")
    simple_MV_cor_plot.hlines([100],*simple_MV_cor_plot.get_xlim(), colors='black')
    simple_MV_cor_plot.vlines([100],*simple_MV_cor_plot.get_ylim(), colors='black')
    fig = simple_MV_cor_plot.get_figure()
    fig.savefig("MV_MIMIC_Block_correlation_DSEmb_vs_Chart.pdf", bbox='tight')
    fig.savefig("MV_MIMIC_Block_correlation_DSEmb_vs_Chart.png", bbox='tight')
    plt.close()


upto_test_idx_CV = int(testDatasetSize * len(full_data_CV0))
df_holdout_CV = full_data_CV0.iloc[:upto_test_idx_CV]  # this part of the dataset wass not touched during HP tuning
df_train_CV = full_data_CV0.iloc[upto_test_idx_CV:] # this was the dataset which was divided into two parts for hp tuning; using it fully to train now

upto_test_idx_MV = int(testDatasetSize * len(full_data_MV0))
df_holdout_MV = full_data_MV0.iloc[:upto_test_idx_MV] # this part of the dataset wass not touched during HP tuning
df_train_MV = full_data_MV0.iloc[upto_test_idx_MV:] # this was the dataset which was divided into two parts for hp tuning; using it fully to train now

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

# ##breakpoint()

# file saving logistics
saving_dir = output_dir + 'Emb_Vs_Chart_CL_basedcomparison_'+str(datetime.date.today()) +  '/MIMIC_data/#ofhidden_layers_' + str(
    num_of_hidden_layers) + '/L_dim_' + str(hidden_dim) + "_orthoStatus_" + str(
    orthogonalization_type) + "_BNStatus_" + str(batchnorm) + "_Alpha_" + str(alpha) + "_on_GPU_TWO-stage_vs_KMF_vsothers"+str(datetime.datetime.now())

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

AVG_over_Dataset_samples_X1_tr = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name = saving_dir + "/" + "Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name)

AVG_over_Dataset_samples_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_sim_Cor = saving_dir + "/" + "Simple_correlation_Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_sim_Cor):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_sim_Cor)

AVG_over_Dataset_samples_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_CL = saving_dir + "/" + "CL_Mismatch_metric_L_" + str(encKnwMapWidthFinal) + "_Real_data_BNStatus_" + str(batchnorm) + "_temp_" + str(tau) + "_.txt"

if os.path.exists(file_name_CL):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_CL)

AVG_over_Dataset_samples_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_CL_with_Dec = saving_dir + "/" + "CL_with_Ded_Mismatch_metric_L_" + str(encKnwMapWidthFinal) + "_Real_data_BNStatus_" + str(batchnorm) + "_temp_" + str(tau) + "_.txt"

if os.path.exists(file_name_CL_with_Dec):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_CL_with_Dec)

AVG_over_Dataset_samples_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_CL_KMF_l = saving_dir + "/" + "CL_KMF_l_Mismatch_metric_L_" + str(encKnwMapWidthFinal) + "_Real_data_BNStatus_" + str(batchnorm) + "_temp_" + str(tau) + "_.txt"

if os.path.exists(file_name_CL_KMF_l):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_CL_KMF_l)

AVG_over_Dataset_samples_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_CL_KMF_enc = saving_dir + "/" + "CL_KMF_enc_Mismatch_metric_L_" + str(encKnwMapWidthFinal) + "_Real_data_BNStatus_" + str(batchnorm) + "_temp_" + str(tau) + "_.txt"

if os.path.exists(file_name_CL_KMF_enc):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_CL_KMF_enc)

AVG_over_Dataset_samples_X1_tr_KANG= np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_KANG = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_KANG = saving_dir + "/" + "KANG_Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_KANG):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_KANG)

AVG_over_Dataset_samples_X1_tr_RG = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_RG = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

file_name_RG = saving_dir + "/" + "RG_Mismatch_metric_L_" + str(hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_RG):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_RG)

f = open(file_name, 'w')
f.write("\n \n *** Chimeric AE Present file settings ***")
f.write("\n \n MIMIC data results  ")
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

f = open(file_name_CL, 'w')
f.write("\n \n *** CL Present file settings ***")
f.write("\n \n MIMIC data results  ")
f.write("\n Batch norm {0}\t ".format(batchnorm))
f.write("\n Size of L {0}\t".format(encKnwMapWidthFinal))
f.write("\n Temperature {0}\t".format(tau))
f.write("\n Number of epochs {0}\t".format(epochs))
f.write("\n Starting learning rate {0}\t ".format(learningRate))
f.write("\n Batch size {0}\t".format(batchSize))
f.write("\n Masking ratio {0}\t".format(masking_ratio))
f.write("\n Initial dropout rate {0}\t".format(dropout_rate_CL))
f.write("\n")
f.close()


f = open(file_name_sim_Cor, 'w')
f.write("\n \n *** Simple_correlation Present file settings ***")
f.close()

f = open(file_name_KANG, 'w')
f.write("\n \n *** KANG Present file settings ***")
f.close()

for sample_no in range(1,num_of_dataset_samples+1):
    print("\n ********************************************************")
    print(" \n Run STARTS for sample no ", sample_no, "\n")
    print(" ******************************************************** \n")

    # AVG_F1_X1_tr, AVG_F1_X2_tr, m_x1, m_x2_tr, F1_elongated_X1_tr, F1_elongated_X2_tr, AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, m_x1_sim_Cor, m_x2_tr_sim_Cor, F1_elongated_X1_tr_sim_Cor, F1_elongated_X2_tr_sim_Cor, AVG_F1_X1_tr_KANG, m_X1_tr_KANG, F1_elongated_X1_tr_KANG, AVG_F1_X1_tr_RG, AVG_F1_X2_tr_RG, m_x1_RG, m_x2_tr_RG, F1_elongated_X1_tr_RG, F1_elongated_X2_tr_RG  = main(sample_no)
    AVG_F1_X1_tr, AVG_F1_X2_tr, m_x1, m_x2_tr, F1_elongated_X1_tr, F1_elongated_X2_tr, MatchMSE_elongated_X1_tr, ReconMSE_elongated_X1_tr , AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, m_x1_sim_Cor, m_x2_tr_sim_Cor, F1_elongated_X1_tr_sim_Cor, F1_elongated_X2_tr_sim_Cor, MatchMSE_elongated_X1_tr_simCor ,AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, m_x1_CL, m_x2_tr_CL, F1_elongated_X1_tr_CL, F1_elongated_X2_tr_CL, MatchMSE_elongated_X1_tr_CL, AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, m_x1_CL_with_Dec, m_x2_tr_CL_with_Dec, F1_elongated_X1_tr_CL_with_Dec, F1_elongated_X2_tr_CL_with_Dec, MatchMSE_elongated_X1_tr_CL_Dec, ReconMSE_elongated_X1_tr_CL_Dec, AVG_F1_X1_tr_CL_kmfl, AVG_F1_X2_tr_CL_kmfl, m_x1_CL_kmfl, m_x2_tr_CL_kmfl, F1_elongated_X1_tr_CL_kmfl, F1_elongated_X2_tr_CL_kmfl, MatchMSE_elongated_X1_tr_CL_KMFl, AVG_F1_X1_tr_CL_kmfen, AVG_F1_X2_tr_CL_kmfen, m_x1_CL_kmfen, m_x2_tr_CL_kmfen, F1_elongated_X1_tr_CL_kmfen, F1_elongated_X2_tr_CL_kmfen, MatchMSE_elongated_X1_tr_CL_KMFen, AVG_F1_X1_tr_KANG, m_X1_tr_KANG, F1_elongated_X1_tr_KANG, MatchMSE_elongated_X1_tr_KANG = main(sample_no)

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
        MatchMSE_elongated_X1_tr_list = MatchMSE_elongated_X1_tr
        ReconMSE_elongated_X1_tr_list = ReconMSE_elongated_X1_tr
    else:
        F1_elongated_x1_tr_list = np.hstack((F1_elongated_x1_tr_list, F1_elongated_X1_tr))
        F1_elongated_x2_tr_list = np.hstack((F1_elongated_x2_tr_list, F1_elongated_X2_tr))
        MatchMSE_elongated_X1_tr_list = np.hstack((MatchMSE_elongated_X1_tr_list, MatchMSE_elongated_X1_tr))
        ReconMSE_elongated_X1_tr_list = np.hstack((ReconMSE_elongated_X1_tr_list, ReconMSE_elongated_X1_tr))


    # for Simple_correlation (Simple_correlation)

    AVG_over_Dataset_samples_X1_tr_sim_Cor[:,sample_no-1] = m_x1_sim_Cor
    AVG_over_Dataset_samples_X2_tr_sim_Cor[:,sample_no-1] = m_x2_tr_sim_Cor

    f = open(file_name_sim_Cor,'a')
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
        MatchMSE_elongated_X1_tr_list_sim_Cor = MatchMSE_elongated_X1_tr_simCor
    else:
        F1_elongated_x1_tr_list_sim_Cor = np.hstack((F1_elongated_x1_tr_list_sim_Cor, F1_elongated_X1_tr_sim_Cor))
        F1_elongated_x2_tr_list_sim_Cor = np.hstack((F1_elongated_x2_tr_list_sim_Cor, F1_elongated_X2_tr_sim_Cor))
        MatchMSE_elongated_X1_tr_list_sim_Cor = np.hstack((MatchMSE_elongated_X1_tr_list_sim_Cor, MatchMSE_elongated_X1_tr_simCor))


    # for CL

    AVG_over_Dataset_samples_X1_tr_CL[:,sample_no-1] = m_x1_CL
    AVG_over_Dataset_samples_X2_tr_CL[:,sample_no-1] = m_x2_tr_CL

    f = open(file_name_CL,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_CL))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr_CL))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL = F1_elongated_X1_tr_CL
        F1_elongated_x2_tr_list_CL = F1_elongated_X2_tr_CL
        MatchMSE_elongated_X1_tr_list_CL = MatchMSE_elongated_X1_tr_CL
    else:
        F1_elongated_x1_tr_list_CL = np.hstack((F1_elongated_x1_tr_list_CL, F1_elongated_X1_tr_CL))
        F1_elongated_x2_tr_list_CL = np.hstack((F1_elongated_x2_tr_list_CL, F1_elongated_X2_tr_CL))
        MatchMSE_elongated_X1_tr_list_CL = np.hstack((MatchMSE_elongated_X1_tr_list_CL, MatchMSE_elongated_X1_tr_CL))


    # for CL + dec

    AVG_over_Dataset_samples_X1_tr_CL_with_Dec[:,sample_no-1] = m_x1_CL_with_Dec
    AVG_over_Dataset_samples_X2_tr_CL_with_Dec[:,sample_no-1] = m_x2_tr_CL_with_Dec

    f = open(file_name_CL_with_Dec,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_CL_with_Dec))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr_CL_with_Dec))
    f.write("\n \n ")
    f.close()

    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_with_Dec = F1_elongated_X1_tr_CL_with_Dec
        F1_elongated_x2_tr_list_CL_with_Dec = F1_elongated_X2_tr_CL_with_Dec
        MatchMSE_elongated_X1_tr_list_CL_with_Dec = MatchMSE_elongated_X1_tr_CL_Dec
        ReconMSE_elongated_X1_tr_list_CL_with_Dec = ReconMSE_elongated_X1_tr_CL_Dec
    else:
        F1_elongated_x1_tr_list_CL_with_Dec = np.hstack((F1_elongated_x1_tr_list_CL_with_Dec, F1_elongated_X1_tr_CL_with_Dec))
        F1_elongated_x2_tr_list_CL_with_Dec = np.hstack((F1_elongated_x2_tr_list_CL_with_Dec, F1_elongated_X2_tr_CL_with_Dec))
        MatchMSE_elongated_X1_tr_list_CL_with_Dec = np.hstack((MatchMSE_elongated_X1_tr_list_CL_with_Dec, MatchMSE_elongated_X1_tr_CL_Dec))
        ReconMSE_elongated_X1_tr_list_CL_with_Dec = np.hstack((ReconMSE_elongated_X1_tr_list_CL_with_Dec, ReconMSE_elongated_X1_tr_CL_Dec))


    # for CL + KMFl

    AVG_over_Dataset_samples_X1_tr_CL_KMF_l[:,sample_no-1] = m_x1_CL_kmfl
    AVG_over_Dataset_samples_X2_tr_CL_KMF_l[:,sample_no-1] = m_x2_tr_CL_kmfl

    f = open(file_name_CL_KMF_l,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_CL_kmfl))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr_CL_kmfl))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_kmfl = F1_elongated_X1_tr_CL_kmfl
        F1_elongated_x2_tr_list_CL_kmfl = F1_elongated_X2_tr_CL_kmfl
        MatchMSE_elongated_X1_tr_list_CL_KMFl = MatchMSE_elongated_X1_tr_CL_KMFl
    else:
        F1_elongated_x1_tr_list_CL_kmfl = np.hstack((F1_elongated_x1_tr_list_CL_kmfl, F1_elongated_X1_tr_CL_kmfl))
        F1_elongated_x2_tr_list_CL_kmfl = np.hstack((F1_elongated_x2_tr_list_CL_kmfl, F1_elongated_X2_tr_CL_kmfl))
        MatchMSE_elongated_X1_tr_list_CL_KMFl = np.hstack((MatchMSE_elongated_X1_tr_list_CL_KMFl, MatchMSE_elongated_X1_tr_CL_KMFl))

    # for CL + KMFen

    AVG_over_Dataset_samples_X1_tr_CL_KMF_enc[:,sample_no-1] = m_x1_CL_kmfen
    AVG_over_Dataset_samples_X2_tr_CL_KMF_enc[:,sample_no-1] = m_x2_tr_CL_kmfen

    f = open(file_name_CL_KMF_enc,'a')
    f.write("\n \n F1 for different trials on sample number {0}".format(sample_no))
    f.write("\n X1_train \n")
    f.write("{0}".format(AVG_F1_X1_tr_CL_kmfen))
    f.write("\n X2_train \n")
    f.write("{0}".format(AVG_F1_X2_tr_CL_kmfen))
    f.write("\n \n ")
    f.close()


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_kmfen = F1_elongated_X1_tr_CL_kmfen
        F1_elongated_x2_tr_list_CL_kmfen = F1_elongated_X2_tr_CL_kmfen
        MatchMSE_elongated_X1_tr_list_CL_KMFen = MatchMSE_elongated_X1_tr_CL_KMFen
    else:
        F1_elongated_x1_tr_list_CL_kmfen = np.hstack((F1_elongated_x1_tr_list_CL_kmfen, F1_elongated_X1_tr_CL_kmfen))
        F1_elongated_x2_tr_list_CL_kmfen = np.hstack((F1_elongated_x2_tr_list_CL_kmfen, F1_elongated_X2_tr_CL_kmfen))
        MatchMSE_elongated_X1_tr_list_CL_KMFen = np.hstack((MatchMSE_elongated_X1_tr_list_CL_KMFen, MatchMSE_elongated_X1_tr_CL_KMFen))

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
        MatchMSE_elongated_X1_tr_list_KANG = MatchMSE_elongated_X1_tr_KANG
        # F1_elongated_x2_tr_list_KANG = F1_elongated_X2_tr_KANG
    else:
        F1_elongated_x1_tr_list_KANG = np.hstack((F1_elongated_x1_tr_list_KANG, F1_elongated_X1_tr_KANG))
        # F1_elongated_x2_tr_list_KANG = np.hstack((F1_elongated_x2_tr_list_KANG, F1_elongated_X2_tr_KANG))
        MatchMSE_elongated_X1_tr_list_KANG = np.hstack((MatchMSE_elongated_X1_tr_list_KANG, MatchMSE_elongated_X1_tr_KANG))

    print("\n ********************************************************")
    print(" \n Run ENDS for sample no ", sample_no, "\n ")
    print(" ******************************************************** \n")

file_name_violin = saving_dir + "/" + "F1_For_violin_Mismatch_metric_L_" + str(
    hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_violin):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_violin)

f = open(file_name_violin,'a')
f.write("\n \n List of mapped features \n ")
f.write("{0}".format(list_of_number_mapped_variables))
f.write("\n \n F1 for across trials and perm ")
f.write("\n X1_train \n")
f.write("{0}".format(F1_elongated_x1_tr_list))
f.write("\n X2_train \n")
f.write("{0}".format(F1_elongated_x2_tr_list))
f.write("\n X1_train_CL \n")
f.write("{0}".format(F1_elongated_x1_tr_list_CL))
f.write("\n X2_train_CL \n")
f.write("{0}".format(F1_elongated_x2_tr_list_CL))
f.write("\n X1_train_CL_Dec \n")
f.write("{0}".format(F1_elongated_x1_tr_list_CL_with_Dec))
f.write("\n X2_train_CL_Dec \n")
f.write("{0}".format(F1_elongated_x2_tr_list_CL_with_Dec))
f.write("\n X1_train_CL_Dec_KMFl \n")
f.write("{0}".format(F1_elongated_x1_tr_list_CL_kmfl))
f.write("\n X2_train_CL_Dec_KMFl \n")
f.write("{0}".format(F1_elongated_x2_tr_list_CL_kmfl))
f.write("\n X1_train_CL_Dec_KMFen \n")
f.write("{0}".format(F1_elongated_x1_tr_list_CL_kmfen))
f.write("\n X2_train_CL_Dec_KMFen \n")
f.write("{0}".format(F1_elongated_x2_tr_list_CL_kmfen))
f.write("\n X1_train_KANG \n")
f.write("{0}".format(F1_elongated_x1_tr_list_KANG))
f.write("\n X1_train_sim_Cor \n")
f.write("{0}".format(F1_elongated_x1_tr_list_sim_Cor))
f.write("\n X2_train_sim_Cor \n")
f.write("{0}".format(F1_elongated_x2_tr_list_sim_Cor))
f.write("\n \n ")
f.close()

file_name_violin_match_mse = saving_dir + "/" + "MatchMSE_For_violin_Mismatch_metric_L_" + str(
    hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_violin_match_mse):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_violin_match_mse)

f = open(file_name_violin_match_mse,'a')
f.write("\n \n List of mapped features \n ")
f.write("{0}".format(list_of_number_mapped_variables))
f.write("\n \n Match MSE for across trials and perm ")
f.write("\n X1_train \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list))
f.write("\n X1_train_CL \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_CL))
f.write("\n X1_train_CL_Dec \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_CL_with_Dec))
f.write("\n X1_train_CL_Dec_KMFl \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_CL_KMFl))
f.write("\n X1_train_CL_Dec_KMFen \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_CL_KMFen))
f.write("\n X1_train_KANG \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_KANG))
f.write("\n X1_train_sim_Cor \n")
f.write("{0}".format(MatchMSE_elongated_X1_tr_list_sim_Cor))
f.write("\n X2_train_sim_Cor \n") # this is just for down the line preprocessing
f.write("\n \n ")
f.close()


file_name_violin_recon_mse = saving_dir + "/" + "ReconMSE_For_violin_Mismatch_metric_L_" + str(
    hidden_dim) + "_Real_data_orthoStatus_" + str(orthogonalization_type) + "_BNStatus_" + str(batchnorm) + ".txt"

if os.path.exists(file_name_violin_recon_mse):  # removing the existing file so that it doesn't append the file from previous experiments
    os.remove(file_name_violin_recon_mse)

f = open(file_name_violin_recon_mse,'a')
f.write("\n \n List of mapped features \n ")
f.write("{0}".format(list_of_number_mapped_variables))
f.write("\n \n Recon  MSE for across trials and perm ")
f.write("\n X1_train \n")
f.write("{0}".format(ReconMSE_elongated_X1_tr_list))
f.write("\n X1_train_CL_Dec \n")
f.write("{0}".format(ReconMSE_elongated_X1_tr_list_CL_with_Dec))
f.write("\n X2_train_CL_Dec \n") # this is just for down the line preprocessing
f.write("\n \n ")
f.close()


# Computing the average over the datset samples
Mean_over_trials_mismatches_X1_tr = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr = np.zeros(len(list_of_number_mapped_variables))

Mean_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_number_mapped_variables))

Mean_over_trials_mismatches_X1_tr_CL = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_CL = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr_CL = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr_CL = np.zeros(len(list_of_number_mapped_variables))

Mean_over_trials_mismatches_X1_tr_CL_with_Dec = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_CL_with_Dec = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr_CL_with_Dec = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr_CL_with_Dec = np.zeros(len(list_of_number_mapped_variables))


Mean_over_trials_mismatches_X1_tr_CL_kmfl = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_CL_kmfl = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr_CL_kmfl = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr_CL_kmfl = np.zeros(len(list_of_number_mapped_variables))

Mean_over_trials_mismatches_X1_tr_CL_kmfen = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_CL_kmfen = np.zeros(len(list_of_number_mapped_variables))
Mean_over_trials_mismatches_X2_tr_CL_kmfen = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X2_tr_CL_kmfen = np.zeros(len(list_of_number_mapped_variables))

Mean_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_number_mapped_variables))
SD_over_trials_mismatches_X1_tr_KANG = np.zeros(len(list_of_number_mapped_variables))

# Mean_over_trials_mismatches_X1_tr_RG = np.zeros(len(list_of_number_mapped_variables))
# SD_over_trials_mismatches_X1_tr_RG = np.zeros(len(list_of_number_mapped_variables))
# Mean_over_trials_mismatches_X2_tr_RG = np.zeros(len(list_of_number_mapped_variables))
# SD_over_trials_mismatches_X2_tr_RG = np.zeros(len(list_of_number_mapped_variables))

x_axis = np.arange(len(list_of_number_mapped_variables))
x_axis1 = x_axis + 0.05
x_axis2 = x_axis + 0.1

for i in range(len(list_of_number_mapped_variables)):
    Mean_over_trials_mismatches_X1_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr[i] = np.round(np.std(F1_elongated_x1_tr_list[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr[i] = np.round(np.std(F1_elongated_x2_tr_list[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_sim_Cor[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_sim_Cor[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.std(F1_elongated_x1_tr_list_sim_Cor[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.std(F1_elongated_x2_tr_list_sim_Cor[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_KANG[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_KANG[i] = np.round(np.std(F1_elongated_x1_tr_list_KANG[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_CL[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_CL[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_CL[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_CL[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_CL[i] = np.round(np.std(F1_elongated_x1_tr_list_CL[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_CL[i] = np.round(np.std(F1_elongated_x2_tr_list_CL[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_CL_with_Dec[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_CL_with_Dec[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_CL_with_Dec[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_CL_with_Dec[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_CL_with_Dec[i] = np.round(np.std(F1_elongated_x1_tr_list_CL_with_Dec[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_CL_with_Dec[i] = np.round(np.std(F1_elongated_x2_tr_list_CL_with_Dec[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_CL_kmfl[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_CL_KMF_l[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_CL_kmfl[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_CL_KMF_l[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_CL_kmfl[i] = np.round(np.std(F1_elongated_x1_tr_list_CL_kmfl[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_CL_kmfl[i] = np.round(np.std(F1_elongated_x2_tr_list_CL_kmfl[i, :]), decimals=4)

    Mean_over_trials_mismatches_X1_tr_CL_kmfen[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_CL_KMF_enc[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_CL_kmfen[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_CL_KMF_enc[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_CL_kmfen[i] = np.round(np.std(F1_elongated_x1_tr_list_CL_kmfen[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_CL_kmfen[i] = np.round(np.std(F1_elongated_x2_tr_list_CL_kmfen[i, :]), decimals=4)




plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr, color='blue', label=" Chimeric AE ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr, yerr=SD_over_trials_mismatches_X1_tr, linestyle="solid",
             color='blue')
plt.scatter(x_axis1, Mean_over_trials_mismatches_X1_tr_sim_Cor, color='red', label=" KMF ", linestyle='None')
plt.errorbar(x_axis1, Mean_over_trials_mismatches_X1_tr_sim_Cor, yerr=SD_over_trials_mismatches_X1_tr_sim_Cor, linestyle="solid",
             color='red')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_CL, color='pink', label=" CL ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_CL, yerr=SD_over_trials_mismatches_X1_tr_CL, linestyle="solid",
             color='pink')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_CL_with_Dec, color='orange', label=" CL + Dec ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_CL_with_Dec, yerr=SD_over_trials_mismatches_X1_tr_CL_with_Dec, linestyle="solid",
             color='orange')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_CL_kmfl, color='brown', label=" CL + KMFl", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_CL_kmfl, yerr=SD_over_trials_mismatches_X1_tr_CL_kmfl, linestyle="solid",
             color='brown')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_CL_kmfen, color='green', label=" CL + KMFen", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_CL_kmfen, yerr=SD_over_trials_mismatches_X1_tr_CL_kmfen, linestyle="solid",
             color='green')
plt.scatter(x_axis, Mean_over_trials_mismatches_X1_tr_KANG, color='black', label=" KANG ", linestyle='None')
plt.errorbar(x_axis, Mean_over_trials_mismatches_X1_tr_KANG, yerr=SD_over_trials_mismatches_X1_tr_KANG, linestyle="solid",
            color='black')
plt.xticks(x_axis, np.array(list_of_number_mapped_variables))
plt.yticks(np.linspace(0,1,11))
plt.xlabel("Number of mapped features")
plt.ylabel("F1 score across different dataset samples")
plt.title("MIMIC data")
plt.legend()
if num_xtra_feat_inX1 != 0:
    plt.savefig(saving_dir + "/Partial_F1_Comp_ChimVsRG_X1_tr_Real_data_varyingData_num_Mapped_fea_" + str(
        len(list_of_number_mapped_variables)) + ".pdf", bbox='tight')
    plt.savefig(saving_dir + "/Partial_F1_Comp_ChimVsRG_X1_tr_Real_data_varyingData_num_Mapped_fea_" + str(
        len(list_of_number_mapped_variables)) + ".png", bbox='tight')
else:
    plt.savefig(saving_dir + "/Onto_F1_Comp_ChimVsRG_X1_tr_Real_data_varyingData_num_Mapped_fea_" + str(
        len(list_of_number_mapped_variables)) + ".pdf", bbox='tight')
    plt.savefig(saving_dir + "/Onto_F1_Comp_ChimVsRG_X1_tr_Real_data_varyingData_num_Mapped_fea_" + str(
        len(list_of_number_mapped_variables)) + ".png", bbox='tight')
plt.close()


