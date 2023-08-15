"""
This code uses contrastive learning for schema matching problem in following steps:
    1) Trains three encoders, one for the common known mapped features, two for the unmapped features from two databases using a contraastive loss.
    2) Positive pairs are the encoder outputs of mapped and unmapped sets from the same example and rest are unmappped. We can randomly sample for the nnegative pairs.

INPUT:

Full dataset, the model details, number of permutations, number of partitioning of dataset, fraction of data to be permuted, number of mapped features

OUTPUT:

An array with rows as the number of mapped features and columns as the trial number and cell entry as the avg fraction of mismatches for the a fixed trial and the fixed number of mapped variables


This code has randomness over mapped features and unmapped features too

This is the partial mapping setup and this file isused for hp tuning


"""

##TODO : clean up the names (train test etc) once the code starts working

## importing packages

# from sklearn.manifold import TSNE
# import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import svm, linear_model, model_selection, metrics, ensemble
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise,mutual_info_score, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from scipy import linalg, stats
import xgboost as xgb
from matching.games import StableMarriage, HospitalResident
import pingouin as pg
# from imblearn.under_sampling import RandomUnderSampler
# import matplotlib.pyplot as plt
from collections import Counter
import random
from itertools import combinations
import datetime
from datetime import datetime
import os.path
import json, sys, argparse
import math
import pickle


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
        # breakpoint()
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        try:
            l_pos = torch.diag(similarity, self.batch_size)
        except RuntimeError:
            print("Error encountered. Debug.")
            breakpoint()
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
        # breakpoint()
        x_bar = x + torch.normal(0, noise_level, size = x.shape, device='cuda')
    else:
        x_bar = x_bar

    return x_bar

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

def Stable_matching_algorithm(C_X1_train, C_X2_train, index_O_to_R, index_R_to_O,num_mapped_axis):
    # creating the preference dictionaries
    ####### ----------  X1 train ------------- ##########


    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}

    for i in range(C_X1_train.shape[0]):
        sorted_index = np.argsort(-C_X1_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X1_train.shape[1]):
        sorted_index = np.argsort(-C_X1_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X1_train)
    # print(cross_recon_features_pref_X1_train)

    game_X1_train = StableMarriage.create_from_dictionaries(true_features_pref_X1_train,
                                                            cross_recon_features_pref_X1_train)

    ####### ----------  X2 train ------------- ##########

    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}

    for i in range(C_X2_train.shape[0]):
        sorted_index = np.argsort(-C_X2_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):
        sorted_index = np.argsort(-C_X2_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X2_train)
    # print(cross_recon_features_pref_X2_train)

    game_X2_train = StableMarriage.create_from_dictionaries(true_features_pref_X2_train,
                                                            cross_recon_features_pref_X2_train)


    ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)


    # for comparison to the the initial index that were passed
    x1_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x2_train.values()])

    # getting the number of mismatches
    mismatched_x1_train = [i for i, j in zip(index_O_to_R, x1_train_y) if i != j]
    mismatched_x2_train = [i for i, j in zip(index_R_to_O, x2_train_y) if i != j]

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(C_X2_train.shape)

    for i in range(matching_x1_train_matrix.shape[0]):
        # print(i, x1_train_y[i]-1)
        matching_x1_train_matrix[i,x1_train_y[i]-1-num_mapped_axis]=1


    for i in range(matching_x2_train_matrix.shape[0]):
        # print(i, x2_train_y[i]-1)
        matching_x2_train_matrix[i,x2_train_y[i]-1-num_mapped_axis]=1

    print("Mistakes x1")
    print(mismatched_x1_train)
    print(" Mistakes x2 train")
    print(mismatched_x2_train)

    return mismatched_x1_train, mismatched_x2_train, matching_x1_train_matrix, matching_x2_train_matrix

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

    def forward(self, cont_data, cross=2):
        if cross == 2:
            # this is basically to separate the encoder part of AE and use the Jacobian from encoder's output
            activation = self.encoder_hidden_layer1(cont_data)
            if self.batchnorm == 1:
                activation = self.bn1(activation)
            activation = self.encoder_hidden_layer2(activation)
            if self.batchnorm == 1:
                activation = self.bn2(activation)
            activation = torch.tanh(activation)
            activation = self.drop_layer2(activation)
            code0 = self.encoder_output_layer(activation)
            return code0

        if cross == 0:
            # print("inside the normal loop")
            try:
                activation = self.encoder_hidden_layer1(cont_data)
            except:
                print("stuck")
                # breakpoint()
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
        if cross == 1:
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

def Train_cross_AE(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,  Df_holdout_orig, DF_holdout_r, unmapped_features_to_drop_from_orig, normalizing_values_orig, normalizing_values_r, DF_holdout_orig0_not_includedwhiletraining):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1


    num_features = len(reordered_column_names_r) -1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

    print(" -------- Chimeric AE training -------------  ")

    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batch_size, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batch_size, shuffle=True, num_workers=1)

    # if datatype == 'b':
    #     model_orig = AE_binary(input_shape=num_NonCat_features_orig, drop_out_p=dropout_rate).to(device)
    #     model_r = AE_binary(input_shape=num_NonCat_features_r, drop_out_p=dropout_rate).to(device)
    #     criterion = nn.BCELoss()

    if datatype == 'c':
        model_orig = AE_2_hidden_layer(input_shape=num_NonCat_features_orig, batchnorm=batchnorm, drop_out_p=dropout_rate).to(
                    device)
        model_r = AE_2_hidden_layer(input_shape=num_NonCat_features_r, batchnorm=batchnorm, drop_out_p=dropout_rate).to(device)
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
                train_loss_cross_r = criterion(output_cross_r[:, :len(mapped_features)],
                                               x_r[:, :len(mapped_features)])
                train_loss_cross_orig = criterion(output_cross_orig[:, :len(mapped_features)],
                                                  x_o[:, :len(mapped_features)])

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

    # comparing actual reconstruction and cross recontruction on original data
    # latent_code_Orig_fullTrain_orig, recons_orig_train_from_orig = model_orig(
    #     torch.Tensor(df_train_preproc.iloc[:, :-1].values), 0)
    # _, recons_orig_train_frommodelR = model_r(latent_code_Orig_fullTrain_orig, 1)
    #
    # # comparing actual reconstruction and cross recontruction on renamed data
    # latent_code_renamed, recons_rename_train_frommodelR = model_r(
    #     torch.Tensor(df_rename_preproc.iloc[:, :-1].values), 0)
    # _, recons_rename_train_frommodelOrig = model_orig(latent_code_renamed, 1)
    #
    #
    # features_reconst_from_crossR = recons_orig_train_frommodelR.detach().numpy()
    # features_true_orig = df_train_preproc.iloc[:, :-1].values
    #
    # features_reconst_from_crossO = recons_rename_train_frommodelOrig.detach().numpy()
    # features_true_renamed = df_rename_preproc.iloc[:, :-1].values
    #
    #
    # # computing the correlation matrix between original feature values and cross reconstruction
    # CorMatrix_X1_X1_hat_cross = np.zeros((num_features, num_features))
    # CorMatrix_X2_X2_hat_cross = np.zeros((num_features, num_features))
    #
    # for i in range(num_features):
    #     for j in range(num_features):
    #         CorMatrix_X1_X1_hat_cross[i, j] = \
    #             stats.pearsonr(features_true_orig[:, i], features_reconst_from_crossR[:, j])[0]
    #         CorMatrix_X2_X2_hat_cross[i, j] = \
    #             stats.pearsonr(features_true_renamed[:, i], features_reconst_from_crossO[:, j])[0]
    #
    # # selecting the correlation only for unmapped variables
    # short_CorMatrix_X1_X1_hat_cross = CorMatrix_X1_X1_hat_cross[len(mapped_features):,
    #                                   len(mapped_features):]
    # short_CorMatrix_X2_X2_hat_cross = CorMatrix_X2_X2_hat_cross[len(mapped_features):,
    #                                   len(mapped_features):]


    """ JACOBIAN CALCULATION  """
    # breakpoint()
    for param in model_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, :-1].values).to(device)
    grad_sum_unkn_o = torch.zeros((hidden_dim, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_o += torch.autograd.functional.jacobian(model_orig, temp_input[i])

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in model_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, :-1].values).to(device)
    grad_sum_unkn_r = torch.zeros((hidden_dim, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_r += torch.autograd.functional.jacobian(model_r, temp_input[i])

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)

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

    CorMatrix_X1_X1_hat_dir_test = np.zeros((num_NonCat_features_orig, num_NonCat_features_orig))
    for i in range(num_NonCat_features_orig):
        for j in range(num_NonCat_features_orig):
            temp = stats.pearsonr(features_true_orig_test[:, i],
                                  recons_orig_Test_from_orig.cpu().detach().numpy()[:, j])
            CorMatrix_X1_X1_hat_dir_test[i, j] = temp[0]

    CorMatrix_X2_X2_hat_dir_test = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
    for i in range(num_NonCat_features_r):
        for j in range(num_NonCat_features_r):
            temp = stats.pearsonr(features_true_renamed_test[:, i],
                                  recons_rename_Test_frommodelR.cpu().detach().numpy()[:, j])
            CorMatrix_X2_X2_hat_dir_test[i, j] = temp[0]

    np.savetxt("Cor_dir_X1.csv", np.round(CorMatrix_X1_X1_hat_dir_test, decimals=2), delimiter=",")
    np.savetxt("Cor_dir_X2.csv", np.round(CorMatrix_X2_X2_hat_dir_test, decimals=2), delimiter=",")



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


    print(" \n Mistakes by the stage Chimeric method on holdout data")
    print(" Chimeric  X1_train mistakes number on holdout set", unmapped_features_orig-correct_with_match_from_x1_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" Chimeric  X2_train mistakes number on holdout set", unmapped_features_orig-correct_with_match_from_x2_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

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


    print(" -------- Chimeric AE method training ends ------------- \n \n  ")

    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test.shape[0]):
        for j in range(x1_match_matrix_test.shape[1]):
            if x1_match_matrix_test[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])

    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))


    true_match_list = list(set(to_map_r).intersection(to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]

    # Matching metric error
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values, Df_holdout_orig[final_dic_for_compar_matching.values()])

    incorrect_match_dict_x1 = {}

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
                incorrect_match_dict_x1[to_map_orig[i]] = to_map_r[j]
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
                FP_x1 = FP_x1 + 1
                incorrect_match_dict_x1[to_map_orig[i]] = to_map_r[j]

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


    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(list(Df_holdout_orig.columns[mpfeatures:-1]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(list(DF_holdout_r.columns[mpfeatures:-1]).index(i))

    # oracle combination metric
    overall_quality_oracle_comb = criterion(recons_orig_Test_frommodelR[:,
                                                incorrect_match_idx_r_from_x1],
                                                torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)[:,
                                                incorrect_match_idx_orig_from_x1])

    del df_rename_preproc
    # exit()
    return o_to_r_sim, r_to_o_sim, correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2, overall_quality_error_matching_only, overall_quality_oracle_comb

def Train_CL(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,  Df_holdout_orig, DF_holdout_r,partition, trial):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1

    num_features = len(reordered_column_names_r) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batchSize, shuffle=True, num_workers=1)
    dataset_orig_val = TabularDataset(data=Df_holdout_orig, output_col=outcome)
    val_loader_orig = DataLoader(dataset_orig_val, batchSize, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batchSize, shuffle=True, num_workers=1)
    dataset_r_val = TabularDataset(data=DF_holdout_r, output_col=outcome)
    val_loader_r = DataLoader(dataset_r_val, batchSize, shuffle=True, num_workers=1)

    # known_features_encoder = AE_2_hidden_layer_CL(input_shape=mpfeatures, batchnorm=batchnorm, drop_out_p=dropout_rate_CL,
    #                                            repres_dim=represFinal_dim).to(
    #     device)
    # unknown_features_encoder_orig = AE_2_hidden_layer_CL(input_shape=num_NonCat_features_orig - mpfeatures,
    #                                                   batchnorm=batchnorm, drop_out_p=dropout_rate_CL,
    #                                                   repres_dim=represFinal_dim).to(
    #     device)
    # unknown_features_encoder_r = AE_2_hidden_layer_CL(input_shape=num_NonCat_features_r - mpfeatures, batchnorm=batchnorm,
    #                                                drop_out_p=dropout_rate_CL, repres_dim=represFinal_dim).to(
    #     device)

    known_features_encoder = AE_CL(input_shape=mpfeatures, hidden_units_final=encKnwMapWidthFinal,
                                                  hidden_depth=encKnwMapDepth,
                                                  hidden_units=encKnwMapWidth,drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_orig = AE_CL(input_shape=num_NonCat_features_orig - mpfeatures, hidden_units_final=encUknwD1OrigWidthFinal,
                                                  hidden_depth=encUknwD1OrigDepth,
                                                  hidden_units=encUknwD1OrigWidth,drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_r = AE_CL(input_shape=num_NonCat_features_r - mpfeatures, hidden_units_final=encUknwD2ReWidthFinal,
                                                  hidden_depth=encUknwD2ReDepth,
                                                  hidden_units=encUknwD2ReWidth,drop_out_p=dropout_rate_CL).to(device)

    criterion = nn.MSELoss()

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
            if (len(data[0][1]) == len(data[1][1])) & (data[0][1].shape[0]==batchSize):
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
                        # breakpoint()
                        # Replace selected x_bar features with the noisy ones
                        x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                        x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                        # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                        datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)

                # computing the encoded values
                known_rep_o = known_features_encoder(datalist[0])
                # print("passed known train encoder")
                known_rep_r = known_features_encoder(datalist[1])
                unknown_rep_o = unknown_features_encoder_orig(datalist[2])
                # print("passed unknown O train encoder")
                unknown_rep_r = unknown_features_encoder_r(datalist[3])
                # print("passed unknown R train  encoder")

                # breakpoint()
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

        # TODO validation los with the scheduler

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
                    if (len(data[0][1]) == len(data[1][1])) & (data[0][1].shape[0]==batchSize):
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
                                # breakpoint()
                                # Replace selected x_bar features with the noisy ones
                                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
                                x_bar1 = x_bar * (1 - mask1) + x_bar_noisy * mask1

                                # datalist[dt] = torch.concat((datalist[dt], x_bar), axis=0)
                                datalist[dt] = torch.concat((x_bar1, x_bar), axis=0)

                        # computing the encoded values
                        known_rep_o = known_features_encoder(datalist[0])
                        # print("passed known val encoder")
                        known_rep_r = known_features_encoder(datalist[1])
                        unknown_rep_o = unknown_features_encoder_orig(datalist[2])
                        # print("passed unknown O val encoder")
                        unknown_rep_r = unknown_features_encoder_r(datalist[3])
                        # print("passed unknown R val  encoder")

                        # breakpoint()
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

        # display the epoch training loss
        # print("epoch : {}/{}, total loss = {:.8f}".format(epoch + 1, epochs, loss_tr))
        # total_loss.append(loss)

    # breakpoint()
    if False:
        """ plotting the summary importance """

        background_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[:10, :mpfeatures].values),
                                         torch.Tensor(DF_holdout_r.iloc[:10, :mpfeatures].values)], dim=0).to(device)
        shap_to_use_known = torch.concat([torch.Tensor(Df_holdout_orig.iloc[10:100, :mpfeatures].values),
                                          torch.Tensor(DF_holdout_r.iloc[10:100, :mpfeatures].values)], dim=0).to(
            device)
        en_known = shap.DeepExplainer(known_features_encoder, background_known)
        shap_vals_knwn_comb = en_known.shap_values(shap_to_use_known)
        shap.summary_plot(shap_vals_knwn_comb, feature_names=[itemid_label_dict[int(i)] for i in mapped_features],
                          show=False, max_display=len(mapped_features))
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Known Encoder using both data sources ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".pdf")
        plt.close()

        en_unkn_orig = shap.DeepExplainer(unknown_features_encoder_orig,
                                          torch.Tensor(Df_holdout_orig.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_orig = en_unkn_orig.shap_values(
            torch.Tensor(Df_holdout_orig.iloc[10:100, mpfeatures:].values).to(device))
        shap.summary_plot(shap_vals_unkn_orig,
                          feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_orig], show=False,
                          max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder Original ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".pdf")
        plt.close()

        en_unkn_r = shap.DeepExplainer(unknown_features_encoder_r,
                                       torch.Tensor(DF_holdout_r.iloc[:10, mpfeatures:].values).to(device))
        shap_vals_unkn_r = en_unkn_r.shap_values(torch.Tensor(DF_holdout_r.iloc[10:100, mpfeatures:].values).to(device))
        shap.summary_plot(shap_vals_unkn_r, feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_r],
                          show=False, max_display=50)
        plt.legend().set_visible(False)
        plt.title(" Feature importance from Unknown Encoder R ")
        plt.tight_layout()
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".png")
        plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
            represFinal_dim) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
            epochs) + "_representation_dim_" + str(represFinal_dim) + "_np_nt_" + str(partition) + "_" + str(
            trial) + ".pdf")
        plt.close()
    # computing the gradient of the output wrt the input data

    for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    # breakpoint()
    for i in range(temp_input.shape[0]):
        grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]):
        grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

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


    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

    print(" \n Mistakes by the CL method on holdout data")
    print(" CL  X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" CL  X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)


    print(" -------- CL method training ends ------------- \n \n  ")

    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test.shape[0]):
        for j in range(x1_match_matrix_test.shape[1]):
            if x1_match_matrix_test[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])

    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))


    true_match_list = list(set(to_map_r).intersection(to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]

    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])

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

    # print(F1_fromx1)
    # exit()
    return grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test, correct_with_match_from_x2_test, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r, overall_quality_error_matching_only

def Train_CL_withDec(df_train_preproc, df_rename_preproc,
               reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig,
               DF_holdout_r, normalizing_values_orig, normalizing_values_r, P_x1):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_r) - 1

    num_features = len(reordered_column_names_r) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures

    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batchSize, shuffle=True, num_workers=1)
    dataset_orig_val = TabularDataset(data=Df_holdout_orig, output_col=outcome)
    val_loader_orig = DataLoader(dataset_orig_val, batchSize, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batchSize, shuffle=True, num_workers=1)
    dataset_r_val = TabularDataset(data=DF_holdout_r, output_col=outcome)
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
            # print(len(data[0][1]), len(data[1][1]), data[0][1].shape[0], batchSize)
            if (len(data[0][1]) == len(data[1][1])) & (data[0][1].shape[0] == batchSize):
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

                # breakpoint()
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

                # breakpoint()


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
                    # print(len(data[0][1]), len(data[1][1]), data[0][1].shape[0], batchSize)
                    # breakpoint()
                    if (len(data[0][1]) == len(data[1][1])) & (data[0][1].shape[0] == batchSize):
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

                        # print("Debug checking")
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
                        # print(data[0][1].shape[0], batch_size, counting_flag_for_rank_val)
                        if (data[0][1].shape[0] == batchSize):
                            rank_val_o += avg_Rank_o.item()
                            rank_val_r += avg_Rank_r.item()
                            # breakpoint()
                            counting_flag_for_rank_val = counting_flag_for_rank_val + 1

                loss_val = loss_val / (len(val_loader_orig) + len(val_loader_r))
                within_unkn_CL_loss_val = within_unkn_CL_loss_val / (len(val_loader_orig) + len(val_loader_r))

                # breakpoint()
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
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])[0]

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]): grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])[0]

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    # np.savetxt(output_dir+"Grad_unknwn_Orig_" + str(randomSeed) + "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_o.cpu().detach().numpy(), delimiter=",")
    # np.savetxt(output_dir+"Grad_unknwn_R_" + str(randomSeed) +  "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_r.cpu().detach().numpy(), delimiter=",")
    # exit()

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)


    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(
        o_to_r_sim,
        r_to_o_sim,
        P_x1, len(mapped_features))

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

    print(" \n Mistakes by the CL method on holdout data")
    print(" CL + Dec  X1_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)
    print(" CL + Dec  X2_train mistakes number on holdout set",
          unmapped_features_orig - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          unmapped_features_orig - num_xtra_feat_inX1)

    print(" -------- CL + Dec method training ends ------------- \n \n  ")

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


    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test.shape[0]):
        for j in range(x1_match_matrix_test.shape[1]):
            if x1_match_matrix_test[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])

    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))


    true_match_list = list(set(to_map_r).intersection(to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]
    # breakpoint()
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])

    incorrect_match_dict_x1 = {}

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
                incorrect_match_dict_x1[to_map_orig[i]] = to_map_r[j]
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
                FP_x1 = FP_x1 + 1
                incorrect_match_dict_x1[to_map_orig[i]] = to_map_r[j]


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

    incorrect_match_idx_orig_from_x1 = []
    for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(list(Df_holdout_orig.columns[mpfeatures:]).index(i))
    incorrect_match_idx_r_from_x1 = []
    for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(list(DF_holdout_r.columns[mpfeatures:]).index(i))

    # breakpoint()
    overall_quality_oracle_comb = rec_criterion(unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device))[0], 1)[:,incorrect_match_idx_r_from_x1],
                                                torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)[:,incorrect_match_idx_orig_from_x1] )



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
    return grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test, correct_with_match_from_x2_test, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r, overall_quality_error_matching_only, overall_quality_oracle_comb


def Train_KMF(df_train_preproc, df_rename_preproc,
                   P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures - 1
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures - 1
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
    # breakpoint()
    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test_KMF, x2_match_matrix_test_KMF = Matching_via_HRM(
        sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,P_x1,
        len(mapped_features))



    print(" \n Mistakes by the simple correlation method on holdout data")
    print(" Sim_Correlation  X1_train mistakes number", unmapped_features_orig-correct_with_match_from_x1_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number", unmapped_features_orig-correct_with_match_from_x2_test-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    print(" -------- KMF-l methods  ends ------------- \n \n  ")

    to_map_orig = reordered_column_names_orig[-(len(reordered_column_names_orig) - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(len(reordered_column_names_r) - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test_KMF.shape[0]):
        for j in range(x1_match_matrix_test_KMF.shape[1]):
            if x1_match_matrix_test_KMF[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])

    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))


    true_match_list = list(set(to_map_r).intersection(to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]
    # breakpoint()
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])

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

    return CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped, correct_with_match_from_x1_test, correct_with_match_from_x2_test, F1_fromx1, F1_fromx2, overall_quality_error_matching_only

def CL_with_KMF_linear(grad_sum_unkn_o, grad_sum_unkn_r, CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,P_x1
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features, Df_holdout_orig):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1
    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures - 1
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures - 1

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



    print(" \n Mistakes by the CL KMFl method on holdout data")
    print(" CL KMFl X1_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x1_te_CL_KMFl-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" CL KMFl X2_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x2_te_CL_KMFl-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)


    # MisF_X1_te_CL_KMFl = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te_CL_KMFl]
    # MisF_X2_te_CL_KMFl = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te_CL_KMFl]
    #
    # print(" CL KMFl X1_train mistakes", MisF_X1_te_CL_KMFl)
    # print(" CL KMFl X2_train mistakes", MisF_X2_te_CL_KMFl)
    #
    # print(" CL KMFl X1_train mistakes (len) ", len(MisF_X1_te_CL_KMFl), " out of ",
    #       num_NonCat_features_orig - mpfeatures)
    # print(" CL KMFl X2_train mistakes (len) ", len(MisF_X2_te_CL_KMFl), " out of ", num_NonCat_features_r - mpfeatures)

    print(" -------- CL + KMFl method training ends ------------- \n \n  ")

    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test_CL_KMFl.shape[0]):
        for j in range(x1_match_matrix_test_CL_KMFl.shape[1]):
            if x1_match_matrix_test_CL_KMFl[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])


    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))

    true_match_list = list(set(to_map_r).intersection(
        to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]
    # breakpoint()
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])


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

    return  correct_with_match_from_x1_te_CL_KMFl, correct_with_match_from_x2_te_CL_KMFl, F1_fromx1, F1_fromx2, overall_quality_error_matching_only


def CL_with_KMF_CLencoded(grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o, rep_known_val_r, Df_holdout_orig, DF_holdout_r,
                       P_x1
                       , reordered_column_names_orig, reordered_column_names_r,
                       mapped_features):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_r = len(reordered_column_names_r) - 1
    num_NonCat_features_orig = len(reordered_column_names_orig) - 1
    unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures - 1
    unmapped_features_r = len(reordered_column_names_r) - mpfeatures - 1

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

    # Mistakes_X1_te_CL_KMFen, Mistakes_X2_te_CL_KMFen, x1_match_matrix_test_CL_KMFen, x2_match_matrix_test_CL_KMFen = Stable_matching_algorithm(
    #     o_to_r_sim_CL_KMFen,
    #     r_to_o_sim_CL_KMFen,
    #     index_for_mapping_orig_to_rename[len(mapped_features):],
    #     index_for_mapping_rename_to_orig[len(mapped_features):],
    #     len(mapped_features))
    #
    # MisF_X1_te_CL_KMFen = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te_CL_KMFen]
    # MisF_X2_te_CL_KMFen = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te_CL_KMFen]
    #
    # print(" CL KMF-encoded X1_train mistakes", MisF_X1_te_CL_KMFen)
    # print(" CL KMF-encoded X2_train mistakes", MisF_X2_te_CL_KMFen)
    #
    # print(" CL KMF-encoded X1_train mistakes (len) ", len(MisF_X1_te_CL_KMFen), " out of ",
    #       num_NonCat_features_orig - mpfeatures)
    # print(" CL KMF-encoded X2_train mistakes (len) ", len(MisF_X2_te_CL_KMFen), " out of ",
    #       num_NonCat_features_r - mpfeatures)


    correct_with_match_from_x1_te_CL_KMFen, correct_with_match_from_x2_te_CL_KMFen, x1_match_matrix_test_CL_KMFen, x2_match_matrix_test_CL_KMFen = Matching_via_HRM(
        o_to_r_sim_CL_KMFen, r_to_o_sim_CL_KMFen,P_x1,
        len(mapped_features))



    print(" \n Mistakes by the CL KMF-encoded method on holdout data")
    print(" CL KMFl X1_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x1_te_CL_KMFen-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)
    print(" CL KMFl X2_train mistakes (len)", unmapped_features_orig-correct_with_match_from_x2_te_CL_KMFen-num_xtra_feat_inX1, "out of ", unmapped_features_orig-num_xtra_feat_inX1)

    print(" -------- CL + KMF-encoded method training ends ------------- \n \n  ")

    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test_CL_KMFen.shape[0]):
        for j in range(x1_match_matrix_test_CL_KMFen.shape[1]):
            if x1_match_matrix_test_CL_KMFen[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])

    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))

    true_match_list = list(set(to_map_r).intersection(
        to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]
    # breakpoint()
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])


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

    return correct_with_match_from_x1_te_CL_KMFen, correct_with_match_from_x2_te_CL_KMFen, F1_fromx1, F1_fromx2, overall_quality_error_matching_only

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
        self.encoder_hidden_layer1 = nn.Linear(in_features=self.no_of_cont, out_features=80)
        self.bn1 = nn.BatchNorm1d(num_features=80)
        self.encoder_hidden_layer2 = nn.Linear(in_features=80, out_features=40)
        self.bn2 = nn.BatchNorm1d(num_features=40)
        self.drop_layer2 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_output_layer = nn.Linear(in_features=40, out_features=self.output_rep_dim)

    def forward(self, data):
        # if data.shape[1] > 20:
        #     print("Debug")
        #     breakpoint()
        # breakpoint()
        data = self.drop_layer1(data)
        activation = self.encoder_hidden_layer1(data)
        if self.batchnorm == 1:
            activation = self.bn1(activation)
        activation = self.encoder_hidden_layer2(activation)
        if self.batchnorm == 1:
            activation = self.bn2(activation)
        activation = torch.tanh(activation)
        # activation = self.drop_layer2(activation)
        code0 = self.encoder_output_layer(activation)

        return code0

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


    # breakpoint()
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
        # breakpoint()
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


# MI based method that first creates a graph and then looks for a permutation matrix that minimizes the distance between the adjcency matrices of the two graphs
def Kang_MI_HC_opt_with_Euclidean_dist(df_train_preproc, df_rename_preproc, true_perm,
                    reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r, sq_transf_features, P_x1):
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
            for j in range(i + 1, num_NonCat_features_orig):
                MI_orig[i, j] = calc_MI(df_train_preproc.iloc[:, i], df_train_preproc.iloc[:, j], 20)
                MI_orig[j, i] = MI_orig[i, j]

        MI_r = np.zeros((num_NonCat_features_r, num_NonCat_features_r))
        # MI computation
        for i in range(num_NonCat_features_r):
            for j in range(i + 1, num_NonCat_features_r):
                MI_r[i, j] = calc_MI(df_rename_preproc.iloc[:, i], df_rename_preproc.iloc[:, j], 20)
                MI_r[j, i] = MI_r[i, j]

    num_iter = 1000
    # initial random permutation and corresponding distance computation
    initial_perm = np.array(list(np.arange(mpfeatures)) + list(np.random.choice( np.arange(unmapped_features_r), unmapped_features_orig, replace =False)+mpfeatures))

    D_M_normal_ini = 0

    for i in range(len(MI_orig)):
        for j in range(len(MI_orig)):
            # temp = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) / (
            #             MI_orig[i, j] + MI_r[initial_perm[i], initial_perm[j]]))
            temp = (MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]]) * (MI_orig[i, j] - MI_r[initial_perm[i], initial_perm[j]])
            if np.isnan(temp) != True :
                D_M_normal_ini = D_M_normal_ini + temp
                # print(temp)
    D_M_normal_ini = np.sqrt(D_M_normal_ini)
    print("Distance_initial_value", D_M_normal_ini)

    D_M_normal = D_M_normal_ini
    for iter in range(num_iter):
        temp_per = initial_perm.copy()
        # breakpoint()
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
                # temp0 = (1 - alpha * np.abs(MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]]) / (
                #         MI_orig[i, j] + MI_r[temp_per[i], temp_per[j]]))
                temp0 = (MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]])* (MI_orig[i, j] - MI_r[temp_per[i], temp_per[j]])
                if np.isnan(temp0) != True:
                    temp_dist_normal = temp_dist_normal + temp0
                    # print(temp0)
        temp_dist_normal = np.sqrt(temp_dist_normal)
        # updating the cost and the permutation vector
        if temp_dist_normal < D_M_normal:
            # print(" Iteration number where it changed", iter +1)
            # print("initial cost ", D_M_normal)
            # print("initial permutation", initial_perm)

            D_M_normal = temp_dist_normal
            initial_perm = temp_per

            # print(" Updated cost ", D_M_normal)
            # print(" updated permutation ", temp_per)

            print("\n ---------------------- \n")
            print(" Matching after Iteration number ", iter +1)
            print(dict(zip(list(reordered_column_names_orig[mpfeatures:-1]),
                           [reordered_column_names_r[i] for i in initial_perm[mpfeatures:]])))
            print("\n ---------------------- \n")

    mistake_indices = []
    correct_total_fromKANG = 0
    for i in range(mpfeatures, mpfeatures + unmapped_features_orig):
        # print(i)
        if true_perm[i]-1==initial_perm[i]:
            correct_total_fromKANG = correct_total_fromKANG + 1
        else:
            mistake_indices.append(true_perm[i])


    print(" \n Mistakes by the KANG method on holdout data")

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in mistake_indices]

    print(" KANG method  X1 mistakes", MisF_X1_te)

    print(" KANG method mistakes on sq transformation X1 ", list(set(sq_transf_features).intersection(MisF_X1_te)))

    print(" KANG  X1_train mistakes number", unmapped_features_orig-correct_total_fromKANG, "out of ", unmapped_features_orig)

    print(" -------- KANG  methods  ends ------------- \n \n  ")

    del df_rename_preproc

    x1_match_matrix_test = np.zeros(P_x1.shape)
    for i in range(x1_match_matrix_test.shape[0]):
        # print(i, x1_train_y[i]-1)
        x1_match_matrix_test[i, initial_perm[i]] = 1

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

    # exit()
    return MisF_X1_te, F1_fromx1

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

    to_map_orig = reordered_column_names_orig[-(num_NonCat_features_orig - mpfeatures + 1):-1]
    to_map_r = reordered_column_names_r[-(num_NonCat_features_r - mpfeatures + 1):-1]

    predmapped_orig_r = []
    for i in range(x1_match_matrix_test_updated.shape[0]):
        for j in range(x1_match_matrix_test_updated.shape[1]):
            if x1_match_matrix_test_updated[i, j] == 1:
                predmapped_orig_r.append(to_map_r[j])
    # breakpoint()
    predicted_match_dic_x1 = dict(zip(to_map_orig, predmapped_orig_r))


    true_match_list = list(set(to_map_r).intersection(to_map_orig))  # this is different here because there are some features that do not match in other dataset
    match_dict = dict(zip(true_match_list, true_match_list))

    # breakpoint()
    final_dic_for_compar_matching = {}
    for key, val in match_dict.items():
        if val in predicted_match_dic_x1.values():
            final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[list(predicted_match_dic_x1.values()).index(val)]
    # breakpoint()
    overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,Df_holdout_orig[final_dic_for_compar_matching.values()])

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

    return correct_total_fromKANG, F1_fromx1, overall_quality_error_matching_only

def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc,index_for_mapping_orig_to_rename,
                   index_for_mapping_rename_to_orig
                   , reordered_column_names_orig, reordered_column_names_r,
                   mapped_features,Cor_from_df,Df_holdout_orig, DF_holdout_r,sq_transf_features,P_x1):
    mpfeatures = len(mapped_features)
    unmapped_features_orig = len(reordered_column_names_orig)-mpfeatures -1
    unmapped_features_r = len(reordered_column_names_r)-mpfeatures -1

    device = torch.device('cpu')
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

    Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,
                                                               index_for_mapping_orig_to_rename[
                                                               len(mapped_features):],
                                                               index_for_mapping_rename_to_orig[
                                                               len(mapped_features):],
                                                               len(mapped_features))


    test_statistic_num_fromX1 = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for
                                     j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    test_statistic_num_fromX2 = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for
                                     j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[
                                         i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small


    # Bootstrap samples to obtain the standard deviation methods to be later used in p value computation
    num_of_bts = 1
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
        # sim_cor_X1_to_X2_bts = np.matmul(CorMatrix_X1_unmap_mapped_bts, np.transpose(CorMatrix_X2_unmap_mapped_bts))
        # sim_cor_X2_to_X1_bts = np.matmul(CorMatrix_X2_unmap_mapped_bts, np.transpose(CorMatrix_X1_unmap_mapped_bts))

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

    # # correlation values of the accepted matches
    # print(" correlation values of the accepted matches from CC x1 (Holdout_sample)")
    # print(CC_values_for_testing_from_x1_test)
    # print(" correlation values of the accepted matches from CC x2 (Holdout_sample)")
    # print(CC_values_for_testing_from_x2_test)

    # testing whether some of the proposed matches are significant or not;
    # False in the reject list below can be interpreted as the case where the  testing procedure says this match is not significant
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
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]


    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i], reordered_column_names_r[len(mapped_features) + matched_index[0]]]


    # dropping non significant matched feature pairs
    num_insign_dropp_x1 = 0
    num_insign_dropp_x2 = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x1 ", temp_inf_x1.ump_feature_in_X1[i], temp_inf_x1.match_byGS[i], temp_inf_x1.estimated_cross_corr[i])
            temp_inf_x1.drop([i], inplace=True)
            num_insign_dropp_x1 = num_insign_dropp_x1 +1

        if temp_inf_x2.SD_rejects_H0[i] == False:
            # print("Feature pair to be dropped because of non-significance from x2 ", temp_inf_x2.ump_feature_in_X1[i], temp_inf_x2.match_byGS[i], temp_inf_x2.estimated_cross_corr[i])
            temp_inf_x2.drop([i], inplace=True)
            num_insign_dropp_x2 = num_insign_dropp_x2 + 1

    print(" Number of insignificant feature pair drops from x1 ", num_insign_dropp_x1)
    print(" Number of insignificant feature pair drops from x2 ", num_insign_dropp_x2)

    # ordering the
    temp_inf_x1 = temp_inf_x1.sort_values(by='estimated_cross_corr', ascending=False)
    temp_inf_x2 = temp_inf_x2.sort_values(by='estimated_cross_corr', ascending=False)

    num_additional_mapped_for_next_stage_x1 = int(len(temp_inf_x1)/2)
    num_additional_mapped_for_next_stage_x2 = int(len(temp_inf_x2)/2)

    # taking the intersection of the additional mapped features
    temp_x1_x1 = [temp_inf_x1.ump_feature_in_X1[i] for i in list(temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x1_match = [temp_inf_x1.match_byGS[i] for i in list(temp_inf_x1.index)[:num_additional_mapped_for_next_stage_x1]]
    temp_x2_x1 = [temp_inf_x2.ump_feature_in_X1[i] for i in list(temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]
    temp_x2_match = [temp_inf_x2.match_byGS[i] for i in list(temp_inf_x2.index)[:num_additional_mapped_for_next_stage_x2]]



    final_additional_mapped = list(set(temp_x1_x1).intersection(temp_x2_x1))
    final_additional_mapped_corr_match =[]
    for i in final_additional_mapped:
        final_additional_mapped_corr_match.append(temp_x1_match[temp_x1_x1.index(i)])
    # idx_final = temp_inf_x1.index(final_additional_mapped)
    # final_additional_mapped_corr_match = temp_inf_x2[idx_final]

    print(" \n Mistakes by the simple correlation method on holdout data")

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

    print(" KMF  X1 mistakes", MisF_X1_te)
    print(" KMF  X2 mistakes", MisF_X2_te)

    print(" KMF mistakes on sq transformation X1 ", list(set(sq_transf_features).intersection(MisF_X1_te)))
    print(" KMF mistakes on sq transformation X2 ", list(set(sq_transf_features).intersection(MisF_X2_te)))

    print(" Sim_Correlation  X1_train mistakes number", len(Mistakes_X1_te), "out of ", unmapped_features_orig)
    print(" Sim_Correlation  X2_train mistakes number", len(Mistakes_X2_te), "out of ", unmapped_features_r)


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

    return Mistakes_X1_te, Mistakes_X2_te, temp_inf_x1, temp_inf_x2, final_additional_mapped, final_additional_mapped_corr_match, F1_fromx1, F1_fromx2


def RadialGAN(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features, partition_no, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r,sq_transf_features,P_x1):
    device = torch.device('cuda')
    num_features = len(reordered_column_names_r) -1

    print(" -------- RadialGAN method training starts ------------- \n \n  ")


    """ ==================== GRADIENT PENALTY ======================== """

    def calc_gradient_penalty(real_data, fake_data, D_num):
        alpha0 = torch.rand(real_data.size()[0], 1).to(device)
        alpha0 = alpha0.expand(real_data.size())

        interpolates = alpha0 * real_data + ((1 - alpha0) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        if D_num == 1:
            disc_interpolates = D1(interpolates)
        if D_num == 2:
            disc_interpolates = D2(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device = 'cuda'),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * beta

        # print(gradient_penalty)
        return gradient_penalty

    """ ==================== Setting up the disciminators ======================== """

    Wxh_o = weights_init(size=[num_features, num_features])
    bxh_o = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3y_o = weights_init(size=[num_features, 1])
    bh3y_o = Variable(torch.zeros(1, device = "cuda"), requires_grad=True)

    Wxh_r = weights_init(size=[num_features, num_features])
    bxh_r = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3y_r = weights_init(size=[num_features, 1])
    bh3y_r = Variable(torch.zeros(1, device = "cuda"), requires_grad=True)

    def D1(X):
        h = torch.tanh(X.mm(Wxh_o) + bxh_o.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o) + bhh2_o.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o) + bh2h3_o.repeat(h_2.size(0), 1))
        y = torch.sigmoid(h_3.mm(Wh3y_o) + bh3y_o.repeat(h_3.size(0), 1))
        # y = F.linear(h_3, Wh3y_o.t(), bh3y_o.repeat(h_3.size(0), 1))
        return y  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def D2(X):
        h = torch.tanh(X.mm(Wxh_r) + bxh_r.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_r) + bhh2_r.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_r) + bh2h3_r.repeat(h_2.size(0), 1))
        y = torch.sigmoid(h_3.mm(Wh3y_r) + bh3y_r.repeat(h_3.size(0), 1))
        # y = F.linear(h_3, Wh3y_r.t(), bh3y_r.repeat(h_3.size(0), 1))
        return y  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    D1_params = [Wxh_o, bxh_o, Whh2_o, bhh2_o, Wh2h3_o, bh2h3_o, Wh3y_o, bh3y_o]
    D2_params = [Wxh_r, bxh_r, Whh2_r, bhh2_r, Wh2h3_r, bh2h3_r, Wh3y_r, bh3y_r]

    """ ==================== Setting up the encoders F ======================== """

    Wxh_o_F = weights_init(size=[num_features, num_features])
    bxh_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o_F = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o_F = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o_F = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3z_o_F = weights_init(size=[num_features, num_features])
    bh3z_o_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wxh_r_F = weights_init(size=[num_features, num_features])
    bxh_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r_F = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r_F = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r_F = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3z_r_F = weights_init(size=[num_features, num_features])
    bh3z_r_F = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    def F1(X):
        h = torch.tanh(X.mm(Wxh_o_F) + bxh_o_F.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o_F) + bhh2_o_F.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o_F) + bh2h3_o_F.repeat(h_2.size(0), 1))
        z = torch.sigmoid(h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1))
        # z = h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1)
        return z  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def F2(X):
        h = torch.tanh(X.mm(Wxh_o_F) + bxh_o_F.repeat(X.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o_F) + bhh2_o_F.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o_F) + bh2h3_o_F.repeat(h_2.size(0), 1))
        z = torch.sigmoid(h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1))
        # z = h_3.mm(Wh3z_o_F) + bh3z_o_F.repeat(h_3.size(0), 1)
        return z  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    F1_params = [Wxh_o_F, bxh_o_F, Whh2_o_F, bhh2_o_F, Wh2h3_o_F, bh2h3_o_F, Wh3z_o_F, bh3z_o_F]
    F2_params = [Wxh_r_F, bxh_r_F, Whh2_r_F, bhh2_r_F, Wh2h3_r_F, bh2h3_r_F, Wh3z_r_F, bh3z_r_F]

    """ ==================== Setting up the decoders G ======================== """

    Wzh_o_G = weights_init(size=[num_features, num_features])
    bzh_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_o_G = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_o_G = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_o_G = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3x_o_G = weights_init(size=[num_features, num_features])
    bh3x_o_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wzh_r_G = weights_init(size=[num_features, num_features])
    bzh_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Whh2_r_G = weights_init(size=[num_features, int(num_features / 2)])
    bhh2_r_G = Variable(torch.zeros(int(num_features / 2), device = "cuda"), requires_grad=True)

    Wh2h3_r_G = weights_init(size=[int(num_features / 2), num_features])
    bh2h3_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    Wh3x_r_G = weights_init(size=[num_features, num_features])
    bh3x_r_G = Variable(torch.zeros(num_features, device = "cuda"), requires_grad=True)

    def G1(Z):
        h = torch.tanh(Z.mm(Wzh_o_G) + bzh_o_G.repeat(Z.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_o_G) + bhh2_o_G.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_o_G) + bh2h3_o_G.repeat(h_2.size(0), 1))
        # x = F.sigmoid(h_3.mm(Wh3x_o_G) + bh3x_o_G.repeat(h_3.size(0), 1))
        x = h_3.mm(Wh3x_o_G) + bh3x_o_G.repeat(h_3.size(0), 1)
        return x  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    def G2(Z):
        h = torch.tanh(Z.mm(Wzh_r_G) + bzh_r_G.repeat(Z.size(0), 1))
        h_2 = torch.tanh(h.mm(Whh2_r_G) + bhh2_r_G.repeat(h.size(0), 1))
        h_3 = torch.tanh(h_2.mm(Wh2h3_r_G) + bh2h3_r_G.repeat(h_2.size(0), 1))
        # x = F.sigmoid(h_3.mm(Wh3x_r_G) + bh3x_r_G.repeat(h_3.size(0), 1))
        x = h_3.mm(Wh3x_r_G) + bh3x_r_G.repeat(h_3.size(0), 1)
        return x  # *** y will be a vector of 64 points; in case of WGAN need not be probabilities

    G1_params = [Wzh_o_G, bzh_o_G, Whh2_o_G, bhh2_o_G, Wh2h3_o_G, bh2h3_o_G, Wh3x_o_G, bh3x_o_G]
    G2_params = [Wzh_r_G, bzh_r_G, Whh2_r_G, bhh2_r_G, Wh2h3_r_G, bh2h3_r_G, Wh3x_r_G, bh3x_r_G]

    params = D1_params + D2_params + F1_params + F2_params + G1_params + G2_params

    """ ===================== TRAINING STARTS ======================== """

    def reset_grad():
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()).to(device)

    # some data formatting for getting the minibatches
    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batch_size, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batch_size, shuffle=True, num_workers=1)

    # create optimizer objects
    F1_solver = optim.Adam(F1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    G1_solver = optim.Adam(G1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    D1_solver = optim.Adam(D1_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)

    F2_solver = optim.Adam(F2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    G2_solver = optim.Adam(G2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    D2_solver = optim.Adam(D2_params, lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-5)

    # lr scheduler
    scheduler_F1 = torch.optim.lr_scheduler.ReduceLROnPlateau(F1_solver, patience=2, verbose=True)
    scheduler_G1 = torch.optim.lr_scheduler.ReduceLROnPlateau(G1_solver, patience=2, verbose=True)
    scheduler_D1 = torch.optim.lr_scheduler.ReduceLROnPlateau(D1_solver, patience=2, verbose=True)

    scheduler_F2 = torch.optim.lr_scheduler.ReduceLROnPlateau(F2_solver, patience=2, verbose=True)
    scheduler_G2 = torch.optim.lr_scheduler.ReduceLROnPlateau(G2_solver, patience=2, verbose=True)
    scheduler_D2 = torch.optim.lr_scheduler.ReduceLROnPlateau(D2_solver, patience=2, verbose=True)

    total_loss = []
    dis_o_loss = []
    dis_r_loss = []
    G_o_loss = []
    G_r_loss = []
    CYC_o_loss_dir = []
    CYC_r_loss_dir = []
    CYC_o_loss_cross = []
    CYC_r_loss_cross = []

    for epoch in range(epochs_RadialGAN):
        loss = 0
        D_o_loss = 0
        D_r_loss = 0
        gen_o_loss = 0
        gen_r_loss = 0
        cyc_o_loss_dir = 0
        cyc_r_loss_dir = 0
        cyc_o_loss_cross = 0
        cyc_r_loss_cross = 0

        D1_within_epoch_loss = []
        D2_within_epoch_loss = []

        gen1_within_epoch_loss = []
        gen2_within_epoch_loss = []

        cyc1_within_epoch_loss = []
        cyc2_within_epoch_loss = []

        # cyc1_within_epoch_loss_cross = []
        # cyc2_within_epoch_loss_cross = []

        counter_num_mb_for_dis = 0
        for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
            if len(data[0][1]) == len(data[1][1]):
                x_o = data[0][1].to(device)
                x_r = data[1][1].to(device)

                if i == 0 or i % 5 != 0:
                    """  training D1  """
                    D1_real = D1(x_o)
                    D1_fake = D1(G1(F2(x_r)))

                    # train with gradient penalty
                    gradient_penalty1 = calc_gradient_penalty(x_o, G1(F2(x_r)), 1)
                    gradient_penalty1.backward()

                    D1_loss0 = -(torch.mean(D1_real) - torch.mean(D1_fake))
                    D1_loss0.backward()

                    D1_loss = D1_loss0 + gradient_penalty1
                    D1_solver.step()

                    D_o_loss = D_o_loss + D1_loss.item()

                    """  training D2  """
                    D2_real = D2(x_r)
                    D2_fake = D2(G2(F1(x_o)))

                    # train with gradient penalty
                    gradient_penalty2 = calc_gradient_penalty(x_r, G2(F1(x_o)), 2)
                    gradient_penalty2.backward()

                    D2_loss0 = -(torch.mean(D2_real) - torch.mean(D2_fake))
                    D2_loss0.backward()

                    D2_loss = D2_loss0 + gradient_penalty2
                    D2_solver.step()

                    D_r_loss = D_r_loss + D2_loss.item()

                    counter_num_mb_for_dis += 1
                    D1_within_epoch_loss.append(D1_loss.item())
                    D2_within_epoch_loss.append(D2_loss.item())
                    gen1_within_epoch_loss.append(0)
                    gen2_within_epoch_loss.append(0)
                    cyc1_within_epoch_loss.append(0)
                    cyc2_within_epoch_loss.append(0)

                    # Housekeeping - reset gradient
                    reset_grad()

                if i != 0 and i % 5 == 0:
                    D1_fake = D1(G1(F2(x_r)))
                    D2_fake = D2(G2(F1(x_o)))

                    gen1_loss = - torch.mean(D1_fake)
                    gen2_loss = - torch.mean(D2_fake)

                    # cycle consistency loss
                    cyc_loss1 = torch.norm(x_o - G1(F1(x_o))) + torch.norm(F2(x_r) - F1(G1(F2(x_r))))
                    cyc_loss2 = torch.norm(x_r - G2(F2(x_r))) + torch.norm(F1(x_o) - F2(G2(F1(x_o))))

                    # tr_loss = gen1_loss + gen2_loss + Lambda* (cyc_loss1 + cyc_loss2)
                    tr_loss = gen1_loss + gen2_loss + Lambda_dir * (
                                torch.norm(x_o - G1(F1(x_o))) + torch.norm(x_r - G2(F2(x_r)))) + Lambda_on_z * (
                                          torch.norm(F2(x_r) - F1(G1(F2(x_r)))) + torch.norm(F2(x_r) - F1(G1(F2(x_r)))))

                    tr_loss.backward()

                    # updating the parameters
                    G1_solver.step()
                    G2_solver.step()
                    F1_solver.step()
                    F2_solver.step()

                    loss = loss + tr_loss.item()
                    gen_o_loss = gen_o_loss + gen1_loss.item()
                    gen_r_loss = gen_r_loss + gen2_loss.item()
                    cyc_o_loss_dir = cyc_o_loss_dir + torch.norm(x_o - G1(F1(x_o))).item()
                    cyc_r_loss_dir = cyc_r_loss_dir + torch.norm(x_r - G2(F2(x_r))).item()
                    cyc_o_loss_cross = cyc_o_loss_cross + torch.norm(F2(x_r) - F1(G1(F2(x_r)))).item()
                    cyc_r_loss_cross = cyc_r_loss_cross + torch.norm(F1(x_o) - F2(G2(F1(x_o)))).item()

                    D1_within_epoch_loss.append(0)
                    D2_within_epoch_loss.append(0)
                    gen1_within_epoch_loss.append(gen1_loss.item())
                    gen2_within_epoch_loss.append(gen2_loss.item())
                    cyc1_within_epoch_loss.append(cyc_loss1.item())
                    cyc2_within_epoch_loss.append(cyc_loss2.item())

                    # Housekeeping - reset gradient
                    reset_grad()

        # compute the epoch training loss
        loss = loss / (5 * (len(train_loader_orig) + len(train_loader_r)))
        D_o_loss = D_o_loss / (5 * len(train_loader_orig))
        D_r_loss = D_r_loss / (5 * len(train_loader_r))
        gen_o_loss = gen_o_loss / (5 * len(train_loader_orig))
        gen_r_loss = gen_r_loss / (5 * len(train_loader_r))
        cyc_o_loss_dir = cyc_o_loss_dir / (5 * len(train_loader_orig))
        cyc_r_loss_dir = cyc_r_loss_dir / (5 * len(train_loader_r))
        cyc_o_loss_cross = cyc_o_loss_cross / (5 * len(train_loader_orig))
        cyc_r_loss_cross = cyc_r_loss_cross / (5 * len(train_loader_r))

        scheduler_D1.step(D_o_loss)
        scheduler_D2.step(D_r_loss)

        # display the epoch training loss
        print("epoch : {}/{}, total loss = {:.8f}".format(epoch + 1, epochs_RadialGAN, loss))
        print("epoch : {}/{}, cyc orig= {:.8f}".format(epoch + 1, epochs_RadialGAN, cyc_o_loss_dir + cyc_o_loss_cross))
        print("epoch : {}/{}, cyc r= {:.8f}".format(epoch + 1, epochs_RadialGAN, cyc_r_loss_cross + cyc_r_loss_dir))
        print("epoch : {}/{}, gen orig = {:.8f}".format(epoch + 1, epochs_RadialGAN, gen_o_loss))
        print("epoch : {}/{}, gen r = {:.8f}".format(epoch + 1, epochs_RadialGAN, gen_r_loss))
        print("epoch : {}/{}, critic loss ae orig= {:.8f}".format(epoch + 1, epochs_RadialGAN, D_o_loss))
        print("epoch : {}/{}, critic loss ae r= {:.8f}".format(epoch + 1, epochs_RadialGAN, D_r_loss))

        total_loss.append(loss)
        G_o_loss.append(gen_o_loss)
        G_r_loss.append(gen_r_loss)
        CYC_o_loss_dir.append(cyc_o_loss_dir)
        CYC_r_loss_dir.append(cyc_r_loss_dir)
        CYC_o_loss_cross.append(cyc_o_loss_cross)
        CYC_r_loss_cross.append(cyc_r_loss_cross)
        dis_o_loss.append(D_o_loss)
        dis_r_loss.append(D_r_loss)

    # saving_dir = './AE_reconstruction_stuff/Permutation_Datafrom_2021-04-16/Avg_all_random/Frac_Shuffled_' + str(
    #     frac_renamed) + '/RadialGAN_#ofhidden_layers_' + str(
    #     num_of_hidden_layers) + '/Syn_' + str(dataset_number)
    #
    # if not os.path.exists(saving_dir):
    #     os.makedirs(saving_dir)
    #
    # plt.plot(np.array(dis_o_loss), '-o', color='black', label='Dis_orig')
    # plt.plot(np.array(dis_r_loss), '-o', color='red', label='Dis_r')
    # plt.plot(np.array(G_o_loss), '-o', color='blue', label='Gen_orig')
    # plt.plot(np.array(G_r_loss), '-o', color='green', label='Gen_r')
    # plt.plot(np.array(CYC_o_loss_dir), '-o', color='magenta', label='Cyc_orig dir')
    # plt.plot(np.array(CYC_r_loss_dir), '-o', color='yellow', label='Cyc_r_dir')
    # plt.plot(np.array(CYC_o_loss_cross), '-o', color='brown', label='Cyc_orig_on_z')
    # plt.plot(np.array(CYC_r_loss_cross), '-o', color='orange', label='Cyc_r_on_z')
    # plt.title("Training loss over the epochs " + " #epochs " + str(epochs))
    # plt.xlabel("epochs")
    # plt.ylabel("loss value")
    # plt.legend()
    # plt.savefig(saving_dir + "/Training_loss_Across_" + str(epochs) + "epochs_perm#_" + str(partition_no) + ".png",
    #             bbbox='tight')
    # plt.close()
    print("Combined AE loss for partition ", partition_no, " is ", loss)

    """ ===================== TRAINING ENDS ======================== """

    # preparing to get the generated/reconstructed values

    features_true_orig = Df_holdout_orig.iloc[:, :-1].values
    features_true_renamed = DF_holdout_r.iloc[:, :-1].values

    gen_features_orig = G1(F2(torch.Tensor(DF_holdout_r.iloc[:, :-1].values).to(device))).cpu().detach().numpy()
    gen_features_r = G2(F1(torch.Tensor(Df_holdout_orig.iloc[:, :-1].values).to(device))).cpu().detach().numpy()

    # computing the correlation matrix between original feature values and cross reconstruction
    CorMatrix_X1_X1_hat_cross = np.zeros((num_features, num_features))
    CorMatrix_X2_X2_hat_cross = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            CorMatrix_X1_X1_hat_cross[i, j] = \
                stats.pearsonr(features_true_orig[:, i], gen_features_r[:, j])[0]
            CorMatrix_X2_X2_hat_cross[i, j] = \
                stats.pearsonr(features_true_renamed[:, i], gen_features_orig[:, j])[0]

    # selecting the correlation only for unmapped variables
    short_CorMatrix_X1_X1_hat_cross = CorMatrix_X1_X1_hat_cross[len(mapped_features):,
                                      len(mapped_features):]
    short_CorMatrix_X2_X2_hat_cross = CorMatrix_X2_X2_hat_cross[len(mapped_features):,
                                      len(mapped_features):]


    """ Calling the stable marriage algorithm for mappings  """

    Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(short_CorMatrix_X1_X1_hat_cross,
                                                                               short_CorMatrix_X2_X2_hat_cross,
                                                                               index_for_mapping_orig_to_rename[
                                                                               len(mapped_features):],
                                                                               index_for_mapping_rename_to_orig[
                                                                               len(mapped_features):],
                                                                               len(mapped_features))
    mpfeatures = len(mapped_features)
    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures,"\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

    print(" RadialGAN  X1_train mistakes", MisF_X1_te)
    print(" RadialGAN  X2_train mistakes", MisF_X2_te)

    print(" sq transformed features passed in mapped X1", list(set(sq_transf_features).intersection(mapped_features)) )
    print(" sq transformed features passed in mapped X2", list(set(sq_transf_features).intersection(mapped_features)) )

    print(" RadialGAN mistakes on sq transformation X1 ", list(set(sq_transf_features).intersection(MisF_X1_te)))
    print(" RadialGAN mistakes on sq transformation X2 ", list(set(sq_transf_features).intersection(MisF_X2_te)))

    print(" RadialGAN  X1_train mistakes number on holdout set", len(MisF_X1_te), "out of ", num_features - len(mapped_features))
    print(" RadialGAN  X2_train mistakes number on holdout set", len(MisF_X2_te), "out of ", num_features - len(mapped_features))

    print(" -------- RadialGAN method training ends ------------- \n \n  ")

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

    return MisF_X1_te, MisF_X2_te, F1_fromx1, F1_fromx2

def main(dataset_no_sample):

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(6)
    print("Number of threads being used are ", torch.get_num_threads())

    random.seed(100)
    np.random.seed(100)  # fixing the seed for reproducibility

    if dataset_number not in ['Nomao', 'superconductor']:
        # reading the data
        if dataset_number in ['1', '2']:
            filename = data_dir + "SD_" + str(dataset_number) + "/2021-05-18Syn_Data_" + str(
                dataset_number) + "_Sample_no_" + str(
                dataset_no_sample) + "_size_20_10000_for_AE_balanced.csv"  # for dataset 1 and 2
        else:
            filename = data_dir + "SD_" + str(dataset_number) + "/2021-05-21Syn_Data_" + str(
                dataset_number) + "_Sample_no_" + str(
                dataset_no_sample) + "_size_50_10000_for_AE_balanced.csv"  # for dataset 5
        full_Data0 = pd.read_csv(filename)

    elif dataset_number == 'superconductor':
        filename = data_dir + "superconduct/train.csv"
        full_Data0 = pd.read_csv(filename)

        # correlation between the features and the outcome
        # Cor_from_df = full_Data.corr()
        # simple_cor_plot = sns.heatmap(Cor_from_df, cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
        # simple_cor_plot.set_title("Superconductor dataset feature and outcome correlations")
        # fig = simple_cor_plot.get_figure()
        # fig.savefig("Superconductor_Correlogram.pdf", bbox='tight')
        # fig.savefig("Superconductor_Correlogram.png", bbox='tight')
        # plt.close()
        # binarizing the  outcome as the code right now is setup that way. Ultimately it does not matter as the outcome is not being used in any case.
        # the threshold 20 was chosen after seeing the distribution of feature 'critical_temp'
        full_Data0.loc[full_Data0['critical_temp'] < 20, outcome] = 0
        full_Data0.loc[full_Data0['critical_temp'] >= 20, outcome] = 1
        full_Data0.drop(columns=['critical_temp'], inplace=True)

        new_feature_names = ['Col' + str(i + 1) for i in range(full_Data0.shape[1] - 1)] + ['Y']
        full_Data0 = full_Data0.set_axis(new_feature_names, axis=1)  # renamed the continuous columns
    else:  # to be used for Nomao dataset
        filename = data_dir + "Nomao/Nomao.data"
        full_Data0 = pd.read_csv(filename, header=None)
        full_Data0.replace('?', np.nan, inplace=True)
        full_Data0 = full_Data0.loc[:, ~(full_Data0 == 'n').any()]  # dropping the nominal type columns
        full_Data0.drop(columns=[0], inplace=True)  # dropping the id column

        # drop columns with very high missing percentage
        percent_missing = full_Data0.isnull().sum() * 100 / len(full_Data0)
        missing_value_df = pd.DataFrame({'column_name': full_Data0.columns,
                                         'percent_missing': percent_missing}).sort_values(by='percent_missing')
        full_Data0 = full_Data0[missing_value_df[missing_value_df['percent_missing'] < 70.0]['column_name']]

        ## final columns that were selected including the label and excluding the first column that was the id
        ## [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69,
        # 70, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 89, 90, 91, 97, 98, 99, 101, 102, 103, 105, 106, 107, 109,
        # 110, 111, 113, 114, 115, 117, 118, 119]

        full_Data0 = full_Data0.reindex(columns=sorted(full_Data0.columns))  # reindexing so that label is at the end
        new_feature_names = ['Col' + str(i + 1) for i in range(full_Data0.shape[1] - 1)] + ['Y']
        full_Data0 = full_Data0.set_axis(new_feature_names, axis=1)  # renamed the continuous columns

        # converting the columns that are being treated as object type but actually are float
        for a in full_Data0.columns:
            if full_Data0[a].dtype == 'object':
                full_Data0[a] = full_Data0[a].astype('float')

        full_Data0 = full_Data0.fillna(full_Data0.mean())  # filling the missing values

    full_Data = full_Data0.copy()
    full_Data['Y'] = np.where(full_Data['Y'] == -1, 0, 1)  # to make it compatible with xgbt during evaluation

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

    AVG_MISMATCHES_X1_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))


    AVG_MISMATCHES_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    Frac_mismatches_across_trial_perm_X2_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_MISMATCHES_X1_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_MISMATCHES_X2_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    Frac_mismatches_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_with_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_CL_KMF_enc = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    F1_across_trial_perm_X2_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))

    AVG_F1_X1_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))
    AVG_F1_X2_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t))

    F1_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))


    # defining the match mse across trials; storing only for X1 side as that is what we will be using ultimately
    MatchMSE_across_trial_perm_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_CL_KMF_en = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_sim_Cor = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_Kang = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    MatchMSE_across_trial_perm_X1_tr_RG = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))


    ReconMSE_across_trial_perm_X1_tr = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))
    ReconMSE_across_trial_perm_X1_tr_CL_Dec = np.zeros((len(list_of_total_feat_in_D2_minus_mapped), n_t * n_p))


    # the set of mapped features is selected apriori and that same set is used across different sample sizes to avoid variation due to the mapped features
    mapped_random = []
    for i in range(n_t):
        mapped_random.append(np.random.choice(num_features, mpfeatures, replace=False))

    mapped_random = np.array(mapped_random)

    # breakpoint()
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

    no_match_inference_df_from_x1 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr'])
    no_match_inference_df_from_x2 = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr'])

    no_match_inference_df_from_x1_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0'])
    no_match_inference_df_from_x2_Sim_cor = pd.DataFrame(columns=['ump_feature_in_X1', 'match_byGS', 'true_correlation', 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0'])

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
            Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG = np.zeros(n_p)
            Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG = np.zeros(n_p)

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
            F1_for_fixed_trial_fixed_num_mapped_X1_tr_RG = np.zeros(n_p)
            F1_for_fixed_trial_fixed_num_mapped_X2_tr_RG = np.zeros(n_p)

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

            # device = torch.device('cpu')

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


            # if datatype == 'b':  # """ thresholding all feature values at 0 to binarize the data  """
            #     for i in list(df_train_preproc0.columns):
            #         df_train_preproc0.loc[df_train_preproc0[i] > 0, i] = 1
            #         df_train_preproc0.loc[df_train_preproc0[i] < 0, i] = 0
            #         Df_holdout_orig0.loc[Df_holdout_orig0[i] > 0, i] = 1
            #         Df_holdout_orig0.loc[Df_holdout_orig0[i] < 0, i] = 0
            #
            #     for i in list(df_rename_preproc0.columns):
            #         df_rename_preproc0.loc[df_rename_preproc0[i] > 0, i] = 1
            #         df_rename_preproc0.loc[df_rename_preproc0[i] < 0, i] = 0
            #         DF_holdout_r0.loc[DF_holdout_r0[i] > 0, i] = 1
            #         DF_holdout_r0.loc[DF_holdout_r0[i] < 0, i] = 0


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


                # reordering the features (PERMUTATION)
                reorder_feat = unmapped_features_reshuffle.copy()
                random.shuffle(reorder_feat)
                # index_for_mapping_orig_to_rename = [reorder_feat.index(num) + len(mapped_features) + 1 for num
                #                                     in
                #                                     [col for col in df_train_preproc.columns if
                #                                      col not in mapped_features + [outcome]]]
                # index_for_mapping_rename_to_orig = [[col for col in df_train_preproc.columns if
                #                                      col not in mapped_features + [outcome]].index(num) + len(
                #     mapped_features) + 1 for num in reorder_feat]
                #
                # # adding index variables for the mapped variables at the start of the list
                # index_for_mapping_orig_to_rename = list(
                #     np.arange(1, mpfeatures + 1)) + index_for_mapping_orig_to_rename
                # index_for_mapping_rename_to_orig = list(
                #     np.arange(1, mpfeatures + 1)) + index_for_mapping_rename_to_orig
                # print(" Index for mapping orig to rename ", index_for_mapping_orig_to_rename)
                # print(" Index for mapping rename to original ", index_for_mapping_rename_to_orig)

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
                # breakpoint()
                """ AE part preprocessing  starts   """

                correct_with_match_from_x1_test_Kang,F1_x1_kang, match_mse_kang = Kang_MI_HC_opt(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[:-1, :-1],
                    df_train_preproc.columns, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r)
                # breakpoint()
                # Mistakes_X1_tr_sim_Cor, Mistakes_X2_tr_sim_Cor, match_details_x1_sim_cor, match_details_x2_sim_cor, mapp_fea_to_add, mapp_fea_to_add_match, F1_x1_sim_cor, F1_x2_sim_cor  = Simple_maximum_sim_viaCorrelation(df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,Df_holdout_orig,
                #     DF_holdout_r, sq_transf_features,P_x1[len(mapped_features):-1,
                #                                                        len(mapped_features):-1])

                Cor_X1_map_unmap, Cor_X2_map_unmap, correct_with_match_from_x1_test_sim_cor, correct_with_match_from_x2_test_sim_cor, F1_x1_sim_cor, F1_x2_sim_cor,match_mse_kmf = Train_KMF(df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):-1,
                                                                       len(mapped_features):-1], reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,Df_holdout_orig,DF_holdout_r)

                # Mistakes_X1_tr_RG, Mistakes_X2_tr_RG, F1_x1_RG, F1_x2_RG = RadialGAN(df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features ,partition, Cor_from_df, Df_holdout_orig,
                #     DF_holdout_r,sq_transf_features,P_x1[len(mapped_features):-1,
                #                                                        len(mapped_features):-1])

                o_to_r_sim_chimeric, r_to_o_sim_chimeric, correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, match_details_x1, match_details_x2, F1_x1, F1_x2, match_mse_chimeric, recon_mse_chimeric = Train_cross_AE(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):-1, len(mapped_features):-1], reordered_column_names_orig, reordered_column_names_r,
                    mapped_features, Cor_from_df, Df_holdout_orig, DF_holdout_r,  unmapped_features_to_drop_from_orig, normalizing_values_orig, normalizing_values_r, DF_holdout_orig0_not_includedwhiletraining)

                _, _, correct_with_match_from_x1_test_CL, correct_with_match_from_x2_test_CL, F1_x1_CL, F1_x2_CL, _, _, match_mse_cl = Train_CL(
                    df_train_preproc.copy(), df_rename_preproc.copy(), P_x1[len(mapped_features):-1, len(mapped_features):-1], reordered_column_names_orig, reordered_column_names_r,
                    mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r, partition, trial)

                grad_sum_unkn_o, grad_sum_unkn_r, correct_with_match_from_x1_test_CL_Dec, correct_with_match_from_x2_test_CL_Dec, F1_x1_CL_Dec, F1_x2_CL_Dec, rep_known_val_o, rep_known_val_r,  match_mse_cl_dec, recon_mse_cl_dec = Train_CL_withDec(
                    df_train_preproc.copy(), df_rename_preproc.copy(), reordered_column_names_orig, reordered_column_names_r,
                    mapped_features, Cor_from_df, Df_holdout_orig,
                    DF_holdout_r, normalizing_values_orig, normalizing_values_r,
                    P_x1[len(mapped_features):-1, len(mapped_features):-1])

                correct_with_match_from_x1_test_CL_KMFl, correct_with_match_from_x2_test_CL_KMFl, F1_x1_CL_CL_KMFl, F1_x2_CL_CL_KMFl, match_mse_cl_kmfl = CL_with_KMF_linear(grad_sum_unkn_o, grad_sum_unkn_r, Cor_X1_map_unmap,
                                   Cor_X2_map_unmap, P_x1[len(mapped_features):-1, len(mapped_features):-1], reordered_column_names_orig, reordered_column_names_r,
                                   mapped_features, Df_holdout_orig)

                correct_with_match_from_x1_test_CL_KMFen, correct_with_match_from_x2_test_CL_KMFen, F1_x1_CL_CL_KMFen, F1_x2_CL_CL_KMFen, match_mse_cl_kmfen = CL_with_KMF_CLencoded(grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o, rep_known_val_r,
                                      Df_holdout_orig, DF_holdout_r,P_x1[len(mapped_features):-1, len(mapped_features):-1], reordered_column_names_orig, reordered_column_names_r,
                                      mapped_features)

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = 1 - (
                            correct_with_match_from_x1_test ) / (max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = 1 - (
                            correct_with_match_from_x2_test ) / (max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr[
                    partition]
                Frac_mismatches_across_trial_perm_X2_tr[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr[
                    partition]

                no_match_inference_df_from_x1 = pd.concat([no_match_inference_df_from_x1, match_details_x1])
                no_match_inference_df_from_x2 = pd.concat([no_match_inference_df_from_x2, match_details_x2])


                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = 1 - (
                            correct_with_match_from_x1_test_sim_cor ) / (
                                                                                         max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = 1 - (
                            correct_with_match_from_x2_test_sim_cor ) / (
                                                                                         max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr_sim_Cor[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition]
                Frac_mismatches_across_trial_perm_X2_tr_sim_Cor[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition]

                # no_match_inference_df_from_x1_Sim_cor = pd.concat([no_match_inference_df_from_x1_Sim_cor, match_details_x1_sim_cor])
                # no_match_inference_df_from_x2_Sim_cor = pd.concat([no_match_inference_df_from_x2_Sim_cor, match_details_x2_sim_cor])

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = 1-(correct_with_match_from_x1_test_CL)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = 1-(correct_with_match_from_x2_test_CL)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition]

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = 1-(correct_with_match_from_x1_test_CL_Dec)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = 1-(correct_with_match_from_x2_test_CL_Dec)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition]

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition] = 1-(correct_with_match_from_x1_test_CL_KMFl)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition] = 1-(correct_with_match_from_x2_test_CL_KMFl)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_l[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_l[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition]

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition] = 1-(correct_with_match_from_x1_test_CL_KMFen)/(max_mistakes)
                Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition] = 1-(correct_with_match_from_x2_test_CL_KMFen)/(max_mistakes)

                Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_enc[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition]
                Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_enc[m, run_num] = Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition]

                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition] = 1-(correct_with_match_from_x1_test_Kang)/(max_mistakes)
                #
                Frac_mismatches_across_trial_perm_X1_tr_Kang[m, run_num] = \
                Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_Kang[partition]

                # Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_RG[partition] = len(Mistakes_X1_tr_RG)/max_mistakes
                # Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_RG[partition] = len(Mistakes_X2_tr_RG)/max_mistakes
                #
                # Frac_mismatches_across_trial_perm_X1_tr_RG[m, run_num] = len(Mistakes_X1_tr_RG)/max_mistakes
                # Frac_mismatches_across_trial_perm_X2_tr_RG[m, run_num] = len(Mistakes_X2_tr_RG)/max_mistakes

                F1_for_fixed_trial_fixed_num_mapped_X1_tr[partition] = F1_x1
                F1_for_fixed_trial_fixed_num_mapped_X2_tr[partition] = F1_x2

                F1_across_trial_perm_X1_tr[m, run_num] = F1_x1
                F1_across_trial_perm_X2_tr[m, run_num] = F1_x2

                MatchMSE_across_trial_perm_X1_tr[m, run_num] = match_mse_chimeric
                ReconMSE_across_trial_perm_X1_tr[m, run_num] = recon_mse_chimeric

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = F1_x1_sim_cor
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = F1_x2_sim_cor

                F1_across_trial_perm_X1_tr_sim_Cor[m, run_num] = F1_x1_sim_cor
                F1_across_trial_perm_X2_tr_sim_Cor[m, run_num] = F1_x2_sim_cor

                MatchMSE_across_trial_perm_X1_tr_sim_Cor[m, run_num] = match_mse_kmf

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = F1_x1_CL
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = F1_x2_CL

                F1_across_trial_perm_X1_tr_CL[m, run_num] = F1_x1_CL
                F1_across_trial_perm_X2_tr_CL[m, run_num] = F1_x2_CL

                MatchMSE_across_trial_perm_X1_tr_CL[m,run_num] = match_mse_cl

                F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = F1_x1_CL_Dec
                F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = F1_x2_CL_Dec

                F1_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = F1_x1_CL_Dec
                F1_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = F1_x2_CL_Dec

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


                run_num = run_num + 1

                # Deleting the reshuffled as we have already made a copy earlier
                del df_rename_preproc, df_train_preproc, DF_holdout_r, Df_holdout_orig


            # storing the averaged mismatches across all paritition for a fixed trial and fixed number of mapped features

            print(" Simple_correlation when the number of mapped features are  ", mpfeatures)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            print(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

            if Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor != []:
                AVG_MISMATCHES_X1_tr_sim_Cor[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor)
            if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor != []:
                AVG_MISMATCHES_X2_tr_sim_Cor[m, trial] = np.average(Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor)

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

            print("------")

        m = m + 1

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

    # breakpoint()

    # return AVG_F1_X1_tr, AVG_F1_X2_tr, np.average(AVG_F1_X1_tr, axis=1), np.average(AVG_F1_X2_tr, axis = 1), F1_across_trial_perm_X1_tr, F1_across_trial_perm_X2_tr, MatchMSE_across_trial_perm_X1_tr, ReconMSE_across_trial_perm_X1_tr, \
    #        AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, np.average(AVG_F1_X1_tr_sim_Cor, axis=1), np.average(AVG_F1_X2_tr_sim_Cor, axis = 1), F1_across_trial_perm_X1_tr_sim_Cor, F1_across_trial_perm_X2_tr_sim_Cor, MatchMSE_across_trial_perm_X1_tr_sim_Cor,\
    #        AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, np.average(AVG_F1_X1_tr_CL, axis=1), np.average(AVG_F1_X2_tr_CL, axis = 1), F1_across_trial_perm_X1_tr_CL, F1_across_trial_perm_X2_tr_CL,  MatchMSE_across_trial_perm_X1_tr_CL,\
    #        AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, np.average(AVG_F1_X1_tr_CL_with_Dec, axis=1), np.average(AVG_F1_X2_tr_CL_with_Dec,axis=1), F1_across_trial_perm_X1_tr_CL_with_Dec, F1_across_trial_perm_X2_tr_CL_with_Dec, MatchMSE_across_trial_perm_X1_tr_CL_Dec, ReconMSE_across_trial_perm_X1_tr_CL_Dec, \
    #        AVG_F1_X1_tr_CL_KMF_l, AVG_F1_X2_tr_CL_KMF_l, np.average(AVG_F1_X1_tr_CL_KMF_l, axis=1), np.average(AVG_F1_X2_tr_CL_KMF_l, axis = 1), F1_across_trial_perm_X1_tr_CL_KMF_l, F1_across_trial_perm_X2_tr_CL_KMF_l,  MatchMSE_across_trial_perm_X1_tr_CL_KMF_l, \
    #        AVG_F1_X1_tr_CL_KMF_enc, AVG_F1_X2_tr_CL_KMF_enc, np.average(AVG_F1_X1_tr_CL_KMF_enc, axis=1), np.average(AVG_F1_X2_tr_CL_KMF_enc, axis = 1), F1_across_trial_perm_X1_tr_CL_KMF_enc, F1_across_trial_perm_X2_tr_CL_KMF_enc, MatchMSE_across_trial_perm_X1_tr_CL_KMF_en, \
    #        AVG_F1_X1_tr_Kang, np.average(AVG_F1_X1_tr_Kang,axis=1), F1_across_trial_perm_X1_tr_Kang, MatchMSE_across_trial_perm_X1_tr_Kang, AVG_F1_X1_tr_RG, AVG_F1_X2_tr_RG, np.average(AVG_F1_X1_tr_RG, axis=1), np.average(AVG_F1_X2_tr_RG, axis = 1), F1_across_trial_perm_X1_tr_RG, F1_across_trial_perm_X2_tr_RG, MatchMSE_across_trial_perm_X1_tr_RG

    return AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, np.average(AVG_F1_X1_tr_sim_Cor, axis=1), np.average(
    AVG_F1_X2_tr_sim_Cor, axis=1), F1_across_trial_perm_X1_tr_sim_Cor, F1_across_trial_perm_X2_tr_sim_Cor, \
         AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, np.average(AVG_F1_X1_tr_CL, axis=1), np.average(AVG_F1_X2_tr_CL,
                                                                                           axis=1), F1_across_trial_perm_X1_tr_CL, F1_across_trial_perm_X2_tr_CL,\
         AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, np.average(AVG_F1_X1_tr_CL_with_Dec, axis=1), np.average(AVG_F1_X2_tr_CL_with_Dec,axis=1), F1_across_trial_perm_X1_tr_CL_with_Dec, F1_across_trial_perm_X2_tr_CL_with_Dec, \
         AVG_F1_X1_tr_CL_KMF_l, AVG_F1_X2_tr_CL_KMF_l, np.average(AVG_F1_X1_tr_CL_KMF_l, axis=1), np.average(
    AVG_F1_X2_tr_CL_KMF_l, axis=1), F1_across_trial_perm_X1_tr_CL_KMF_l, F1_across_trial_perm_X2_tr_CL_KMF_l, \
         AVG_F1_X1_tr_CL_KMF_enc, AVG_F1_X2_tr_CL_KMF_enc, np.average(AVG_F1_X1_tr_CL_KMF_enc, axis=1), np.average(
    AVG_F1_X2_tr_CL_KMF_enc, axis=1), F1_across_trial_perm_X1_tr_CL_KMF_enc, F1_across_trial_perm_X2_tr_CL_KMF_enc


# presetting the number of threads to be used
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(1.0, device=None)

# starting time of the script
start_time = datetime.now()

parser = argparse.ArgumentParser(description='HP for CL optimization; partial')

## for the Encoders width (Known mapped)
parser.add_argument("--encKnwMapDepth",  default=3, type=int) #
parser.add_argument("--encKnwMapWidth",  default=31, type=int) #
parser.add_argument("--encKnwMapWidthFinal",  default=26, type=int) # this should be same for all three encoders by design
parser.add_argument("--encKnwMapL2",  default=0.2, type=float)
parser.add_argument("--encKnwMapL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset1 (orig))
parser.add_argument("--encUknwD1OrigDepth",  default=4, type=int) #
parser.add_argument("--encUknwD1OrigWidth",  default=37, type=int) #
parser.add_argument("--encUknwD1OrigWidthFinal",  default=None, type=int) # setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD1OrigL2",  default=0.2, type=float)
parser.add_argument("--encUknwD1OrigL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset2 (r))
parser.add_argument("--encUknwD2ReDepth",  default=2, type=int) #
parser.add_argument("--encUknwD2ReWidth",  default=27, type=int) #
parser.add_argument("--encUknwD2ReWidthFinal",  default=None, type=int) ## setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD2ReL2",  default=0.2, type=float)
parser.add_argument("--encUknwD2ReL1",  default=0.1, type=float)

## learning parameters
parser.add_argument("--batchSize",  default=7, type=int) #
parser.add_argument("--learningRate",  default=0.0004, type=float) #
parser.add_argument("--learningRateFactor",  default=0.27, type=float) #
parser.add_argument("--LRPatience",  default=2, type=int) #
parser.add_argument("--epochs",  default=1, type=int) #

## CL specific parameters
parser.add_argument("--tau", default=0.3058, type=float)  # temperature parameter in CL loss
parser.add_argument("--masking_ratio", default=0.4179, type=float)  # ratio for creating a new self augmented view
parser.add_argument("--mpfeatures", default=5, type=int)  # number of mapped_features  For Nomao/Superconductor it is 5/15
parser.add_argument("--large_database", default=35, type= int)  # number of features in the larger database
parser.add_argument("--block_stand_comb", default=0, type=int)  # can set the type as bool too but for now int it is
parser.add_argument("--dropout_rate_CL", default=0.608, type=float)
parser.add_argument("--weightDirDecoder", default =0.5984, type=float)
parser.add_argument("--weightCrossDecoder", default =0.7571, type=float)
parser.add_argument("--weightCombDecoder", default =0.338, type=float)

## dataset and setup parameters
parser.add_argument("--dataset_number",  default='Nomao') # could be integer or string
parser.add_argument("--outcome",  default="Y")  # this is not very relevant but kep for the sake of completeness
parser.add_argument("--frac_renamed", default=0.5, type=float)
parser.add_argument("--randomSeed", default=5640, type=int )
parser.add_argument("--testDatasetSize", default=0.2, type=float) # this top fraction of data is not used by this code
# parser.add_argument("--dataset_no_sample", default=1, type=int) # this is used to decide which one to tune the HPs on in case of synthetic;
parser.add_argument("--num_of_dataset_samples", default=1, type=int) # number of dataset instances to be used from one  distribution
parser.add_argument("--n_p", default=2, type=int) # number of permutations
parser.add_argument("--n_t", default=1, type=int) # number of data partitioning trials
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

list_of_total_Features_in_large_database = [large_database]
num_feat_sq_trans = 0

# data details
num_of_dataset_samples = 1
# mpfeatures = 15  # number of mapped features. For dataset 5 it is 4, For Nomao/Superconductor it is 5/15
num_xtra_feat_inX1 = 0  # 0 in onto case, 5 in partial mapping case
datatype = 'c'  # b denotes when the data needs to be binarized


list_of_total_feat_in_D2_minus_mapped = [i-mpfeatures for i in list_of_total_Features_in_large_database]


# model details
batchnorm = 0  # 1 denotes present and 0 denotes absent
orthogonalization_type = 1  # 0 denotes no orthognalization, 1 denotes simple, 2 denotes SDL, 3 denotes matching orthogonalization
weight_direct = 0.8
weight_cross = 1.1  # 0 denotes no cross loss, 1 denotes cross loss
weight_cycle = 0.8

alpha = 1.2  # used in KANG method

# model architecture and parameter details
hidden_dim = 15
num_of_hidden_layers = 2
batch_size = 64
# epochs = 30
learning_rate = 1e-2
dropout_rate = 0.6


epochs_RadialGAN = 2
beta = 10  # grad penalty parameter
Lambda_dir = 10  # cycle consistency weight on dir
Lambda_on_z = 1  # cycle consistency weight on z

# data_dir = "/research2/trips/Feature_confusion/data/"
data_dir = '/input/'


AVG_over_Dataset_samples_X1_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_sim_Cor = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))

AVG_over_Dataset_samples_X1_tr_CL = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))


AVG_over_Dataset_samples_X1_tr_CL_with_Dec = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_with_Dec = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))


AVG_over_Dataset_samples_X1_tr_CL_KMF_l = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_l = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))

AVG_over_Dataset_samples_X1_tr_CL_KMF_enc = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_enc = np.zeros((len(list_of_total_Features_in_large_database), num_of_dataset_samples))



for sample_no in range(1,num_of_dataset_samples+1):
    print("\n ********************************************************")
    print(" \n Run STARTS for sample no ", sample_no, "  of dataset ", dataset_number, "\n")
    print(" ******************************************************** \n")

    # AVG_F1_X1_tr, AVG_F1_X2_tr, m_x1, m_x2_tr, F1_elongated_X1_tr, F1_elongated_X2_tr, MatchMSE_elongated_X1_tr, ReconMSE_elongated_X1_tr ,  AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, m_x1_sim_Cor, m_x2_tr_sim_Cor, F1_elongated_X1_tr_sim_Cor, F1_elongated_X2_tr_sim_Cor, MatchMSE_elongated_X1_tr_simCor , AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, m_x1_CL, m_x2_tr_CL, F1_elongated_X1_tr_CL, F1_elongated_X2_tr_CL, MatchMSE_elongated_X1_tr_CL, AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, m_x1_CL_with_Dec, m_x2_tr_CL_with_Dec, F1_elongated_X1_tr_CL_with_Dec, F1_elongated_X2_tr_CL_with_Dec, MatchMSE_elongated_X1_tr_CL_Dec, ReconMSE_elongated_X1_tr_CL_Dec, AVG_F1_X1_tr_CL_kmfl, AVG_F1_X2_tr_CL_kmfl, m_x1_CL_kmfl, m_x2_tr_CL_kmfl, F1_elongated_X1_tr_CL_kmfl, F1_elongated_X2_tr_CL_kmfl, MatchMSE_elongated_X1_tr_CL_KMFl, AVG_F1_X1_tr_CL_kmfen, AVG_F1_X2_tr_CL_kmfen, m_x1_CL_kmfen, m_x2_tr_CL_kmfen, F1_elongated_X1_tr_CL_kmfen, F1_elongated_X2_tr_CL_kmfen, MatchMSE_elongated_X1_tr_CL_KMFen, AVG_F1_X1_tr_KANG, m_X1_tr_KANG, F1_elongated_X1_tr_KANG, MatchMSE_elongated_X1_tr_KANG, AVG_F1_X1_tr_RG, AVG_F1_X2_tr_RG, m_x1_RG, m_x2_tr_RG, F1_elongated_X1_tr_RG, F1_elongated_X2_tr_RG, MatchMSE_elongated_X1_tr_RG  = main(sample_no)
    AVG_F1_X1_tr_sim_Cor, AVG_F1_X2_tr_sim_Cor, m_x1_sim_Cor, m_x2_tr_sim_Cor, F1_elongated_X1_tr_sim_Cor, F1_elongated_X2_tr_sim_Cor, AVG_F1_X1_tr_CL, AVG_F1_X2_tr_CL, m_x1_CL, m_x2_tr_CL, F1_elongated_X1_tr_CL, F1_elongated_X2_tr_CL,  AVG_F1_X1_tr_CL_with_Dec, AVG_F1_X2_tr_CL_with_Dec, m_x1_CL_with_Dec, m_x2_tr_CL_with_Dec, F1_elongated_X1_tr_CL_with_Dec, F1_elongated_X2_tr_CL_with_Dec, AVG_F1_X1_tr_CL_kmfl, AVG_F1_X2_tr_CL_kmfl, m_x1_CL_kmfl, m_x2_tr_CL_kmfl, F1_elongated_X1_tr_CL_kmfl, F1_elongated_X2_tr_CL_kmfl,  AVG_F1_X1_tr_CL_kmfen, AVG_F1_X2_tr_CL_kmfen, m_x1_CL_kmfen, m_x2_tr_CL_kmfen, F1_elongated_X1_tr_CL_kmfen, F1_elongated_X2_tr_CL_kmfen  = main(sample_no)

    # for Simple_correlation (Simple_correlation)

    AVG_over_Dataset_samples_X1_tr_sim_Cor[:,sample_no-1] = m_x1_sim_Cor
    AVG_over_Dataset_samples_X2_tr_sim_Cor[:,sample_no-1] = m_x2_tr_sim_Cor



    if sample_no == 1:
        F1_elongated_x1_tr_list_sim_Cor = F1_elongated_X1_tr_sim_Cor
        F1_elongated_x2_tr_list_sim_Cor = F1_elongated_X2_tr_sim_Cor
    else:
        F1_elongated_x1_tr_list_sim_Cor = np.hstack((F1_elongated_x1_tr_list_sim_Cor, F1_elongated_X1_tr_sim_Cor))
        F1_elongated_x2_tr_list_sim_Cor = np.hstack((F1_elongated_x2_tr_list_sim_Cor, F1_elongated_X2_tr_sim_Cor))

    # for CL

    AVG_over_Dataset_samples_X1_tr_CL[:,sample_no-1] = m_x1_CL
    AVG_over_Dataset_samples_X2_tr_CL[:,sample_no-1] = m_x2_tr_CL


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL = F1_elongated_X1_tr_CL
        F1_elongated_x2_tr_list_CL = F1_elongated_X2_tr_CL
    else:
        F1_elongated_x1_tr_list_CL = np.hstack((F1_elongated_x1_tr_list_CL, F1_elongated_X1_tr_CL))
        F1_elongated_x2_tr_list_CL = np.hstack((F1_elongated_x2_tr_list_CL, F1_elongated_X2_tr_CL))

    # for CL + dec

    AVG_over_Dataset_samples_X1_tr_CL_with_Dec[:,sample_no-1] = m_x1_CL_with_Dec
    AVG_over_Dataset_samples_X2_tr_CL_with_Dec[:,sample_no-1] = m_x2_tr_CL_with_Dec


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_with_Dec = F1_elongated_X1_tr_CL_with_Dec
        F1_elongated_x2_tr_list_CL_with_Dec = F1_elongated_X2_tr_CL_with_Dec
    else:
        F1_elongated_x1_tr_list_CL_with_Dec = np.hstack((F1_elongated_x1_tr_list_CL_with_Dec, F1_elongated_X1_tr_CL_with_Dec))
        F1_elongated_x2_tr_list_CL_with_Dec = np.hstack((F1_elongated_x2_tr_list_CL_with_Dec, F1_elongated_X2_tr_CL_with_Dec))



    # for CL + KMFl

    AVG_over_Dataset_samples_X1_tr_CL_KMF_l[:,sample_no-1] = m_x1_CL_kmfl
    AVG_over_Dataset_samples_X2_tr_CL_KMF_l[:,sample_no-1] = m_x2_tr_CL_kmfl


    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_kmfl = F1_elongated_X1_tr_CL_kmfl
        F1_elongated_x2_tr_list_CL_kmfl = F1_elongated_X2_tr_CL_kmfl
    else:
        F1_elongated_x1_tr_list_CL_kmfl = np.hstack((F1_elongated_x1_tr_list_CL_kmfl, F1_elongated_X1_tr_CL_kmfl))
        F1_elongated_x2_tr_list_CL_kmfl = np.hstack((F1_elongated_x2_tr_list_CL_kmfl, F1_elongated_X2_tr_CL_kmfl))

    # for CL + KMFen

    AVG_over_Dataset_samples_X1_tr_CL_KMF_enc[:,sample_no-1] = m_x1_CL_kmfen
    AVG_over_Dataset_samples_X2_tr_CL_KMF_enc[:,sample_no-1] = m_x2_tr_CL_kmfen

    if sample_no == 1:
        F1_elongated_x1_tr_list_CL_kmfen = F1_elongated_X1_tr_CL_kmfen
        F1_elongated_x2_tr_list_CL_kmfen = F1_elongated_X2_tr_CL_kmfen
    else:
        F1_elongated_x1_tr_list_CL_kmfen = np.hstack((F1_elongated_x1_tr_list_CL_kmfen, F1_elongated_X1_tr_CL_kmfen))
        F1_elongated_x2_tr_list_CL_kmfen = np.hstack((F1_elongated_x2_tr_list_CL_kmfen, F1_elongated_X2_tr_CL_kmfen))

    # for RadialGAN (RG)

    # AVG_over_Dataset_samples_X1_tr_RG[:,sample_no-1] = m_x1_RG
    # AVG_over_Dataset_samples_X2_tr_RG[:,sample_no-1] = m_x2_tr_RG
    #
    # f = open(file_name_RG,'a')
    # f.write("\n \n Frac of Mismatches for different trials on sample number {0}".format(sample_no))
    # f.write("\n X1_train \n")
    # f.write("{0}".format(AVG_F1_X1_tr_RG))
    # f.write("\n X2_train \n")
    # f.write("{0}".format(AVG_F1_X2_tr_RG))
    # f.write("\n \n ")
    # f.close()


    # if sample_no == 1:
    #     F1_elongated_x1_tr_list_RG = F1_elongated_X1_tr_RG
    #     F1_elongated_x2_tr_list_RG = F1_elongated_X2_tr_RG
    #     MatchMSE_elongated_X1_tr_list_RG = MatchMSE_elongated_X1_tr_RG
    #
    # else:
    #     F1_elongated_x1_tr_list_RG = np.hstack((F1_elongated_x1_tr_list_RG, F1_elongated_X1_tr_RG))
    #     F1_elongated_x2_tr_list_RG = np.hstack((F1_elongated_x2_tr_list_RG, F1_elongated_X2_tr_RG))
    #     MatchMSE_elongated_X1_tr_list_RG = np.hstack((MatchMSE_elongated_X1_tr_list_RG, MatchMSE_elongated_X1_tr_RG))

    print("\n ********************************************************")
    print(" \n Run ENDS for sample no ", sample_no, "  of dataset ", dataset_number, "\n ")
    print(" ******************************************************** \n")


# Computing the average over the datset samples

Mean_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_sim_Cor = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_sim_Cor = np.zeros(len(list_of_total_Features_in_large_database))

Mean_over_trials_mismatches_X1_tr_CL = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_CL = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_CL = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_CL = np.zeros(len(list_of_total_Features_in_large_database))


Mean_over_trials_mismatches_X1_tr_CL_with_Dec = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_CL_with_Dec = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_CL_with_Dec = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_CL_with_Dec = np.zeros(len(list_of_total_Features_in_large_database))

Mean_over_trials_mismatches_X1_tr_CL_kmfl = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_CL_kmfl = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_CL_kmfl = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_CL_kmfl = np.zeros(len(list_of_total_Features_in_large_database))

Mean_over_trials_mismatches_X1_tr_CL_kmfen = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X1_tr_CL_kmfen = np.zeros(len(list_of_total_Features_in_large_database))
Mean_over_trials_mismatches_X2_tr_CL_kmfen = np.zeros(len(list_of_total_Features_in_large_database))
SD_over_trials_mismatches_X2_tr_CL_kmfen = np.zeros(len(list_of_total_Features_in_large_database))


x_axis = np.arange(len(list_of_total_Features_in_large_database))
x_axis1 = x_axis + 0.05
x_axis2 = x_axis + 0.1

for i in range(len(list_of_total_Features_in_large_database)):

    Mean_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X1_tr_sim_Cor[i, :]), decimals=3)
    Mean_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.mean(AVG_over_Dataset_samples_X2_tr_sim_Cor[i, :]), decimals=3)
    SD_over_trials_mismatches_X1_tr_sim_Cor[i] = np.round(np.std(F1_elongated_x1_tr_list_sim_Cor[i, :]), decimals=4)
    SD_over_trials_mismatches_X2_tr_sim_Cor[i] = np.round(np.std(F1_elongated_x2_tr_list_sim_Cor[i, :]), decimals=4)

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


print(" Mean F1 score values when # of features in large database is ", large_database)
print("\t KMF \t CL \t CL+dec  \t CL+KMFl \t CL+KMFenc")
print(Mean_over_trials_mismatches_X1_tr_sim_Cor, "\t", Mean_over_trials_mismatches_X1_tr_CL, "\t", Mean_over_trials_mismatches_X1_tr_CL_with_Dec, "\t", Mean_over_trials_mismatches_X1_tr_CL_kmfl, "\t", Mean_over_trials_mismatches_X1_tr_CL_kmfen)
print(SD_over_trials_mismatches_X1_tr_sim_Cor, "\t", SD_over_trials_mismatches_X1_tr_CL, "\t", SD_over_trials_mismatches_X1_tr_CL_with_Dec, "\t", SD_over_trials_mismatches_X1_tr_CL_kmfl, "\t", SD_over_trials_mismatches_X1_tr_CL_kmfen)

end_time = datetime.now()  # only writing part is remaining in the code to time
timetaken = end_time - start_time
print("time taken to run the complete training script", timetaken)

csvdata = {
  'hp': json.dumps(vars(args)),
  'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
  'Mean_f1_KMF': Mean_over_trials_mismatches_X1_tr_sim_Cor,
  'Mean_f1_CL': Mean_over_trials_mismatches_X1_tr_CL,
  'Mean_f1_CL_withDec': Mean_over_trials_mismatches_X1_tr_CL_with_Dec,
  'Mean_f1_CL_KMFl': Mean_over_trials_mismatches_X1_tr_CL_kmfl,
  'Mean_f1_CL_KMFEn': Mean_over_trials_mismatches_X1_tr_CL_kmfen,
  'Num_inLargeDB': args.large_database,
  'name': args.nameinfo,
  'target': args.dataset_number,
  'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
  'time': timetaken
}

csvdata = pd.DataFrame(csvdata)
outputcsv = os.path.join('/output/', args.outputcsv)
if (os.path.exists(outputcsv)):
  csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
else:
  csvdata.to_csv(outputcsv, header=True, index=False)
