"""
This code uses contrastive learning for schema matching problem in following steps:
    1) Trains three encoders, one for the common known mapped features, two for the unmapped features from two databases using a contraastive loss.
    2) Positive pairs are the encoder outputs of mapped and unmapped sets from the same example and rest are unmappped.

To be used for HP tuning. The results are on a separate holdout data. The top 20% of the data is kept aside for testing.
For synthetic datasets this could be nil as there are various dataet instances from the same distribution.
From the remaining dataset train and validation is obtained.

INPUT:

Full dataset, the model details, number of permutations, number of partitioning of dataset, fraction of data to be permuted, number of mapped features

OUTPUT:

List of matches

"""

# importing packages

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


def generate_noisy_xbar(x, noise_type="Zero-out", noise_level=0.1):
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
    x_bar = x + torch.normal(0, noise_level, size=x.shape, device='cuda')
  else:
    x_bar = x_bar

  return x_bar

def Stable_matching_algorithm(C_X1_train, C_X2_train, index_O_to_R, index_R_to_O, num_mapped_axis):
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
    matching_x1_train_matrix[i, x1_train_y[i] - 1 - num_mapped_axis] = 1

  for i in range(matching_x2_train_matrix.shape[0]):
    # print(i, x2_train_y[i]-1)
    matching_x2_train_matrix[i, x2_train_y[i] - 1 - num_mapped_axis] = 1

  print("Mistakes x1")
  print(mismatched_x1_train)
  print(" Mistakes x2 train")
  print(mismatched_x2_train)

  return mismatched_x1_train, mismatched_x2_train, matching_x1_train_matrix, matching_x2_train_matrix

def Train_CL(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename, index_for_mapping_rename_to_orig,
               reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig,
               DF_holdout_r, normalizing_values_orig, normalizing_values_r, P_x1):
    mpfeatures = len(mapped_features)
    device = torch.device('cuda')
    num_NonCat_features_orig = len(reordered_column_names_r) - 1

    num_features = len(reordered_column_names_r) - 1
    num_NonCat_features_r = len(reordered_column_names_r) - 1

    dataset_orig = TabularDataset(data=df_train_preproc, output_col=outcome)
    train_loader_orig = DataLoader(dataset_orig, batchSize, shuffle=True, num_workers=1)
    dataset_orig_val = TabularDataset(data=Df_holdout_orig, output_col=outcome)
    val_loader_orig = DataLoader(dataset_orig_val, batchSize, shuffle=True, num_workers=1)

    dataset_r = TabularDataset(data=df_rename_preproc, output_col=outcome)
    train_loader_r = DataLoader(dataset_r, batchSize, shuffle=True, num_workers=1)
    dataset_r_val = TabularDataset(data=DF_holdout_r, output_col=outcome)
    val_loader_r = DataLoader(dataset_r_val, batchSize, shuffle=True, num_workers=1)

    known_features_encoder = AE_CL(input_shape=mpfeatures, hidden_units_final=encKnwMapWidthFinal,
                                                  hidden_depth=encKnwMapDepth,
                                                  hidden_units=encKnwMapWidth,drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_orig = AE_CL(input_shape=num_NonCat_features_orig - mpfeatures, hidden_units_final=encUknwD1OrigWidthFinal,
                                                  hidden_depth=encUknwD1OrigDepth,
                                                  hidden_units=encUknwD1OrigWidth,drop_out_p=dropout_rate_CL).to(device)
    unknown_features_encoder_r = AE_CL(input_shape=num_NonCat_features_r - mpfeatures, hidden_units_final=encUknwD2ReWidthFinal,
                                                  hidden_depth=encUknwD2ReDepth,
                                                  hidden_units=encUknwD2ReWidth,drop_out_p=dropout_rate_CL).to(device)

    withinCL_options = {'batch_size': batchSize, "tau": tau, "device": device, "cosine_similarity": True}
    aug_loss = JointLoss(withinCL_options)

    # breakpoint()
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
      # computing the representations from the trained encoders
      known_features_encoder.train()
      unknown_features_encoder_orig.train()
      unknown_features_encoder_r.train()

      for i, data in enumerate(zip(train_loader_orig, train_loader_r)):
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

          if True:  # adding self augmentation
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


      with torch.no_grad():
        # computing the representations from the trained encoders
        known_features_encoder.eval()
        unknown_features_encoder_orig.eval()
        unknown_features_encoder_r.eval()

        loss_val = 0
        loss_val_o = 0
        loss_val_r = 0
        rank_val_o = 0
        rank_val_r = 0
        counting_flag_for_rank_val = 0
        within_unkn_CL_loss_val = 0

        for i, data in enumerate(zip(val_loader_orig, val_loader_r)):
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
            contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r = NTXentLoss(known_rep_o, known_rep_r,
                                                                                        unknown_rep_o, unknown_rep_r,
                                                                                        tau)

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
      scheduler_known.step(loss_val)
      scheduler_unk_orig.step(loss_val)
      scheduler_unk_r.step(loss_val)

    # breakpoint()

    # computing the gradient of the output wrt the input data

    for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_o = torch.zeros((encUknwD1OrigWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]):
      grad_sum_unkn_o += torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])

    grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

    for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
    temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:-1].values).to(device)
    grad_sum_unkn_r = torch.zeros((encUknwD2ReWidthFinal, temp_input.shape[-1])).to(device)
    for i in range(temp_input.shape[0]):
      grad_sum_unkn_r += torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])

    grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

    o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
    r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                            np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)

    Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(
      o_to_r_sim,
      r_to_o_sim,
      index_for_mapping_orig_to_rename[len(mapped_features):],
      index_for_mapping_rename_to_orig[len(mapped_features):],
      len(mapped_features))

    print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

    MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
    MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

    print(" CL  X1_train mistakes", MisF_X1_te)
    print(" CL  X2_train mistakes", MisF_X2_te)

    print(" CL  X1_train mistakes (len) ", len(MisF_X1_te), " out of ", num_NonCat_features_orig - mpfeatures)
    print(" CL  X2_train mistakes (len) ", len(MisF_X2_te), " out of ", num_NonCat_features_r - mpfeatures)

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

    return grad_sum_unkn_o, grad_sum_unkn_r, MisF_X1_te, MisF_X2_te, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r


def Train_CL_withDec(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename,
                     index_for_mapping_rename_to_orig,
                     reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df,
                     Df_holdout_orig,
                     DF_holdout_r, normalizing_values_orig, normalizing_values_r, P_x1):
  mpfeatures = len(mapped_features)
  device = torch.device('cuda')
  num_NonCat_features_orig = len(reordered_column_names_r) - 1

  num_features = len(reordered_column_names_r) - 1
  num_NonCat_features_r = len(reordered_column_names_r) - 1

  unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures
  unmapped_features_r = len(reordered_column_names_r) - mpfeatures

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
  unknown_features_encoder_orig = AE_CL_withDec(input_shape=num_NonCat_features_orig - mpfeatures,
                                                hidden_units_final=encUknwD1OrigWidthFinal,
                                                hidden_depth=encUknwD1OrigDepth,
                                                hidden_units=encUknwD1OrigWidth, drop_out_p=dropout_rate_CL).to(device)
  unknown_features_encoder_r = AE_CL_withDec(input_shape=num_NonCat_features_r - mpfeatures,
                                             hidden_units_final=encUknwD2ReWidthFinal,
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
        points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + ".png")
    plt.savefig(
      saving_dir + "/Before_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
        encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
        points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + ".pdf")
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
        direct_rec_loss = rec_criterion(known_dec_o, datalist[0]) + rec_criterion(known_dec_r,
                                                                                  datalist[1]) + rec_criterion(
          unknown_dec_o, datalist[2]) + rec_criterion(unknown_dec_r, datalist[3])

        # breakpoint()
        # Cross reconstruction (unknown to known, combined rep to known, combined rep to unknown)
        known_cross_dec_o = known_features_encoder(unknown_rep_o, 1)
        known_cross_dec_r = known_features_encoder(unknown_rep_r, 1)

        supervised_loss_Known = rec_criterion(known_cross_dec_o, datalist[0]) + rec_criterion(known_cross_dec_r,
                                                                                              datalist[1])

        known_comb_dec_o = known_features_encoder(0.5 * unknown_rep_o + 0.5 * known_rep_o, 1)
        known_comb_dec_r = known_features_encoder(0.5 * unknown_rep_r + 0.5 * known_rep_r, 1)
        unknown_comb_dec_o = unknown_features_encoder_orig(0.5 * unknown_rep_o + 0.5 * known_rep_o, 1)
        unknown_comb_dec_r = unknown_features_encoder_r(0.5 * unknown_rep_r + 0.5 * known_rep_r, 1)

        Comb_supervised_loss = rec_criterion(known_comb_dec_o, datalist[0]) + rec_criterion(known_comb_dec_r,
                                                                                            datalist[1]) \
                               + rec_criterion(unknown_comb_dec_o, datalist[2]) + rec_criterion(unknown_comb_dec_r,
                                                                                                datalist[3])

        # breakpoint()

        if True:  # adding self augmentation
          self_aug_loss_un = aug_loss(unknown_rep_o) + aug_loss(unknown_rep_r) + aug_loss(
            known_rep_o) + aug_loss(known_rep_r)
          contrastive_loss = contrastive_loss + self_aug_loss_un

        # combining the contrastive and decoder losses # TODO: multipliers of the various losses
        contrastive_loss = contrastive_loss + 100 * (
                  weightDirDecoder * direct_rec_loss + weightCrossDecoder * supervised_loss_Known + weightCombDecoder * Comb_supervised_loss)

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
            contrastive_loss_o, contrastive_loss_r, avg_Rank_o, avg_Rank_r = NTXentLoss(known_rep_o, known_rep_r,
                                                                                        unknown_rep_o, unknown_rep_r,
                                                                                        tau)

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
        points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + ".png")
    plt.savefig(
      saving_dir + "/After_training_Comp_ModalityGapSVD_Dataset_MIMIC_X_o_dim_" + str(
        encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_#datapoints_" + str(
        points_to_plot) + "_epochs_" + str(epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + ".pdf")
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
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".png")
    plt.savefig(saving_dir + "/basic_MIMIC_Known_encoder_Comb_summary_" + str(
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".pdf")
    plt.close()

    en_unkn_orig = shap.DeepExplainer(unknown_features_encoder_orig,
                                      torch.Tensor(Df_holdout_orig.iloc[:10, mpfeatures:].values).to(device))
    shap_vals_unkn_orig = en_unkn_orig.shap_values(
      torch.Tensor(Df_holdout_orig.iloc[10:100, mpfeatures:].values).to(device))

    array_shap_dimUn_orig = np.zeros((len(shap_vals_unkn_orig), len(reordered_column_names_orig[mpfeatures:])))
    for i in range(len(shap_vals_unkn_orig)): array_shap_dimUn_orig[i] = np.mean(
      np.absolute(np.transpose(shap_vals_unkn_orig[i])), 1)

    shap.summary_plot(shap_vals_unkn_orig,
                      feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_orig[mpfeatures:]],
                      show=False,
                      max_display=50)
    plt.legend().set_visible(False)
    plt.title(" Feature importance from Unknown Encoder Original ")
    plt.tight_layout()
    plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".png")
    plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_Orig_summary_" + str(
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".pdf")
    plt.close()

    en_unkn_r = shap.DeepExplainer(unknown_features_encoder_r,
                                   torch.Tensor(DF_holdout_r.iloc[:10, mpfeatures:].values).to(device))
    shap_vals_unkn_r = en_unkn_r.shap_values(torch.Tensor(DF_holdout_r.iloc[10:100, mpfeatures:].values).to(device))

    array_shap_dimUn_r = np.zeros((len(shap_vals_unkn_r), len(reordered_column_names_r[mpfeatures:])))
    for i in range(len(shap_vals_unkn_r)): array_shap_dimUn_r[i] = np.mean(
      np.absolute(np.transpose(shap_vals_unkn_r[i])), 1)

    shap.summary_plot(shap_vals_unkn_r,
                      feature_names=[itemid_label_dict[int(i)] for i in reordered_column_names_r[mpfeatures:]],
                      show=False, max_display=50)
    plt.legend().set_visible(False)
    plt.title(" Feature importance from Unknown Encoder R ")
    plt.tight_layout()
    plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".png")
    plt.savefig(saving_dir + "/basic_MIMIC_unknown_encoder_R_summary_" + str(
      encKnwMapWidthFinal) + "_tau_" + str(tau) + "_mapped_features_" + str(mpfeatures) + "_epochs_" + str(
      epochs) + "_representation_dim_" + str(encKnwMapWidthFinal) + "_np_nt_" + str(partition) + "_" + str(
      trial) + ".pdf")
    plt.close()

  # #breakpoint()
  # computing the gradient of the output wrt the input data
  # breakpoint()
  #

  for param in unknown_features_encoder_orig.parameters(): param.requires_grad = False
  temp_input = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)
  grad_sum_unkn_o = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
  for i in range(temp_input.shape[0]): grad_sum_unkn_o += \
  torch.autograd.functional.jacobian(unknown_features_encoder_orig, temp_input[i])[0]

  grad_sum_unkn_o = grad_sum_unkn_o / temp_input.shape[0]

  for param in unknown_features_encoder_r.parameters(): param.requires_grad = False
  temp_input = torch.Tensor(DF_holdout_r.iloc[:, mpfeatures:-1].values).to(device)
  grad_sum_unkn_r = torch.zeros((encKnwMapWidthFinal, temp_input.shape[-1])).to(device)
  for i in range(temp_input.shape[0]): grad_sum_unkn_r += \
  torch.autograd.functional.jacobian(unknown_features_encoder_r, temp_input[i])[0]

  grad_sum_unkn_r = grad_sum_unkn_r / temp_input.shape[0]

  # np.savetxt(output_dir+"Grad_unknwn_Orig_" + str(randomSeed) + "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_o.cpu().detach().numpy(), delimiter=",")
  # np.savetxt(output_dir+"Grad_unknwn_R_" + str(randomSeed) +  "_mp_Features_" +str(len(mapped_features))+ ".csv", grad_sum_unkn_r.cpu().detach().numpy(), delimiter=",")
  # exit()

  o_to_r_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_o.cpu().detach().numpy()),
                                          np.transpose(grad_sum_unkn_r.cpu().detach().numpy()), dense_output=True)
  r_to_o_sim = pairwise.cosine_similarity(np.transpose(grad_sum_unkn_r.cpu().detach().numpy()),
                                          np.transpose(grad_sum_unkn_o.cpu().detach().numpy()), dense_output=True)

  Mistakes_X1_te, Mistakes_X2_te, x1_match_matrix_test, x2_match_matrix_test = Stable_matching_algorithm(
    o_to_r_sim,
    r_to_o_sim,
    index_for_mapping_orig_to_rename[len(mapped_features):],
    index_for_mapping_rename_to_orig[len(mapped_features):],
    len(mapped_features))

  print("\n \n List of mismatched feature number when # of mapped features are ", mpfeatures, "\n ")

  MisF_X1_te = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te]
  MisF_X2_te = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te]

  print(" CL + Dec  X1_train mistakes", MisF_X1_te)
  print(" CL + Dec X2_train mistakes", MisF_X2_te)

  print(" CL + Dec X1_train mistakes (len) ", len(MisF_X1_te), " out of ", num_NonCat_features_orig - mpfeatures)
  print(" CL + Dec X2_train mistakes (len) ", len(MisF_X2_te), " out of ", num_NonCat_features_r - mpfeatures)

  print(" -------- CL + Dec method training ends ------------- \n \n  ")

  # to compare
  if False:
    # reconstrOrig_from_r_correct_matches = unknown_features_encoder_r(unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[:,correct_match_idx_r_from_x1] and
    Orig_correct_matches = torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device)[:,
                           correct_match_idx_orig_from_x1]

    # plotting the reconstruction after matching
    row_index_no_orig = np.random.choice(len(Orig_correct_matches), 500, replace=False)
    al = Df_holdout_orig.columns[mpfeatures:]
    for i in range(len(correct_match_idx_orig_from_x1)):  # plotting on the correctly mapped features
      x_axis = Df_holdout_orig.iloc[:, mpfeatures:].values[row_index_no_orig, correct_match_idx_orig_from_x1[i]]
      y_axis = unknown_features_encoder_r(
        unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:].values).to(device))[0], 1)[
        row_index_no_orig, correct_match_idx_r_from_x1[i]].cpu().detach().numpy()
      plt.scatter(x_axis, y_axis, color='blue')
      plt.xlabel("true X1 feature value")
      plt.ylabel("cross reconstructed feature value ")
      temp = stats.pearsonr(x_axis, y_axis)[0]
      plt.figtext(0.6, 0.8, "Cor_value = " + str(np.round(temp, decimals=3)))
      plt.title(" number of mapped feature  " + str(mpfeatures) + " & " + str(
        itemid_label_dict[int(al[correct_match_idx_orig_from_x1[i]])]) + " correctly mapped ", fontsize=8)
      plt.savefig(saving_dir + "/Cross_recon_qua_" + str(
        itemid_label_dict[int(al[correct_match_idx_orig_from_x1[i]])]) + "_Cor_Map" + ".png", bbox='tight')
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

  true_match_list = ['Col' + str(i + 1) for i in range(num_NonCat_features_orig)]
  match_dict = dict(zip(true_match_list, true_match_list))

  # breakpoint()
  final_dic_for_compar_matching = {}
  for key, val in match_dict.items():
    if val in predicted_match_dic_x1.values():
      final_dic_for_compar_matching[key] = list(predicted_match_dic_x1.keys())[
        list(predicted_match_dic_x1.values()).index(val)]
  # breakpoint()
  overall_quality_error_matching_only = mean_squared_error(Df_holdout_orig[final_dic_for_compar_matching.keys()].values,
                                                           Df_holdout_orig[final_dic_for_compar_matching.values()])

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
  for i in list(incorrect_match_dict_x1.keys()): incorrect_match_idx_orig_from_x1.append(
    list(Df_holdout_orig.columns[mpfeatures:]).index(i))
  incorrect_match_idx_r_from_x1 = []
  for i in list(incorrect_match_dict_x1.values()): incorrect_match_idx_r_from_x1.append(
    list(DF_holdout_r.columns[mpfeatures:]).index(i))

  # breakpoint()
  overall_quality_oracle_comb = rec_criterion(unknown_features_encoder_r(
    unknown_features_encoder_orig(torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device))[0], 1)[:,
                                              incorrect_match_idx_r_from_x1],
                                              torch.Tensor(Df_holdout_orig.iloc[:, mpfeatures:-1].values).to(device)[:,
                                              incorrect_match_idx_orig_from_x1])

  # encoding to be used later
  known_features_encoder.eval()
  # training dataset
  rep_known_val_o = known_features_encoder(
    torch.Tensor(Df_holdout_orig.iloc[:, :mpfeatures].values).to(device))[0].cpu().detach().numpy()
  rep_known_val_r = known_features_encoder(
    torch.Tensor(DF_holdout_r.iloc[:, :mpfeatures].values).to(device))[0].cpu().detach().numpy()

  del df_rename_preproc
  print("F1 + Dec from CL ", F1_fromx1)
  print('Matching metric ', overall_quality_error_matching_only, 'Oracle combo metric ', overall_quality_oracle_comb)
  # breakpoint()
  # exit()
  return grad_sum_unkn_o, grad_sum_unkn_r, MisF_X1_te, MisF_X2_te, F1_fromx1, F1_fromx2, rep_known_val_o, rep_known_val_r, overall_quality_error_matching_only, overall_quality_oracle_comb


def Train_KMF(df_train_preproc, df_rename_preproc, index_for_mapping_orig_to_rename,
              index_for_mapping_rename_to_orig
              , reordered_column_names_orig, reordered_column_names_r,
              mapped_features, Cor_from_df, Df_holdout_orig, DF_holdout_r, P_x1):
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

  Mistakes_X1_te_KMF, Mistakes_X2_te_KMF, x1_match_matrix_test_KMF, x2_match_matrix_test_KMF = Stable_matching_algorithm(
    sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1,
    index_for_mapping_orig_to_rename[
    len(mapped_features):],
    index_for_mapping_rename_to_orig[
    len(mapped_features):],
    len(mapped_features))

  MisF_X1_te_KMF = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te_KMF]
  MisF_X2_te_KMF = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te_KMF]

  print(" KMFl X1_train mistakes", MisF_X1_te_KMF)
  print(" KMFl X2_train mistakes", MisF_X2_te_KMF)

  print(" KMFl X1_train mistakes (len) ", len(MisF_X1_te_KMF), " out of ", unmapped_features_orig)
  print(" KMFl X2_train mistakes (len) ", len(MisF_X2_te_KMF), " out of ", unmapped_features_r)

  print(" -------- KMF-l methods  ends ------------- \n \n  ")

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

  return CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped, Mistakes_X1_te_KMF, Mistakes_X2_te_KMF, F1_fromx1, F1_fromx2


def CL_with_KMF_linear(grad_sum_unkn_o, grad_sum_unkn_r, CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped,
                       index_for_mapping_orig_to_rename,
                       index_for_mapping_rename_to_orig
                       , reordered_column_names_orig, reordered_column_names_r,
                       mapped_features, P_x1):
  mpfeatures = len(mapped_features)
  device = torch.device('cuda')
  num_NonCat_features_orig = len(reordered_column_names_r) - 1

  num_features = len(reordered_column_names_r) - 1
  num_NonCat_features_r = len(reordered_column_names_r) - 1

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

  Mistakes_X1_te_CL_KMFl, Mistakes_X2_te_CL_KMFl, x1_match_matrix_test_CL_KMFl, x2_match_matrix_test_CL_KMFl = Stable_matching_algorithm(
    o_to_r_sim_CL_KMFl,
    r_to_o_sim_CL_KMFl,
    index_for_mapping_orig_to_rename[len(mapped_features):],
    index_for_mapping_rename_to_orig[len(mapped_features):],
    len(mapped_features))

  MisF_X1_te_CL_KMFl = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te_CL_KMFl]
  MisF_X2_te_CL_KMFl = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te_CL_KMFl]

  print(" CL KMFl X1_train mistakes", MisF_X1_te_CL_KMFl)
  print(" CL KMFl X2_train mistakes", MisF_X2_te_CL_KMFl)

  print(" CL KMFl X1_train mistakes (len) ", len(MisF_X1_te_CL_KMFl), " out of ",
        num_NonCat_features_orig - mpfeatures)
  print(" CL KMFl X2_train mistakes (len) ", len(MisF_X2_te_CL_KMFl), " out of ", num_NonCat_features_r - mpfeatures)

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

  return MisF_X1_te_CL_KMFl, MisF_X2_te_CL_KMFl, F1_fromx1, F1_fromx2


def CL_with_KMF_CLencoded(grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o, rep_known_val_r, Df_holdout_orig,
                          DF_holdout_r,
                          index_for_mapping_orig_to_rename,
                          index_for_mapping_rename_to_orig
                          , reordered_column_names_orig, reordered_column_names_r,
                          mapped_features, P_x1):
  mpfeatures = len(mapped_features)
  device = torch.device('cuda')
  num_NonCat_features_orig = len(reordered_column_names_orig) - 1

  num_features = len(reordered_column_names_r) - 1
  num_NonCat_features_r = len(reordered_column_names_r) - 1

  unmapped_features_orig = len(reordered_column_names_orig) - mpfeatures - 1
  unmapped_features_r = len(reordered_column_names_r) - mpfeatures - 1

  # computing the correlation matrix between original feature values and cross reconstruction
  CorMatrix_X1_unmap_mappedE = np.zeros((unmapped_features_orig, encUknwD1OrigWidthFinal))
  CorMatrix_X2_unmap_mappedE = np.zeros((unmapped_features_r, encUknwD2ReWidthFinal))

  for i in range(unmapped_features_orig):
    for j in range(encUknwD1OrigWidthFinal):
      temp = stats.pearsonr(Df_holdout_orig.values[:, mpfeatures + i], rep_known_val_o[:, j])
      CorMatrix_X1_unmap_mappedE[i, j] = temp[0]

  for i in range(unmapped_features_r):
    for j in range(encUknwD2ReWidthFinal):
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
  # breakpoint()
  o_to_r_sim_CL_KMFen = pairwise.cosine_similarity(grad_sum_unkn_o_concat_en, grad_sum_unkn_r_concat_en,
                                                   dense_output=True)
  r_to_o_sim_CL_KMFen = pairwise.cosine_similarity(grad_sum_unkn_r_concat_en, grad_sum_unkn_o_concat_en,
                                                   dense_output=True)

  Mistakes_X1_te_CL_KMFen, Mistakes_X2_te_CL_KMFen, x1_match_matrix_test_CL_KMFen, x2_match_matrix_test_CL_KMFen = Stable_matching_algorithm(
    o_to_r_sim_CL_KMFen,
    r_to_o_sim_CL_KMFen,
    index_for_mapping_orig_to_rename[len(mapped_features):],
    index_for_mapping_rename_to_orig[len(mapped_features):],
    len(mapped_features))

  MisF_X1_te_CL_KMFen = [reordered_column_names_r[i - 1] for i in Mistakes_X1_te_CL_KMFen]
  MisF_X2_te_CL_KMFen = [reordered_column_names_orig[i - 1] for i in Mistakes_X2_te_CL_KMFen]

  print(" CL KMF-encoded X1_train mistakes", MisF_X1_te_CL_KMFen)
  print(" CL KMF-encoded X2_train mistakes", MisF_X2_te_CL_KMFen)

  print(" CL KMF-encoded X1_train mistakes (len) ", len(MisF_X1_te_CL_KMFen), " out of ",
        num_NonCat_features_orig - mpfeatures)
  print(" CL KMF-encoded X2_train mistakes (len) ", len(MisF_X2_te_CL_KMFen), " out of ",
        num_NonCat_features_r - mpfeatures)

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

  return MisF_X1_te_CL_KMFen, MisF_X2_te_CL_KMFen, F1_fromx1, F1_fromx2


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


# def NTXentLoss0(embeddings_knw_o, embeddings_knw_r, embeddings_unknw_o, embeddings_unknw_r,
#                temperature=0.1):  # embeddings from known features of both databases followed by the unknown features
#   # compute the cosine similarity bu first normalizing and then matrix multiplying the known and unknown tensors
#   cos_sim_o = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
#                                      torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
#                         temperature)
#   cos_sim_or = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
#                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
#                          temperature)
#   cos_sim_r = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
#                                      torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
#                         temperature)
#   cos_sim_ro = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
#                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
#                          temperature)
#   # for numerical stability  ## TODO update this logit name
#   logits_max_o, _ = torch.max(cos_sim_o, dim=1, keepdim=True)
#   logits_o = cos_sim_o - logits_max_o.detach()
#   logits_max_or, _ = torch.max(cos_sim_or, dim=1, keepdim=True)
#   logits_or = cos_sim_or - logits_max_or.detach()
#   logits_max_r, _ = torch.max(cos_sim_r, dim=1, keepdim=True)
#   logits_r = cos_sim_r - logits_max_r.detach()
#   logits_max_ro, _ = torch.max(cos_sim_ro, dim=1, keepdim=True)
#   logits_ro = cos_sim_ro - logits_max_ro.detach()
#
#   if True:
#     # computing the exp logits
#     exp_o = torch.exp(logits_o)
#     exp_r = torch.exp(logits_r)
#     batch_loss = - torch.log(exp_o.diag() / exp_o.sum(dim=0)).sum() - torch.log(exp_o.diag() / exp_o.sum(dim=1)).sum() \
#                  - torch.log(exp_r.diag() / exp_r.sum(dim=0)).sum() - torch.log(exp_r.diag() / exp_r.sum(dim=1)).sum()
#
#   # alternative way of computing the loss where the unknown feature part of the examples from the other database are treated as negative examples
#   if False:
#     cos_sim_combined = torch.concat(
#       [torch.concat([logits_o, logits_or], dim=1), torch.concat([logits_ro, logits_r], dim=1)], dim=0)
#     exp_comb = torch.exp(cos_sim_combined)
#     batch_loss = - torch.log(exp_comb.diag() / exp_comb.sum(dim=0)).sum() - torch.log(
#       exp_comb.diag() / exp_comb.sum(dim=1)).sum()
#
#   return batch_loss

def NTXentLoss(embeddings_knw_o, embeddings_knw_r, embeddings_unknw_o, embeddings_unknw_r,
                 temperature=0.1):  # embeddings from known features of both databases followed by the unknown features
    # compute the cosine similarity bu first normalizing and then matrix multiplying the known and unknown tensors
    cos_sim_o = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
                          temperature)
    cos_sim_or = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
                                        torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
                           temperature)
    cos_sim_r = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
                          temperature)
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
      batch_loss_o = - torch.log(exp_o.diag() / exp_o.sum(dim=0)).sum() - torch.log(
        exp_o.diag() / exp_o.sum(dim=1)).sum()
      batch_loss_r = - torch.log(exp_r.diag() / exp_r.sum(dim=0)).sum() - torch.log(
        exp_r.diag() / exp_r.sum(dim=1)).sum()
      # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
      # since we are computing the rank on the similarity so higher the better
      avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_o.cpu().detach().numpy(), axis=1)) / len(cos_sim_o)
      avg_rank_cos_sim_r = np.trace(stats.rankdata(cos_sim_r.cpu().detach().numpy(), axis=1)) / len(cos_sim_r)

    # alternative way of computing the loss where the unknown feature part of the examples from the other database are treated as negative examples
    if False:
      cos_sim_combined = torch.concat(
        [torch.concat([logits_o, logits_or], dim=1), torch.concat([logits_ro, logits_r], dim=1)], dim=0)
      exp_comb = torch.exp(cos_sim_combined)
      batch_loss = - torch.log(exp_comb.diag() / exp_comb.sum(dim=0)).sum() - torch.log(
        exp_comb.diag() / exp_comb.sum(dim=1)).sum()
      # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
      # since we are computing the rank on the similarity so higher the better
      # breakpoint()
      avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_combined.cpu().detach().numpy(), axis=1)) / len(
        cos_sim_combined)
      avg_rank_cos_sim_r = avg_rank_cos_sim_o
      batch_loss_o = batch_loss
      batch_loss_r = batch_loss

    # print("This batch's loss and avg rank ", batch_loss_o.item(), batch_loss_r.item(), avg_rank_cos_sim_o, avg_rank_cos_sim_r)
    return batch_loss_o, batch_loss_r, avg_rank_cos_sim_o, avg_rank_cos_sim_r

def main(dataset_no_sample):

  #breakpoint()
  if dataset_number not in  ['Nomao', 'superconductor']:
      # reading the data
      if dataset_number in ['1','2']:
        filename = data_dir + "SD_" + str(dataset_number) + "/2021-05-18Syn_Data_" + str(
          dataset_number) + "_Sample_no_" + str(
          dataset_no_sample) + "_size_20_10000_for_AE_balanced.csv"  # for dataset 1 and 2
      else:
        filename = data_dir + "SD_" + str(dataset_number) + "/2021-05-21Syn_Data_" + str(dataset_number) +  "_Sample_no_" + str(dataset_no_sample) +"_size_50_10000_for_AE_balanced.csv"  # for dataset 5
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

      ## final columns that were selected including the label and exclusing the first column that was the id
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

  full_Data0['Y'] = np.where(full_Data0['Y'] == -1, 0, 1)  # to make it compatible with xgbt during evaluation

  upto_test_idx = int(testDatasetSize * len(full_Data0))
  test = full_Data0.iloc[:upto_test_idx] # this part of the dataset is not touched during HP tuning


  full_Data = full_Data0.iloc[upto_test_idx:] # this is the dataset which will be divided into two parts for hp tuning
  num_sample = full_Data.shape[0]
  num_features = full_Data.shape[1] - 1

  # full data initial correlation
  Feature_matrix = full_Data.iloc[:, :-1]
  Cor_from_df = Feature_matrix.corr()

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

  m = 0  # variables to keep track of the iterations over number of mapped features
  for mpfeatures in list_of_number_mapped_variables:
    run_num = 0  # variable to keep track of the run number out of n_t*n_p
    print("\n ********************************************************")
    print("Run when there are ", mpfeatures, " mapped features starts")
    print(" ******************************************************** \n")

    for trial in range(n_t):

      # array for saving the frac of mistakes
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

      # array for saving F1 scores
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


      # the copies are being made because of the multiple trials
      df_train = full_Data.copy()

      print("\n ********************************************************")
      print("Trial number ", trial + 1, "   starts when there are ", mpfeatures, " mapped features")
      print(" ******************************************************** \n")

      # shuffling the mapped and unmapped
      mapped_random = np.random.choice(num_features, mpfeatures, replace=False)
      unmapped_random = list(set(np.arange(num_features)) - set(mapped_random))

      mapped_features = ["Col" + str(i + 1) for i in mapped_random]
      possible_feat_to_shuffle = ["Col" + str(i + 1) for i in unmapped_random]

      print("List of mapped features for trial number", trial + 1, "is ", mapped_features)

      # reordering to making sure that mapped features are at the starts of the vector
      feature_names = mapped_features + possible_feat_to_shuffle
      df_train = df_train.reindex(columns=feature_names + [outcome])

      # keeping a holdout sample aside
      Df_for_training, Df_holdout = model_selection.train_test_split(df_train, test_size=0.1,
                                                                     random_state=42 * trial * 10,
                                                                     stratify=df_train[outcome])

      # splitting the holdout df into two for using in the two databases
      Df_holdout_orig0, DF_holdout_r0 = model_selection.train_test_split(Df_holdout, test_size=0.5, random_state=42,
                                                                         stratify=Df_holdout[outcome])

      df_train1, df_train2 = model_selection.train_test_split(Df_for_training, test_size=frac_renamed,
                                                              random_state=42 * trial * 10,
                                                              stratify=Df_for_training[outcome])

      print(" trial data details \n")
      print("size of total train", len(df_train))
      print("size of train original", len(df_train1))
      print("size of train renamed", len(df_train2))

      device = torch.device('cpu')

      """ ORIGINAL DATA PREP """
      # data pre-processing normalization
      normalizing_values_orig = {}
      normalizing_values_orig['mean'] = df_train1[feature_names].mean(axis=0)
      normalizing_values_orig['std'] = df_train1[feature_names].std(axis=0)
      normalizing_values_orig['min'] = df_train1[feature_names].min(axis=0)
      normalizing_values_orig['max'] = df_train1[feature_names].max(axis=0)

      df_train_preproc0 = normalization(df_train1, 'mean_std', normalizing_values_orig, feature_names)
      Df_holdout_orig0 = normalization(Df_holdout_orig0, 'mean_std', normalizing_values_orig, feature_names)
      reordered_column_names_orig = mapped_features + [col for col in df_train_preproc0.columns if
                                                       col not in mapped_features + [outcome]] + [outcome]
      df_train_preproc0 = df_train_preproc0.reindex(columns=reordered_column_names_orig)
      Df_holdout_orig0 = Df_holdout_orig0.reindex(columns=reordered_column_names_orig)

      """ SHUFFLED FEATURES DATA PREP """
      feature_names_r = feature_names

      sq_transf_features = []
      # square transformation on *num_feat_sq_trans* randomly selected unmapped variables
      if num_feat_sq_trans > 0:
          feat_index_for_trans = np.random.choice(np.arange(mpfeatures, num_features), num_feat_sq_trans,
                                                  replace=False)

          for j in feat_index_for_trans:
              df_train2.iloc[:, j] = df_train2.iloc[:, j] * df_train2.iloc[:, j]
              DF_holdout_r0.iloc[:, j] = DF_holdout_r0.iloc[:, j] * DF_holdout_r0.iloc[:, j]
          print("\n  Type of transformation is non 1-1 like square")
          sq_transf_features = [list(df_train2.columns)[i] for i in feat_index_for_trans]
          print("Transformed feature names for dataset rename is ",
                sq_transf_features)

      # data preprocessing
      normalizing_values_r = {}
      normalizing_values_r['mean'] = df_train2[feature_names_r].mean(axis=0)
      normalizing_values_r['std'] = df_train2[feature_names_r].std(axis=0)
      normalizing_values_r['min'] = df_train2[feature_names_r].min(axis=0)
      normalizing_values_r['max'] = df_train2[feature_names_r].max(axis=0)

      df_rename_preproc0 = normalization(df_train2, 'mean_std', normalizing_values_r, feature_names_r)
      DF_holdout_r0 = normalization(DF_holdout_r0, 'mean_std', normalizing_values_r, feature_names_r)

      if datatype == 'b':  # """ thresholding all feature values at 0 to binarize the data  """
        for i in list(df_train_preproc0.columns):
          df_train_preproc0.loc[df_train_preproc0[i] > 0, i] = 1
          df_train_preproc0.loc[df_train_preproc0[i] < 0, i] = 0
          Df_holdout_orig0.loc[Df_holdout_orig0[i] > 0, i] = 1
          Df_holdout_orig0.loc[Df_holdout_orig0[i] < 0, i] = 0

        for i in list(df_rename_preproc0.columns):
          df_rename_preproc0.loc[df_rename_preproc0[i] > 0, i] = 1
          df_rename_preproc0.loc[df_rename_preproc0[i] < 0, i] = 0
          DF_holdout_r0.loc[DF_holdout_r0[i] > 0, i] = 1
          DF_holdout_r0.loc[DF_holdout_r0[i] < 0, i] = 0

      # maximum possible mistakes for this trial
      max_mistakes = len(feature_names) - len(mapped_features)

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
        reorder_feat = possible_feat_to_shuffle.copy()
        random.shuffle(reorder_feat)
        index_for_mapping_orig_to_rename = [reorder_feat.index(num) + len(mapped_features) + 1 for num
                                            in
                                            [col for col in df_train_preproc.columns if
                                             col not in mapped_features + [outcome]]]
        index_for_mapping_rename_to_orig = [[col for col in df_train_preproc.columns if
                                             col not in mapped_features + [outcome]].index(num) + len(
          mapped_features) + 1 for num in reorder_feat]

        # adding index variables for the mapped variables at the start of the list
        index_for_mapping_orig_to_rename = list(
          np.arange(1, mpfeatures + 1)) + index_for_mapping_orig_to_rename
        index_for_mapping_rename_to_orig = list(
          np.arange(1, mpfeatures + 1)) + index_for_mapping_rename_to_orig
        print(" Index for mapping orig to rename ", index_for_mapping_orig_to_rename)
        print(" Index for mapping rename to original ", index_for_mapping_rename_to_orig)

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
              P_x1[i, j] = 1

        """ AE part preprocessing  starts   """


        Cor_X1_map_unmap, Cor_X2_map_unmap, Mistakes_X1_tr_sim_Cor, Mistakes_X2_tr_sim_Cor, F1_x1_sim_cor, F1_x2_sim_cor = Train_KMF(
          df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename,
          index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r, mapped_features,
          Cor_from_df, Df_holdout_orig,
          DF_holdout_r, P_x1[len(mapped_features):-1,
                                            len(mapped_features):-1])


        _, _, Mistakes_X1_tr_CL, Mistakes_X2_tr_CL, F1_x1_CL, F1_x2_CL, _, _ = Train_CL(
          df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename,
          index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r,
          mapped_features, Cor_from_df, Df_holdout_orig,
          DF_holdout_r, normalizing_values_orig, normalizing_values_r,
          P_x1[len(mapped_features):-1, len(mapped_features):-1])

        grad_sum_unkn_o, grad_sum_unkn_r, Mistakes_X1_tr_CL_Dec, Mistakes_X2_tr_CL_Dec, F1_x1_CL_Dec, F1_x2_CL_Dec, rep_known_val_o, rep_known_val_r, _, _ = Train_CL_withDec(
          df_train_preproc.copy(), df_rename_preproc.copy(), index_for_mapping_orig_to_rename,
          index_for_mapping_rename_to_orig, reordered_column_names_orig, reordered_column_names_r,
          mapped_features, Cor_from_df, Df_holdout_orig,
          DF_holdout_r, normalizing_values_orig, normalizing_values_r,
          P_x1[len(mapped_features):-1, len(mapped_features):-1])

        Mistakes_X1_tr_CL_KMFl, Mistakes_X2_tr_CL_KMFl, F1_x1_CL_CL_KMFl, F1_x2_CL_CL_KMFl = CL_with_KMF_linear(
          grad_sum_unkn_o, grad_sum_unkn_r, Cor_X1_map_unmap,
          Cor_X2_map_unmap, index_for_mapping_orig_to_rename,
          index_for_mapping_rename_to_orig
          , reordered_column_names_orig, reordered_column_names_r,
          mapped_features, P_x1[len(mapped_features):-1, len(mapped_features):-1])

        # breakpoint()
        Mistakes_X1_tr_CL_KMFen, Mistakes_X2_tr_CL_KMFen, F1_x1_CL_CL_KMFen, F1_x2_CL_CL_KMFen = CL_with_KMF_CLencoded(
          grad_sum_unkn_o, grad_sum_unkn_r, rep_known_val_o, rep_known_val_r,
          Df_holdout_orig, DF_holdout_r,
          index_for_mapping_orig_to_rename,
          index_for_mapping_rename_to_orig
          , reordered_column_names_orig, reordered_column_names_r,
          mapped_features, P_x1[len(mapped_features):-1, len(mapped_features):-1])


        Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = len(Mistakes_X1_tr_sim_Cor) / max_mistakes
        Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = len(Mistakes_X2_tr_sim_Cor) / max_mistakes

        Frac_mismatches_across_trial_perm_X1_tr_sim_Cor[m, run_num] = len(Mistakes_X1_tr_sim_Cor) / max_mistakes
        Frac_mismatches_across_trial_perm_X2_tr_sim_Cor[m, run_num] = len(Mistakes_X2_tr_sim_Cor) / max_mistakes


        Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = len(Mistakes_X1_tr_CL) / max_mistakes
        Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = len(Mistakes_X2_tr_CL) / max_mistakes

        Frac_mismatches_across_trial_perm_X1_tr_CL[m, run_num] = len(Mistakes_X1_tr_CL) / max_mistakes
        Frac_mismatches_across_trial_perm_X2_tr_CL[m, run_num] = len(Mistakes_X2_tr_CL) / max_mistakes

        Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = len(Mistakes_X1_tr_CL_Dec) / (
          max_mistakes)
        Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = len(Mistakes_X2_tr_CL_Dec) / (
          max_mistakes)

        Frac_mismatches_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = len(Mistakes_X1_tr_CL_Dec) / (max_mistakes)
        Frac_mismatches_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = len(Mistakes_X2_tr_CL_Dec) / (max_mistakes)

        Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition] = len(Mistakes_X1_tr_CL_KMFl) / max_mistakes
        Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition] = len(Mistakes_X2_tr_CL_KMFl) / max_mistakes

        Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_l[m, run_num] = len(Mistakes_X1_tr_CL_KMFl) / max_mistakes
        Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_l[m, run_num] = len(Mistakes_X2_tr_CL_KMFl) / max_mistakes

        Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition] = len(
          Mistakes_X1_tr_CL_KMFen) / max_mistakes
        Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition] = len(
          Mistakes_X2_tr_CL_KMFen) / max_mistakes

        Frac_mismatches_across_trial_perm_X1_tr_CL_KMF_enc[m, run_num] = len(Mistakes_X1_tr_CL_KMFen) / max_mistakes
        Frac_mismatches_across_trial_perm_X2_tr_CL_KMF_enc[m, run_num] = len(Mistakes_X2_tr_CL_KMFen) / max_mistakes


        F1_for_fixed_trial_fixed_num_mapped_X1_tr_sim_Cor[partition] = F1_x1_sim_cor
        F1_for_fixed_trial_fixed_num_mapped_X2_tr_sim_Cor[partition] = F1_x2_sim_cor

        F1_across_trial_perm_X1_tr_sim_Cor[m, run_num] = F1_x1_sim_cor
        F1_across_trial_perm_X2_tr_sim_Cor[m, run_num] = F1_x2_sim_cor

        F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL[partition] = F1_x1_CL
        F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL[partition] = F1_x2_CL

        F1_across_trial_perm_X1_tr_CL[m, run_num] = F1_x1_CL
        F1_across_trial_perm_X2_tr_CL[m, run_num] = F1_x2_CL

        F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec[partition] = F1_x1_CL_Dec
        F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec[partition] = F1_x2_CL_Dec

        F1_across_trial_perm_X1_tr_CL_with_Dec[m, run_num] = F1_x1_CL_Dec
        F1_across_trial_perm_X2_tr_CL_with_Dec[m, run_num] = F1_x2_CL_Dec

        F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_l[partition] = F1_x1_CL_CL_KMFl
        F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_l[partition] = F1_x2_CL_CL_KMFl

        F1_across_trial_perm_X1_tr_CL_KMF_l[m, run_num] = F1_x1_CL_CL_KMFl
        F1_across_trial_perm_X2_tr_CL_KMF_l[m, run_num] = F1_x2_CL_CL_KMFl

        F1_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc[partition] = F1_x1_CL_CL_KMFen
        F1_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc[partition] = F1_x2_CL_CL_KMFen

        F1_across_trial_perm_X1_tr_CL_KMF_enc[m, run_num] = F1_x1_CL_CL_KMFen
        F1_across_trial_perm_X2_tr_CL_KMF_enc[m, run_num] = F1_x2_CL_CL_KMFen


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
        AVG_MISMATCHES_X1_tr_CL_with_Dec[m, trial] = np.average(
          Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_with_Dec)
      if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec != []:
        AVG_MISMATCHES_X2_tr_CL_with_Dec[m, trial] = np.average(
          Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_with_Dec)

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
        AVG_MISMATCHES_X1_tr_CL_KMF_enc[m, trial] = np.average(
          Mistakes_for_fixed_trial_fixed_num_mapped_X1_tr_CL_KMF_enc)
      if Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc != []:
        AVG_MISMATCHES_X2_tr_CL_KMF_enc[m, trial] = np.average(
          Mistakes_for_fixed_trial_fixed_num_mapped_X2_tr_CL_KMF_enc)

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

parser = argparse.ArgumentParser(description='HP for CL optimization')

## for the Encoders width (Known mapped)
parser.add_argument("--encKnwMapDepth",  default=3, type=int) #
parser.add_argument("--encKnwMapWidth",  default=80, type=int) #
parser.add_argument("--encKnwMapWidthFinal",  default=15, type=int) # this should be same for all three encoders by design
parser.add_argument("--encKnwMapL2",  default=0.2, type=float)
parser.add_argument("--encKnwMapL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset1 (orig))
parser.add_argument("--encUknwD1OrigDepth",  default=3, type=int) #
parser.add_argument("--encUknwD1OrigWidth",  default=80, type=int) #
parser.add_argument("--encUknwD1OrigWidthFinal",  default=None, type=int) # setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD1OrigL2",  default=0.2, type=float)
parser.add_argument("--encUknwD1OrigL1",  default=0.1, type=float)

## for the Encoders width (Unmapped dataset2 (r))
parser.add_argument("--encUknwD2ReDepth",  default=3, type=int) #
parser.add_argument("--encUknwD2ReWidth",  default=80, type=int) #
parser.add_argument("--encUknwD2ReWidthFinal",  default=None, type=int) ## setting this as none and assigning the value from known_mapped in a bit
parser.add_argument("--encUknwD2ReL2",  default=0.2, type=float)
parser.add_argument("--encUknwD2ReL1",  default=0.1, type=float)

## learning parameters
parser.add_argument("--batchSize",  default=64, type=int) #
parser.add_argument("--learningRate",  default=1e-3, type=float) #
parser.add_argument("--learningRateFactor",  default=0.1, type=float) #
parser.add_argument("--LRPatience",  default=2, type=int) #
parser.add_argument("--epochs",  default=1, type=int) #

## CL specific parameters
parser.add_argument("--tau", default=0.1, type=float)  # temperature parameter in CL loss
parser.add_argument("--masking_ratio", default=0.1, type=float)  # ratio for creating a new self augmented view
parser.add_argument("--mp_features", default=2, type=int)  # number of mapped_features
parser.add_argument("--block_stand_comb", default=0, type=int)  # can set the type as bool too but for now int it is
parser.add_argument("--dropout_rate_CL", default=0.2, type=float)
# not using the following features for now
# parser.add_argument("--mp_features_start", default=2, type=float)  # minimum number of mapped_features
# parser.add_argument("--mp_features_end", default=10, type=float)  # maximum number of mapped_features
# parser.add_argument("--mp_features_steps", default=5, type=float)  # number of mp_features to test between start and stop; check compatibility with abbove two
parser.add_argument("--weightDirDecoder", default =0.5984, type=float)
parser.add_argument("--weightCrossDecoder", default =0.7571, type=float)
parser.add_argument("--weightCombDecoder", default =0.338, type=float)

## dataset and setup parameters
parser.add_argument("--dataset_number",  default=1) # could be integer or string
parser.add_argument("--outcome",  default="Y")  # this is not very relevant but kep for the sake of completeness
parser.add_argument("--frac_renamed", default=0.5, type=float)
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--testDatasetSize", default=0.2, type=float) # this top fraction of data is not used by this code
# parser.add_argument("--dataset_no_sample", default=1, type=int) # this is used to decide which one to tune the HPs on in case of synthetic;
parser.add_argument("--num_of_dataset_samples", default=1, type=int) # number of dataset instances to be used from one  distribution
parser.add_argument("--n_p", default=4, type=int) # number of permutations
parser.add_argument("--n_t", default=3, type=int) # number of data partitioning trials
parser.add_argument("--datatype", default='c') # either continous or binary
parser.add_argument("--num_feat_sq_trans", default=0, type=int)  # how many features in the second database are square transformed


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
np.random.seed(randomSeed)

# data_dir = "/research2/trips/Feature_confusion/data/"
data_dir = '/input/'

# list_of_number_mapped_variables = np.linspace(start=mp_features_start, stop=mp_features_end, num=mp_features_steps)
list_of_number_mapped_variables = [mp_features]

AVG_over_Dataset_samples_X1_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_sim_Cor = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))


AVG_over_Dataset_samples_X1_tr_CL = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

AVG_over_Dataset_samples_X1_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_with_Dec = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

AVG_over_Dataset_samples_X1_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_l = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))


AVG_over_Dataset_samples_X1_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))
AVG_over_Dataset_samples_X2_tr_CL_KMF_enc = np.zeros((len(list_of_number_mapped_variables), num_of_dataset_samples))

for sample_no in range(1,num_of_dataset_samples+1):
    print("\n ********************************************************")
    print(" \n Run STARTS for sample no ", sample_no, "  of dataset ", dataset_number, "\n")
    print(" ******************************************************** \n")

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


    print("\n ********************************************************")
    print(" \n Run ENDS for sample no ", sample_no, "  of dataset ", dataset_number, "\n ")
    print(" ******************************************************** \n")

# Computing the average over the datset samples
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


for i in range(len(list_of_number_mapped_variables)):
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


print(" Mean F1 score values when # of mapped features is ", mp_features)
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
  'mp_features': args.mp_features,
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

# breakpoint()
