#   @title Making and getting datasets

from datasets import make_data
from loss import mtlr_nll, cox_nll
from models import BayesianBaseModel
from models import mtlr_survival, mtlr, BayesHsLinMtlr, BayesEleMtlr, BayesHsMtlr, BayesLinMtlr, BayesMtlr
from models import cox_survival, CoxPH, BayesHsLinCox, BayesEleCox, BayesHsCox, BayesLinCox, BayesCox
from utils import (save_predictions, save_params, train_val_test_stratified_split, reformat_survival, print_performance,
                   NumericArrayLike, make_time_bins, two_sided_olshen)
from plots import plot_weights_dist, plot_curve_with_bar, plot_weights_hist
from hyper_params import load_parser, generate_parser
# Avoid importing SurvivalEVAL to prevent R/rpy2 dependency during fast runs
# from SurvivalEVAL.ci_evaluation import thickness, coverage
# from SurvivalEVAL import BaseEvaluator
import matplotlib.pyplot as plt
import os


#batchbald stuff:
import blackhc.project.script
from tqdm.auto import tqdm

import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

import torch
from torch.utils.data import Dataset, DataLoader




import math
import numpy as np
import argparse
import pandas as pd
from tqdm import trange
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

import random


def make_SUPPORT(args):
  from ucimlrepo import fetch_ucirepo
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import LabelEncoder, StandardScaler

  support2 = fetch_ucirepo(id=880)
  d = pd.DataFrame(support2.data['original'])
  d.head()
  d.rename(columns={'death': 'event'}, inplace=True)
  d.rename(columns={'age': 'time'}, inplace=True)
  data = d


  # List of columns to drop
  cols_to_drop = [
      "hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
      "adlp", "adls", "dzgroup", "sps", "aps", "surv2m", "surv6m", "prg2m",
      "prg6m", "dnr", "dnrday", "hday"
  ]

  # Drop the specified columns
  data = data.drop(columns=cols_to_drop)

  # Convert 'ca' to binary
  data["ca"] = (data["ca"] == "metastatic").astype(int)

  # Fill missing values
  fill_vals = {
      "alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01, "bun": 6.51,
      "wblc": 9, "urine": 2502, "edu": data["edu"].mean(),
      "ph": data["ph"].mean(), "glucose": data["glucose"].mean(),
      "scoma": data["scoma"].mean(), "meanbp": data["meanbp"].mean(),
      "hrt": data["hrt"].mean(), "resp": data["resp"].mean(),
      "temp": data["temp"].mean(), "sod": data["sod"].mean(),
      "income": data["income"].mode()[0], "race": data["race"].mode()[0]
  }
  data = data.fillna(fill_vals)

  # Encode categorical variables
  data['sex'] = data['sex'].map({'male': 1, 'female': 0})
  data['income'] = data['income'].map({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3})

  # Define columns to skip standardization
  skip_cols = ['event', 'time', 'sex', 'dzclass', 'race', 'diabetes', 'dementia', 'ca', 'income']

  # Store the original values of skip_cols
  original_values = {col: data[col].copy() for col in skip_cols}

  # Define columns to standardize
  cols_standardize = [col for col in data.columns if col not in skip_cols]

  # Standardize the appropriate columns
  scaler = StandardScaler()
  data_standardized = pd.DataFrame(scaler.fit_transform(data[cols_standardize]), columns=cols_standardize, index=data.index)

  # Combine standardized columns with original skip_cols
  data_final = pd.concat([data_standardized, pd.DataFrame(original_values)], axis=1)

  # Ensure 'event' is int type
  data_final['event'] = data_final['event'].astype(int)

  print("Columns after preprocessing:", data_final.columns.tolist())
  print("Columns standardized:", cols_standardize)
  print("Event column unique values:", data_final['event'].unique())

  data = data_final

  # one-hot encode categorical variables
  onehot_cols = ["dzclass", "race"]
  data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
  data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})


#   assert "time" in data.columns and "event" in data.columns, "The event time variable and censor indicator " \
#                                                             "variable is missing or need to be renamed."
#   args.n_features = data.shape[1]
#   args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#   args.device = "cuda" if torch.cuda.is_available() else "cpu"
#   device = torch.device(args.device)

#   #here...

#   num_iter = 10
#   args.num_epochs = 100 #change this
#   sizeofbatch = 200 #change this
#   print("the setting, num_iter,num_epochs,sizeofbatch  = ", num_iter,args.num_epochs,sizeofbatch)

#   path = save_params(args)

#   if 'true_time' in data.columns:
#       feature_names = data.drop(["time", "event", "true_time"], axis=1).columns.to_list()
#       num_features = args.n_features - 3
#   else:
#       feature_names = data.drop(["time", "event"], axis=1).columns.to_list()
#       num_features = args.n_features - 2

#   seed = args.seed
#   torch.manual_seed(seed)
#   np.random.seed(seed)

  data = data.astype(float)

  # data_train, _, data_test = train_val_test_stratified_split(data, stratify_colname='both',
  #                                                             frac_train=0.8, frac_test=0.2,
  #                                                             random_state=seed)

  # datapreserver = data.copy()

  data = data.drop(data.columns[14], axis=1)
  return data





import argparse
import sys

#Returns prepared dataset for SUPPORT, NACD, and MIMIC
def Get_Dataset(dataset):
#   if dataset == "NACD":
#     model_type = 'BayesianMTLR'
#   else:
#     model_type = 'BayesianLinearMTLR'


  model_type = 'BayesianMTLR'

  # Simulate command-line arguments
  sys.argv = ['run_models.py',
              '--dataset', dataset,
              '--model', model_type,  #not linearMTLR!!!
              '--lr', '0.00008',
              '--batch_method', 'batchbald_true']

  # Create the argument parser
  parser = argparse.ArgumentParser(description='Run models with specified parameters.')

  # Define your arguments here
  parser.add_argument('--dataset', type=str, choices=['Synthetic-I', 'Synthetic-II', 'Synthetic-III', 'SUPPORT', 'NACD', 'MIMIC'])
  parser.add_argument('--model', type=str, choices=['MTLR', 'BayesianHorseshoeLinearMTLR', 'BayesianElementwiseMTLR', 'BayesianHorseshoeMTLR', 'BayesianLinearMTLR', 'BayesianMTLR', 'CoxPH', 'BayesianHorseshoeLinearCox', 'BayesianElementwiseCox', 'BayesianHorseshoeCox', 'BayesianLinearCox', 'BayesianCox'])
  parser.add_argument('--lr', type=float, help='Learning rate')
  parser.add_argument('--batch_method', type=str, help='Batch method')

  # Parse the arguments
  args = parser.parse_args()

  # Accessing the arguments
  print(f"Dataset: {args.dataset}")
  print(f"Model: {args.model}")
  print(f"Learning Rate: {args.lr}")
  print(f"Batch Method: {args.batch_method}")

  args = generate_parser()

  my_r = 20
  args.seed = my_r
  np.random.seed(my_r)
  torch.manual_seed(my_r)
  seed = args.seed

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # torch.autograd.set_detect_anomaly(True)

  if args.dataset == 'SUPPORT':
    data = make_SUPPORT(args)
  else:
    data, coef = make_data(args.dataset)

  if(args.dataset == 'NACD'):
      data = data.rename(columns={'SURVIVAL':'time','CENSORED':'event'})

  assert "time" in data.columns and "event" in data.columns, "The event time variable and censor indicator " \
                                                              "variable is missing or need to be renamed."
  args.n_features = data.shape[1]
  args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  args.device = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(args.device)

#change this
  args.num_epochs = 100 #change this


  path = save_params(args)

  if 'true_time' in data.columns:
      feature_names = data.drop(["time", "event", "true_time"], axis=1).columns.to_list()
      num_features = args.n_features - 3
  else:
      feature_names = data.drop(["time", "event"], axis=1).columns.to_list()
      num_features = args.n_features - 2


  data = data.astype(float)

  # data_train, _, data_test = train_val_test_stratified_split(data, stratify_colname='both',
  #                                                             frac_train=0.8, frac_test=0.2,
  #                                                             random_state=seed)

  print("data.shape = ",data.shape)

#   datapreserver = data.copy()
#   return datapreserver,args
  return data,args