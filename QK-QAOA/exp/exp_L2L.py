import argparse
## our model
import L2L
import L2L_FWP
import Optim
import QKLSTM
import QLSTM
import FWP
import QAOA
## basis
import numpy as np
import pandas as pd
import pickle
## random
import random
import math
from typing import List, Callable, Tuple
## ML
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
## QML
import pennylane as qml
from pennylane import qaoa
import networkx as nx
## access file
import os

torch.manual_seed(42)
np.random.seed(42)
qml.math.random.seed(42)
random.seed(42)

"""
Define data loading and hyperparameter
"""
parser = argparse.ArgumentParser(description='QAOA with sequence models training')
parser.add_argument('--dataset_save_path', type=str, default='datasets.pkl', help='Path to load the dataset')
parser.add_argument('--model_type', type=str, required=True, help='Model type to train (e.g., LSTM, QK, QLSTM, FWP)')
parser.add_argument('--mapping_type', type=str, default='Linear', help='Mapping type (e.g., Linear, ID)')
parser.add_argument('--layers', type=int, default=1, help='Number of sequence model layers')
parser.add_argument('--input_feature_dim', type=int, default=4, help='Input feature dimension for the model')
parser.add_argument('--max_total_params', type=int, default=4, help='Max total parameters for QAOA ansatz')
parser.add_argument('--loss_function_type', type=str, default='weighted', help='Loss function type (e.g., weighted, observed improvement)')
parser.add_argument('--qaoa_layers', type=int, default=2, help='Number of layers for QAOA')
parser.add_argument('--lr_sequence', type=float, default=8e-5, help='Learning rate for sequence model')
parser.add_argument('--lr_mapping', type=float, default=1e-4, help='Learning rate for mapping layer')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--steps_recurrent_loop_train', type=int, default=10, help='Number of recurrent steps during training')
parser.add_argument('--conv_tol_sequence', type=float, default=1e-5, help='Convergence tolerance for training')
parser.add_argument('--model_save_path', type=str, default='models_default', help='Path to save the trained model')
parser.add_argument('--time_out', type=int, default=2*60*60, help='Timeout in seconds for training')
parser.add_argument('--continue_train', type = bool,default = False, help='whether continue training from existing model')
parser.add_argument('--load_path', type=str, default=None, help='Path to load a pre-trained model')
parser.add_argument('--qaoa_optimizer', type=str, default='Adam', help='Optimizer for QAOA optimization e.g. ADAM or SGD')
parser.add_argument('--lr_qaoa', type=float, default=0.01, help='Learning rate for QAOA optimization')
parser.add_argument('--max_iter_qaoa', type=int, default=300, help='Max iterations for QAOA optimization')
parser.add_argument('--conv_tol_qaoa', type=float, default=1e-6, help='Convergence tolerance for QAOA optimization')
args = parser.parse_args()


"""
Data loading 
"""
with open(args.dataset_save_path, 'rb') as f:
    loaded_datasets = pickle.load(f)
    train_set = loaded_datasets['train_set']
    val_set = loaded_datasets['val_set']
    test_set = loaded_datasets['test_set']

"""
Define model and training
"""
# --- Model Training ---
def build_and_train_model(model_type,
                          mapping_type,
                          layers, 
                          input_feature_dim,
                          max_total_params,
                          loss_function_type,
                          qaoa_layers,
                          lr_sequence,
                          lr_mapping,
                          epochs, 
                          steps_recurrent_loop_train,
                          conv_tol_sequence,
                          Model_save_path,
                          train_set,
                          val_set,
                          time_out,
                          continue_train,
                          load_path):
    """
    Define model and then training
         Args:
          model_type (str): the model type. LSTM, QKLSTM, QKLSTM, or FWP
          mapping_type (str): the mapping model type, Linear (FC) or ID
          layers [int]: the number of layers for LSTM, QKLSTM, QLSTM or FWP
          input_feature_dim [int]: the number of params for LSTM output to QAOA ansatz
          max_total_params [int]: the max numbers for QAOA we need for this experiment
          loss_function_type (str): the loss function type, weighted or observed improvement (define by Verdon's papaer)
          qaoa_layers [int]: the number of QAOA layers for ansatz
          lr_lstm [float]: learning rate for LSTM, QKLSTM, QLSTM or FWP
          lr_mapping [float]: learning rate for mapping model
          epochs [int]: the number of training epochs
          steps_recurrent_loop_train [int]: the number of recurrent step for training
          conv_tol_lstm [float]: the convergence tolerance for training
          Model_save_path (str): the path to save model's parameters
          train_set: the training dataset
          val_set: the validation dataset
          time_out [int]: the time out for training (in seconds)
          continue_train [bool]: whether continue training from existing model
          load_path (str): the path to load existing model's parameters

         Outputs:
          model: the trained model
          trainer: the trainer object
    """
    if model_type in ["LSTM", "QK", "QLSTM"]:
        model = L2L.L2L(model_type = model_type,
                        mapping_type= mapping_type,
                        layers = layers,
                        input_feature_dim = input_feature_dim,
                        max_total_params = max_total_params,
                        loss_function_type = loss_function_type,
                        )
    elif model_type == "FWP":
        model = L2L_FWP.L2L_FWP(mapping_type = mapping_type,
                                 layers=layers,
                                 input_feature_dim=input_feature_dim,
                                 max_total_params=max_total_params,
                                 loss_function_type=loss_function_type,
                                 )
    
    if continue_train == True:
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
        print("load params successful!")
    
    print(f"--- Model Summary ---")
    print(model)
    if model_type == "FWP":
        model_params = sum(p.numel() for p in model.fwp.parameters())
    else: 
        model_params = sum(p.numel() for p in model.lstm.parameters())
    print(f"  {model_type} Parameters: {model_params}")
    
    mapping_params = sum(p.numel() for p in model.mapping.parameters())
    print(f"  Mapping Parameters: {mapping_params}")

    trainer = Optim.ModelTrain(model = model,
                               qaoa_layers = qaoa_layers,
                               lr_lstm = lr_sequence,
                               lr_mapping= lr_mapping,
                               num_rnn_iteration = steps_recurrent_loop_train,
                               )
    
    print(f"\n--- Training {model_type} Model ---")

    trainer.train(train_data = train_set,
                  val_data = val_set,
                  epochs = epochs,
                  conv_tol_lstm = conv_tol_sequence,
                  time_out = time_out,
                  save_path = Model_save_path,
                  )
    
    torch.save(model.state_dict(), f"{Model_save_path}_{model_type}_{lr_sequence}_{lr_mapping}.pth")
    print("Model saved successfully!")
    
    return model, trainer

build_and_train_model(
    model_type = args.model_type,
    mapping_type = args.mapping_type,
    layers = args.layers,
    input_feature_dim = args.input_feature_dim,
    max_total_params = args.max_total_params,
    loss_function_type = args.loss_function_type,
    qaoa_layers = args.qaoa_layers,
    lr_sequence = args.lr_sequence,
    lr_mapping = args.lr_mapping,
    epochs = args.epochs,
    steps_recurrent_loop_train = args.steps_recurrent_loop_train,
    conv_tol_sequence = args.conv_tol_sequence,
    Model_save_path = args.model_save_path,
    train_set = train_set,
    val_set = val_set,
    time_out = args.time_out,
    continue_train = args.continue_train,
    load_path = args.load_path
)
                          