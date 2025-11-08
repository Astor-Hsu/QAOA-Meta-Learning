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
import networkx as nx
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
# --- Training/Data Arguments ---
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
# --- QAOA Optimization Arguments ---
parser.add_argument('--qaoa_optimizer', type=str, default='ADAM', help='Optimizer for QAOA optimization e.g. ADAM or SGD')
parser.add_argument('--lr_qaoa', type=float, default=0.01, help='Learning rate for QAOA optimization')
parser.add_argument('--max_iter_qaoa', type=int, default=300, help='Max iterations for QAOA optimization')
parser.add_argument('--conv_tol_qaoa', type=float, default=1e-6, help='Convergence tolerance for QAOA optimization')
# --- Testing Arguments ---
parser.add_argument('--steps_recurrent_loop_test', type=int, default=10, help='Number of recurrent steps during testing (Phase I)')
parser.add_argument('--Results_save_path', type=str, default='results_default', help='Path to save the test results')

def parse_arguments():
    return parser.parse_args()

"""
Define model and training
"""
# --- Model Training ---
def build_and_train_model(args,
                          train_set,
                          val_set,
                          ):
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
    model_type = args.model_type
    mapping_type = args.mapping_type
    layers = args.layers
    input_feature_dim = args.input_feature_dim
    max_total_params = args.max_total_params
    loss_function_type = args.loss_function_type
    qaoa_layers = args.qaoa_layers
    lr_sequence = args.lr_sequence
    lr_mapping = args.lr_mapping
    epochs = args.epochs
    steps_recurrent_loop_train = args.steps_recurrent_loop_train
    conv_tol_sequence = args.conv_tol_sequence
    Model_save_path = args.model_save_path
    time_out = args.time_out
    continue_train = args.continue_train
    load_path = args.load_path

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
        model_params = sum(p.numel() for p in model.sequence.parameters())
    print(f"  {model_type} Parameters: {model_params}")
    
    mapping_params = sum(p.numel() for p in model.mapping.parameters())
    print(f"  Mapping Parameters: {mapping_params}")

    trainer = Optim.ModelTrain(model = model,
                               qaoa_layers = qaoa_layers,
                               lr_sequence = lr_sequence,
                               lr_mapping= lr_mapping,
                               num_rnn_iteration = steps_recurrent_loop_train,
                               )
    
    print(f"\n--- Training {model_type} Model ---")

    trainer.train(train_data = train_set,
                  val_data = val_set,
                  epochs = epochs,
                  conv_tol_sequence = conv_tol_sequence,
                  time_out = time_out,
                  save_path = Model_save_path,
                  )
    
    torch.save(model.state_dict(), f"{Model_save_path}_{model_type}_{lr_sequence}_{lr_mapping}.pth")
    print("Model saved successfully!")
    
    return model, trainer

# Model Testing
def run_experiment(args):

    # Data loading 
    with open(args.dataset_save_path, 'rb') as f:
        loaded_datasets = pickle.load(f)
        train_set = loaded_datasets['train_data']
        val_set = loaded_datasets['val_data']
        test_set = loaded_datasets['test_data']
    print(f"Datasets loaded from {args.dataset_save_path}")
    print(f"Train = {len(train_set)} samples, Val = {len(val_set)} samples, Test = {len(test_set)} samples")
    print(f"The first train graph has {len(train_set[0].nodes)} nodes and {len(train_set[0].edges)} edges.")
    nx.draw(train_set[0])

    print("\n--- Building and Training Model ---")
    
    model, trainer = build_and_train_model(
        args = args,
        train_set = train_set,
        val_set = val_set,
        )
    
    if args.model_type in ["LSTM", "QK", "QLSTM"]:
        model = L2L.L2L(model_type = args.model_type,
                       mapping_type= args.mapping_type,
                       layers = args.layers,
                       input_feature_dim = args.input_feature_dim,
                       max_total_params = args.max_total_params,
                       loss_function_type = args.loss_function_type,
                       )
    
    elif args.model_type == "FWP":
        model = L2L_FWP.L2L_FWP(mapping_type= args.mapping_type,
                                layers = args.layers,
                                input_feature_dim = args.input_feature_dim,
                                max_total_params = args.max_total_params,
                                loss_function_type = args.loss_function_type,
                                )
        
    state_dict = torch.load(f"best_{args.model_type}_model_{args.model_save_path}.pth")
    model.load_state_dict(state_dict)
    print(f"\n Successfully loaded best model")
    trainer = Optim.ModelTrain(model = model,
                               qaoa_layers = args.qaoa_layers,
                               lr_sequence = args.lr_sequence,
                               lr_mapping = args.lr_mapping,
                               num_rnn_iteration = args.steps_recurrent_loop_train,
                               )
    
    print(f"\n--- Evaluating Model ---")
    print(f"The first test graph has {len(test_set[0].nodes)} nodes and {len(test_set[0].edges)} edges.")
    nx.draw(test_set[0])
    
    for i in range(len(test_set)):
        graph_test_result = {}
        test_graph = test_set[i]
    
        sequence_predicted_params_list, sequence_predicted_energies_list = trainer.evaluate(
            graph_data = test_graph,
            num_rnn_iteration = args.steps_recurrent_loop_test)
    
        print(f"\n--- Test Graph {i+1}/{len(test_set)} (Nodes: {len(test_graph.nodes)}, Edges: {len(test_graph.edges)}) ---")
        print(f"{args.model_type} predicted energies:{sequence_predicted_params_list}")
        print(f"{args.model_type} predicted params:{sequence_predicted_energies_list[-1]}")
       
        # use sequence model output as initial params for QAOA to optimize
        print(f"\n--- QAOA optimization after model (Phase II) ---")
        equence_qaoa = QAOA.QAOA(graph = test_graph, 
                                 n_layers = args.qaoa_layers, 
                                 with_meta =  True)
        
        opt_sequence_qaoa = QAOA.QAOAptimizer(sequence_qaoa)
        conv_iter_sequence, final_params_sequence, final_energy_sequence, params_history_sequence, cost_history_sequence = opt_sequence_qaoa.run_optimization(
            initial_params = sequence_predicted_params_list[-1],
            optimizer = args.qaoa_optimizer,
            max_iter = args.max_iter_qaoa,
            learning_rate = args.lr_qaoa,
            conv_tol = args.conv_tol_qaoa
            )
    
        sequence_qaoa_params = np.array([p.detach().numpy() if hasattr(p, "detach") else p for p in params_history_sequence])
        np.savez(f"{self.model.model_type}_QAOA_node_{len(test_graph.nodes)}_edge_{len(test_graph.edges)}.npz", params = sequence_qaoa_params)
    
        print(f"\n--- Standard QAOA, Random params ---")
        params_rand = torch.rand(args.input_feature_dim, dtype = torch.float32)
        qaoa_test_rand = QAOA.QAOA(graph = test_graph, 
                                   n_layers = args.qaoa_layers, 
                                   with_meta =  False)
        
        opt_rand_qaoa = QAOA.QAOAptimizer(qaoa_test_rand)
        conv_iter_rand, final_params_rand, final_energy_rand, params_history_rand, cost_history_rand = opt_rand_qaoa.run_optimization(
            initial_params = params_rand,
            optimizer = args.qaoa_optimizer,
            max_iter = args.max_iter_qaoa,
            learning_rate = args.lr_qaoa,
            conv_tol = args.conv_tol_qaoa
            )
        
        qaoa_params_rand = np.array([p.detach().numpy() if hasattr(p, "detach") else p for p in params_history_rand])
        np.savez(f"QAOA_Random_node_{len(test_graph.nodes)}_edge_{len(test_graph.edges)}_{i}.npz", params = qaoa_params_rand)

        # Draw result
        print("Result of MaxCut QAOA")
        plt.figure(figsize = (15,8))
        font = {'size':16}
        plt.rc('font', **font)
        plt.plot(np.arange(0, len(args.steps_recurrent_loop_test)), sequence_predicted_energies_list, label=f'{args.model_type}-QAOA', ls="dashed", color = "darkgreen", markersize = 9)
        plt.plot(np.arange(args.steps_recurrent_loop_test, args.steps_recurrent_loop_test + len(cost_history_sequence)), cost_history_sequence, label=f'QAOA after {args.model_type}', color = "darkgreen", markersize = 9)
        plt.plot(np.arange(0, len(args.conv_tol_qaoa)), cost_history_rand, label='QAOA, Random', color = "darkred", markersize = 9)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"num_node ={len(test_graph.nodes)}, num_edge = {len(test_graph.edges)}")
        plt.set_xlim([0-5, args.conv_tol_qaoa + 5])
        plt.legend()
        plt.show()

        graph_test_result = {
            'Phase I': pd.Series(sequence_predicted_energies_list),
            'Phase II':pd.Series(cost_history_sequence),
            'Random': pd.Series(cost_history_rand),
            }
                          
        df_result = pd.DataFrame(graph_test_result)
        df_result.to_csv(f"{args.Results_save_path}_node_{len(test_graph.nodes)}_{i}_edge_{len(test_graph.edges)}_{i}.csv", index = False)
        print("\n--- Saving Complete ---")

def main():
    args = parse_arguments()
    run_experiment(args)

if __name__=="__main__":
  main()
