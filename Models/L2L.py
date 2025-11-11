import QAOA
import QKLSTM
import QLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class L2L(nn.Module):
    """
    Sequence-QAOA recurrent model
    """
    def __init__(self, model_type = "QK", mapping_type = "ID",layers = 1, input_feature_dim = 0, max_total_params = 0, 
                 loss_function_type = "weighted", device = 'cpu', backend = "lightning.qubit", qubits = 4):
        super(L2L, self).__init__()
        """
        Define model
         Args:
          model_type (str): the model type. LSTM, QKLSTM or QKLSTM
          mapping_type (str): the mapping model type, Linear (FC) or ID
          input_feature_dim [int]: the number of params for LSTM output to QAOA ansatz
          max_total_params [int]: the max numbers for QAOA we need for this experiment
          loss_function_type (str): the loss function type, weighted or observed improvement (define by Verdon's papaer)
          device (str): device for training, 'cpu' or 'cuda:0'
          backend (str): the quantum backend for QKLSTM or QLSTM
          qubits (int): the number of qubits for QKLSTM or QLSTM

         Outputs:
          self.sequence [model]: LSTM, QK-LSTM, QLSTM
          self.mapping [model]: linear model, ID 
        """
        self.model_type = model_type
        self.mapping_type = mapping_type
        self.input_feature_dim = input_feature_dim 
        self.max_total_params = max_total_params
        self.loss_function_type = loss_function_type
        self.layers = layers
        self.device = device
        self.backend = backend
        self.qubits = qubits
        
        """
         input_feature_dim + 1 (cost) ->  sequence model -> input_feature_dim -> linear model -> max_total_params
        """
        if self.model_type == "LSTM":
            self.sequence = nn.LSTM(input_size = self.input_feature_dim + 1,
                                hidden_size = self.input_feature_dim,
                                num_layers = self.layers,
                                batch_first = True).to(self.device)
        elif self.model_type == "QK":
            self.sequence = QKLSTM.QKLSTM(input_size = self.input_feature_dim+1,
                                          hidden_size = self.input_feature_dim+1, 
                                          n_qubits = self.qubits,
                                          n_qlayers = self.layers,
                                          backend = self.backend,
                                          device = self.device,)
        elif self.model_type == "QLSTM":
            self.sequence = QLSTM.QLSTM(input_size = self.input_feature_dim+1 ,
                                        hidden_size = self.input_feature_dim+1,
                                        n_qubits = self.qubits,
                                        n_qlayers = self.layers,
                                        backend = self.backend,
                                        device = self.device,
                                        )
        
        if self.mapping_type == "Linear":
            self.mapping = nn.Linear(self.input_feature_dim+1, self.max_total_params, bias = True).to(self.device)
        elif self.mapping_type == "ID":
          self.mapping = nn.Identity().to(self.device)
        

    def forward(self, graph_cost, num_rnn_iteration, intermediate_steps = False):

        current_cost = torch.zeros((1, 1), dtype = torch.float32).to(self.device)
        current_params = torch.zeros((1, self.input_feature_dim), dtype = torch.float32).to(self.device)

        if self.model_type == "QK":
            hidden_state_size = self.input_feature_dim + 1
        elif self.model_type == "QLSTM":
            hidden_state_size = self.input_feature_dim + 1
        else:
            hidden_state_size = self.input_feature_dim

        current_h = torch.zeros((self.sequence.num_layers, 1, hidden_state_size), dtype = torch.float32).to(self.device)
        current_c = torch.zeros((self.sequence.num_layers, 1, hidden_state_size), dtype = torch.float32).to(self.device)

        param_outputs = []
        cost_outputs = []

        for i in range(num_rnn_iteration):
            new_input = torch.cat([current_cost, current_params], dim = 1).unsqueeze(1)
            new_params, (new_h, new_c) = self.sequence(new_input, (current_h, current_c))
           
            new_params = new_params.squeeze(1) 
            
            params = self.mapping(new_params.squeeze(1)).to(self.device)
            
            _cost = graph_cost(params.squeeze(0).to(self.device))
            new_cost = torch.as_tensor(_cost, dtype=torch.float32, device=self.device).view(1,1)
            
            param_outputs.append(params)
            cost_outputs.append(new_cost)

            current_cost = new_cost
            if self.model_type == "QK":
                current_params = params 
            elif self.model_type == "QLSTM":
                current_params = params
            elif self.model_type == "LSTM":
                current_params = new_params
            current_h = new_h
            current_c = new_c

        # meta-loss function
        loss = 0.0

        if self.loss_function_type == "weighted":
            for t in range(len(cost_outputs)):
                coeff = 0.1*(t+1)
                loss += cost_outputs[t]*coeff
            loss = loss/len(cost_outputs)

        elif self.loss_function_type == "observed improvement":
            cost = torch.stack(cost_outputs).to(self.device)
            zero = torch.tensor([0.0])
            for t in range(1, len(cost_outputs)):
                f_j = torch.min(cost[:t])
                penalty = torch.minimum(cost[t]-f_j, zero)
                loss += penalty
            loss = loss/len(cost_outputs)

        if intermediate_steps:
            return param_outputs, loss
        else:
            return loss
