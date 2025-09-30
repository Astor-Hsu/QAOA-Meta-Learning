import VQE
import QKLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, input_shape, out_shape, hidden_dims = [16, 12]):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dims[0], bias=True)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=True)
        self.fc3 = nn.Linear(hidden_dims[1], out_shape, bias=True)
        self.dropout = nn.Dropout(p = 0.2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc3(x))
        return x
    
class LSTM(nn.Module):
    """
    LSTM-VQE recurrent model
    """
    def __init__(self, model_type = "QK", mapping_type = "Linear",layers = 1, input_feature_dim = 0, max_total_params = 0, max_qubits = 0, loss_function_type = "weighted", max_s_params = 0, max_d_params = 0):
        super(LSTM, self).__init__()
        """
        Define model
         Args:
          model_type [str]: the model type. LSTM or QK (QKLSTM)
          mapping_type [str]: the mapping model type, Linear or DNN
          input_feature_dim [int]: the number of params for LSTM output to UCCSD ansatz
          max_total_params [int]: the max numbers for VQE we need for this experiment
          max_qubits [int]: the max qubits we need for this experiment
          loss_function_type [str]: the loss function type, weighted or observed improvement (define by papaer)
          the below two args are for single and double linear model if you plan to use
          max_s_params [int]: the max single excitation params we need for this experiment
          max_d_params [int]: the max double excitation params we need for this experiment

         Outputs:
          self.lstm [model]: LSTM or QK
          self.mapping [model]: linear model, DNN or SD(single and double linear model)
        """
        self.model_type = model_type
        self.mapping_type = mapping_type
        self.input_feature_dim = input_feature_dim 
        self.max_total_params = max_total_params
        self.max_qubits = max_qubits
        self.loss_function_type = loss_function_type
        self.max_s_params = max_s_params
        self.max_d_params = max_d_params
        self.layers = layers
        
        """
         input_feature_dim + 1 (cost) -> LSTM or QK -> input_feature_dim -> linear model -> max_total_params
        """
        if self.model_type == "LSTM":
            self.lstm = nn.LSTM(input_size = self.input_feature_dim + 1 ,
                                hidden_size = self.input_feature_dim,
                                num_layers = self.layers,
                                batch_first = True)
        elif self.model_type == "QK":
            self.lstm = QKLSTM.QKLSTM(input_size = self.input_feature_dim + 1,
                               hidden_size = self.input_feature_dim,
                               n_qubits = 4,
                               n_qlayers = self.layers)
        
        if self.mapping_type == "Linear":
            self.mapping = nn.Linear(self.input_feature_dim, self.max_total_params, bias = True)
        elif self.mapping_type == "DNN":
            self.mapping = DNN(self.input_feature_dim, self.max_total_params)
        
        elif self.mapping_type == "DS":
            self.single_mapping = nn.Linear(input_feature_dim//2, self.max_s_params)
            self.double_mapping = nn.Linear(input_feature_dim//2, self.max_d_params)
        
        """
        If you plan to use two linear model after LSTM or QK, not just one, aka, DS:
        LSTM or QK -> input_feature_dim -> input_feature_dim//2 -> single mapping -> max_s_params
                           -> input_feature_dim//2 -> double mapping -> max_d_params
        """
        

    def forward(self, molecule_cost, num_rnn_iteration, intermediate_steps = False):

        current_cost = torch.zeros((1, 1), dtype = torch.float32).to(device)
        current_params = torch.zeros((1, self.input_feature_dim), dtype = torch.float32).to(device)
        current_h = torch.zeros((self.lstm.num_layers, 1, self.input_feature_dim), dtype = torch.float32).to(device)
        current_c = torch.zeros((self.lstm.num_layers, 1, self.input_feature_dim), dtype = torch.float32).to(device)

        param_outputs = []
        cost_outputs = []

        for i in range(num_rnn_iteration):
            new_input = torch.cat([current_cost, current_params], dim = 1).unsqueeze(1)
            new_params, (new_h, new_c) = self.lstm(new_input, (current_h, current_c))
            new_params = new_params.squeeze(1)
            
            if self.mapping_type == "DS":
                single_input = new_params[:, :self.input_feature_dim//2]
                double_input = new_params[:, self.input_feature_dim//2:]
                single_params = self.single_mapping(single_input)
                double_params = self.double_mapping(double_input)
                params = torch.cat([single_params, double_params], dim = 1)
            else:
                params = self.mapping(new_params.squeeze(1))

            _cost = molecule_cost(params.squeeze(0))
            new_cost = _cost.view(1,1).float()

            param_outputs.append(params)
            cost_outputs.append(new_cost)

            current_cost = new_cost
            current_params = new_params
            current_h = new_h
            current_c = new_c

        # loss function
        loss = 0.0

        if self.loss_function_type == "weighted":
            for t in range(len(cost_outputs)):
                coeff = 0.1*(t+1)
                loss += cost_outputs[t]*coeff
            loss = loss/len(cost_outputs)

        elif self.loss_function_type == "observed improvement":
            cost = torch.stack(cost_outputs)
            for t in range(1, len(cost_outputs)):
                f_j = torch.min(cost[:t])
                penalty = torch.clamp(cost[t]-f_j, min = 0.0)
                loss += penalty
            loss = loss/len(cost_outputs)

        if intermediate_steps:
            return param_outputs, loss
        else:
            return loss

class ModelTrain:
    """
    Train and evaluate model
    """
    def __init__(self, model, lr_lstm = 0.01, lr_mapping = 0.01, num_rnn_iteration = 5):
        """
        Args:
         model [model]
         lr_lstm [float]: learning rate of LSTM or QK
         lr_mapping [float]: learning rate of linear model
         num_rnn_iteration [int]: number of RNN recurrent
         optimizer: ADAM
        """
        self.model = model
        self.model.to(device)
        self.lr_lstm = lr_lstm
        self.lr_mapping = lr_mapping
        self.num_rnn_iteration = num_rnn_iteration

        if self.model.mapping_type == "DS":
                learning_rate = [
                    {'params':self.model.lstm.parameters(), 'lr': self.lr_lstm},
                    {'params':self.model.single_mapping.parameters(), 'lr': self.lr_mapping},
                    {'params':self.model.double_mapping.parameters(), 'lr': self.lr_mapping}
                    ]
        else:
            learning_rate = [
                 {'params':self.model.lstm.parameters(), 'lr': self.lr_lstm},
                 {'params':self.model.mapping.parameters(), 'lr': self.lr_mapping}
                 ]
            
        self.optimizer = optim.Adam(learning_rate)

    def train_step(self, loss_qnode, num_rnn_iteration):
        self.optimizer.zero_grad()
        loss = self.model(loss_qnode, num_rnn_iteration)
        loss.backward()

        self.optimizer.step()

        return loss

    def train(self, train_data, epochs = 5, conv_tol_lstm = 1e-5):
        """
        Args:
         train_data [list]: train set
         epochs [int]: number of epochs
         conv_tol_lstm [float]: the convergence tolerance of LSTM or QK
        """
        self.model.train()
        previous_mean_loss = None

        print(f"\n--- Starting {self.model.model_type} Model Training ---")
        for epoch in range(epochs):
            if epoch%1 == 0:
                print(f"Epoch {epoch+1}/{epochs}")

            epoch_loss = []
            for i, molecule_data in enumerate(train_data):
                vqe = VQE.VQE(data_entry = molecule_data, 
                              max_qubits = self.model.max_qubits,
                              max_s_params = self.model.max_s_params,
                              with_LSTM = True,
                              with_DS = (self.model.mapping_type == "DS")) 
                
                loss_qnode = vqe.get_loss_function()
                loss = self.train_step(loss_qnode, self.num_rnn_iteration)
                epoch_loss.append(loss.item())

                if (i+1) % 10 == 0:
                    print(f" > Molecule {i+1}/{len(train_data)} - Loss: {loss.item():.8f}")

            #epoch_loss = np.array(epoch_loss)
            mean_loss = np.mean(epoch_loss)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1} Mean loss: {mean_loss:.8f}")

            if previous_mean_loss is not None:
                change = abs(previous_mean_loss - mean_loss)/abs(previous_mean_loss)
                if change <= conv_tol_lstm:
                    print(f"Traning converged at epoch {epoch+1}")
                    break

            previous_mean_loss = mean_loss

    def evaluate(self, molecule_data, num_rnn_iteration = 5):
        """
        Args:
         molecule_data [list]: molecule data from test set
         num_rnn_iteration [float]: number of RNN recurrent
        Outputs:
         lstm_guesses [list]: params predicted by model
         lstm_energies [list]: energies predicted by model
        """
        self.model.eval()
        print(f"\n--- Starting {self.model.model_type} Model Testing ---")
        vqe = VQE.VQE(data_entry = molecule_data, 
                              max_qubits = self.model.max_qubits,
                              max_s_params = self.model.max_s_params,
                              with_LSTM = True,
                              with_DS = (self.model.mapping_type == "DS")) 
        
        loss_qnode = vqe.get_loss_function()
        with torch.no_grad():
            params, loss = self.model(loss_qnode, num_rnn_iteration, intermediate_steps = True)
        
        lstm_guesses = [p.squeeze(0) for p in params]
        lstm_energies = [loss_qnode(guess).item() for guess in lstm_guesses]

        return lstm_guesses, lstm_energies
