import QAOA
import QKLSTM
import QLSTM
import FWP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelTrain:
    """
    Train and evaluate model
    """
    def __init__(self, model, qaoa_layers, lr_lstm = 0.01, lr_mapping = 0.01, num_rnn_iteration = 5):
        """
        Args:
         model [model]: the model defined by L2L or L2L_FWP
         qaoa_layers [int]: number of QAOA layers (P)
         lr_lstm [float]: learning rate of LSTM or QK
         lr_mapping [float]: learning rate of linear model
         num_rnn_iteration [int]: number of sequence model Phase I recurrent step (T)
         optimizer: RMSprop
        """
        self.model = model
        self.model.to(device)
        self.qaoa_layers = qaoa_layers
        self.lr_lstm = lr_lstm
        self.lr_mapping = lr_mapping
        self.num_rnn_iteration = num_rnn_iteration
        
        if self.model.model_type == "FWP":
            learning_rate = [
                {'params': self.model.fwp.parameters(), 'lr': self.lr_lstm},
                {'params': self.model.mapping.parameters(), 'lr': self.lr_mapping}
            ]

        elif self.model.mapping_type == "DS":
            learning_rate = [
                {'params':self.model.lstm.parameters(), 'lr': self.lr_lstm},
                {'params':self.model.single_mapping.parameters(), 'lr': self.lr_mapping},
                {'params':self.model.double_mapping.parameters(), 'lr': self.lr_mapping}
                    ]
        elif self.model.mapping_type == "ID":
            learning_rate = [
                {'params':self.model.lstm.parameters(), 'lr': self.lr_lstm},
                 ]
        else:
            learning_rate = [
                 {'params':self.model.lstm.parameters(), 'lr': self.lr_lstm},
                 {'params':self.model.mapping.parameters(), 'lr': self.lr_mapping}
                 ]
        
        
        self.optimizer = optim.RMSprop(learning_rate) 

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3, min_lr=1e-7)

    def train_step(self, loss_qnode, num_rnn_iteration):
        self.optimizer.zero_grad()
        loss = self.model(loss_qnode, num_rnn_iteration).to(device)
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_data, val_data, epochs = 5, conv_tol_lstm = 1e-5, time_out = 3600, save_path = None):
        """
        Args:
         train_data [list]: train set
         val_data [list]: validation set
         epochs [int]: number of epochs
         conv_tol_lstm [float]: the convergence tolerance of sequence model
         time_out [int]: time out for training (in seconds)
         save_path [str]: path to save the best model
        """
        self.model.train()
        previous_mean_loss = None
        mean_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience = 0
        start_time = time.time()

        train_cost = [QAOA.QAOA(graph = g,
                                    n_layers = self.qaoa_layers,
                                    with_meta=  True).get_loss_function() for g in train_data]
        
        val_cost = [QAOA.QAOA(graph = g,
                                    n_layers = self.qaoa_layers,
                                    with_meta=  True).get_loss_function() for g in val_data]

        print(f"\n--- Starting {self.model.model_type} Model Training ---")
        for epoch in range(epochs):
            elapsed_time = time.time() - start_time
            if elapsed_time > time_out:
                print(f"Training stopped after {epoch+1}/{epochs}")
                print(f"mean loss:{mean_loss_history}")
                print(f"mean val loss:{val_loss_history}")
                break 
            
            if epoch%1 == 0:
                print(f"Epoch {epoch+1}/{epochs}")

            epoch_loss = []
            
            for i, graph_data in enumerate(train_cost):
                
                loss_qnode = train_cost[i]
                loss = self.train_step(loss_qnode, self.num_rnn_iteration)
                epoch_loss.append(loss.item())

                if (i+1) % 200 == 0:
                    print(f" > Molecule {i+1}/{len(train_data)} - Loss: {loss.item():.8f}")
                
            mean_loss = np.mean(epoch_loss)
            mean_loss_history.append(mean_loss)
            
            self.model.eval()
            val_loss = 0
            
            for i, graph_data in enumerate(val_cost):
            
                loss_qnode = val_cost[i]
                params, loss = self.model(loss_qnode, self.num_rnn_iteration, intermediate_steps = True)
                cost = loss_qnode(params[-1]).item()
                val_loss+= cost
            mean_val_loss = val_loss/len(val_data)
            val_loss_history.append(mean_val_loss)
            self.model.train()

            self.scheduler.step(mean_val_loss)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1} Mean loss: {mean_loss:.8f}, Mean val loss:{mean_val_loss:.8f}")
                lr_lstm = self.optimizer.param_groups[0]['lr']
                print(f"Current {self.model.model_type} learning rate: {lr_lstm:.10f}")
                if self.model.mapping_type != "ID":
                    lr_mapping =  self.optimizer.param_groups[1]['lr']
                    print(f"Current mapping learning rate: {lr_mapping:.10f}")

            if previous_mean_loss is not None:
                #change = abs(previous_mean_loss - mean_loss)/abs(previous_mean_loss)
                change = abs(previous_mean_loss - mean_loss)
                if change <= conv_tol_lstm:
                    print(f"Traning converged at epoch {epoch+1}")
                    print(f"mean loss:{mean_loss_history}")
                    print(f"mean val loss:{val_loss_history}")
                    break
                
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience = 0
                torch.save(self.model.state_dict(), f'best_{self.model.model_type}_model_{save_path}.pth')
            else:
                patience +=1
                print(f"patience:{patience}")
                if patience >=6:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    print(f"mean loss:{mean_loss_history}")
                    print(f"mean val loss:{val_loss_history}")
                    break

            previous_mean_loss = mean_loss

        print(f"mean loss:{mean_loss_history}")
        print(f"mean val loss:{val_loss_history}")
        
    def evaluate(self, graph_data, num_rnn_iteration = 5):
        """
        Args:
         graph_data [list]: molecule data from test set
         num_rnn_iteration [float]: number of sequence model Phase I recurrent step (T)
        Outputs:
         l2l_guesses [list]: params predicted by model
         l2l_cost [list]: cost corresponding to l2l_guesses
        """
        self.model.eval()
        print(f"\n--- Starting {self.model.model_type} Model Testing ---")
        qaoa = QAOA.QAOA(graph = graph_data, 
                         n_layers = self.qaoa_layers, 
                         with_meta=  True)
        
        loss_qnode = qaoa.get_loss_function()
        with torch.no_grad():
            params, loss = self.model(loss_qnode, num_rnn_iteration, intermediate_steps = True)
        
        l2l_guesses = [p.squeeze(0) for p in params]
        l2l_cost = [loss_qnode(guess).item() for guess in l2l_guesses]

        return l2l_guesses, l2l_cost
