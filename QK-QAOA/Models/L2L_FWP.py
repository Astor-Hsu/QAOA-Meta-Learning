import FWP
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class L2L_FWP(nn.Module):
    """
    Define FWP model

    Args:
        mapping_type (str): the type of mapping model, Linear (FC) or ID
        input_feature_dim (int): the number of parameters for LSTM output to QAOA ansatz
        max_total_params [int]: the max numbers of parameters for QAOA we need for this experiment
        loss_function_type [str]: the meta-loss function type, weighted or observed improvement (define by Verdon's papaer)
        layers (int): the number of layers for FWP model
    """
    def __init__(self, mapping_type="ID", input_feature_dim=0, max_total_params=0,
                 loss_function_type="weighted", layers=1):
        super().__init__()
        self.model_type = "FWP"
        self.mapping_type = mapping_type
        self.input_feature_dim = input_feature_dim
        self.max_total_params = max_total_params
        self.loss_function_type = loss_function_type
        self.layers = layers

        # define FWP model
        self.fwp = FWP.FWP(
            s_dim=self.input_feature_dim + 1,  
            a_dim=self.input_feature_dim,   
            n_qubits=2,
            n_layers=self.layers,
            backend="default.qubit",
            device=device
        ).to(device)

        # define mapping model
        if self.mapping_type == "Linear":
            self.mapping = nn.Linear(self.input_feature_dim, self.max_total_params).to(device)
        elif self.mapping_type == "ID":
            self.mapping = nn.Identity().to(device)

    def forward(self, molecule_cost, num_iteration, intermediate_steps=False):
        current_cost = torch.zeros((1, 1), dtype=torch.float32).to(device)
        current_params = torch.zeros((1, self.input_feature_dim), dtype=torch.float32).to(device)

        param_outputs, cost_outputs = [], []

        for _ in range(num_iteration):
            # (batch, s_dim)
            new_input = torch.cat([current_cost, current_params.view(current_params.size(0), -1)], dim=1)

            #  seq_len=1 → (batch, 1, s_dim)
            new_input = new_input.unsqueeze(1)

            # FWP forward
            new_params = self.fwp(new_input)  # (batch, a_dim)

            params = self.mapping(new_params).to(device)
            _cost = molecule_cost(params.squeeze(0).to(device))
            new_cost = torch.as_tensor(_cost, dtype=torch.float32, device=device).view(1, 1)      
            param_outputs.append(params)
            cost_outputs.append(new_cost)
            current_cost = new_cost
            current_params = new_params

        # meta-loss function
        loss = 0.0
        if self.loss_function_type == "weighted":
            for t in range(len(cost_outputs)):
                coeff = 0.1 * (t + 1)
                loss += cost_outputs[t] * coeff
            loss = loss / len(cost_outputs)

        elif self.loss_function_type == "observed improvement":
            cost = torch.stack(cost_outputs).to(device)
            zero = torch.tensor([0.0], device=device)
            for t in range(1, len(cost_outputs)):
                f_j = torch.min(cost[:t])
                penalty = torch.minimum(cost[t] - f_j, zero)
                loss += penalty
            loss = loss / len(cost_outputs)

        return (param_outputs, loss) if intermediate_steps else loss