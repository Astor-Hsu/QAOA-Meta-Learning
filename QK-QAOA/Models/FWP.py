import torch
import torch.nn as nn
import pennylane as qml
from functools import partial

######################
# VQC 
######################
def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

def quantum_net(inputs, q_weights, n_outputs):
    n_dep = q_weights.shape[0]
    n_qub = q_weights.shape[1]
    H_layer(n_qub)
    RY_layer(inputs)

    for k in range(n_dep):
        entangling_layer(n_qub)
        RY_layer(q_weights[k])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_outputs)]

######################
# Wrapper so Torch can use it
######################
class HideSignature:
    def __init__(self, partial_func):
        self.partial_func = partial_func
    def __call__(self, inputs, q_weights):
        return self.partial_func(inputs, q_weights)

class BatchVQC:
    """Batch wrapper for VQC"""
    def __init__(self, q_func):
        self.q_func = q_func
    def __call__(self, inputs, q_weights):
        res_all = []
        for input_item, q_weight_item in zip(inputs, q_weights):
            res = self.q_func(input_item, q_weight_item)
            # Ensure output is always float32 for downstream compatibility
            res_all.append(torch.stack(res).to(torch.float32))
        return torch.stack(res_all)

######################
# FWP Cell 
######################
class FWPCell(nn.Module):
    def __init__(self, s_dim, a_dim, n_qubits=4, n_layers=2, backend="lightning.gpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = n_layers
        self.a_dim = a_dim

        # quantum device + QNode
        dev = qml.device(backend, wires=self.n_qubits)
        qnode = qml.QNode(HideSignature(partial(quantum_net, n_outputs=a_dim)), dev, interface="torch")

        self.q_func = BatchVQC(qnode)
        self.input_encoder = nn.Linear(s_dim, n_qubits)
        # slow programmer
        self.slow_program_encoder = nn.Linear(s_dim, n_qubits)
        self.slow_program_layer_idx = nn.Linear(n_qubits, self.q_depth)
        self.slow_program_qubit_idx = nn.Linear(n_qubits, self.n_qubits)

        # === Post-processing: map n_qubits → a_dim ===
        self.post_mapping = nn.Linear(self.n_qubits, self.a_dim)  

    def forward(self, x_t, prev_params):
        # Accepts x_t of shape (batch, s_dim) or (batch, 1, s_dim)
        # Accepts prev_params of shape (batch, n_layers, n_qubits)
        # Ensure all tensors are float32 for compatibility
        x_t = x_t.to(torch.float32)
        prev_params = prev_params.to(torch.float32)

        # If x_t has shape (batch, 1, s_dim), squeeze the sequence dimension
        if x_t.dim() == 3 and x_t.shape[1] == 1:
            x_t = x_t.squeeze(1)

        encoded_inputs = self.input_encoder(x_t)
        res = self.slow_program_encoder(x_t)

        res_layer_idx = self.slow_program_layer_idx(res)   # (batch, n_layers)
        res_qubit_idx = self.slow_program_qubit_idx(res)   # (batch, n_qubits)

        # outer product for fast params
        out_circuit_params = []
        for layer_idx, qubit_idx in zip(res_layer_idx, res_qubit_idx):
            outer_product = torch.outer(layer_idx, qubit_idx)
            out_circuit_params.append(outer_product)
        out_circuit_params = torch.stack(out_circuit_params)

        # accumulate
        out_circuit_params = out_circuit_params + prev_params

        # run quantum net
        res = self.q_func(encoded_inputs, out_circuit_params)
        res = res.to(torch.float32)  # Ensure float32 before post_mapping

        # post-process (map n_qubits → a_dim)
        res = self.post_mapping(res)   

        return res, out_circuit_params

    def initial_fast_params(self, batch_size):
        # Ensure float32 dtype
        return torch.zeros(batch_size, self.q_depth, self.n_qubits, dtype=torch.float32)

######################
# Full FWP 
######################
class FWP(nn.Module):
    def __init__(self, s_dim, a_dim, n_qubits=8, n_layers=2, backend="lightning.cpu", device="cuda:0"):
        super().__init__()
        self.fwp_cell = FWPCell(s_dim, a_dim, n_qubits, n_layers, backend)

    def forward(self, x):
        """
        x: (batch, seq_len, s_dim) or (batch, s_dim)
        """
        # Ensure input is float32 for all downstream ops
        x = x.to(torch.float32)
        if x.dim() == 2:
            # (batch, s_dim) → (batch, 1, s_dim)
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.size()
        fast_params = self.fwp_cell.initial_fast_params(batch_size)

        outputs = []
        for t in range(seq_len):
            out_t, fast_params = self.fwp_cell(x[:, t, :], fast_params)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)   # (batch, seq_len, a_dim)
        return outputs
