import torch
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import torch.nn as nn


class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits,
        n_qlayers,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        backend="lightning.gpu",
        device=torch.device("cuda:1"),
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.num_layers = n_qlayers
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.device = device

        self.quantum_wires = {
            "forget": [f"wire_forget_{i}" for i in range(self.n_qubits)],
            "input": [f"wire_input_{i}" for i in range(self.n_qubits)],
            "update": [f"wire_update_{i}" for i in range(self.n_qubits)],
            "output": [f"wire_output_{i}" for i in range(self.n_qubits)],
        }

        self.quantum_devices = {
            "forget": qml.device(backend, wires=self.quantum_wires["forget"]),
            "input": qml.device(backend, wires=self.quantum_wires["input"]),
            "update": qml.device(backend, wires=self.quantum_wires["update"]),
            "output": qml.device(backend, wires=self.quantum_wires["output"]),
        }

        self.clayer_in = nn.Linear(self.concat_size, self.n_qubits).to(device)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size).to(device)

        weight_shapes = {"weights": (self.n_qlayers, self.n_qubits)}
        initial_weights = init.uniform_(torch.empty(n_qlayers, n_qubits), a=-np.pi, b=np.pi).to(device)
       
        # weight = {"weights": initial_weights}

        self.quantum_nodes = {
            k: qml.QNode(self._build_circuit(k), dev, interface="torch")
            for k, dev in self.quantum_devices.items()
        }

        self.kernel = {
            gate_type: qml.qnn.TorchLayer(
                self.quantum_nodes[gate_type],
                weight_shapes ,#weight_shapes
            ).to(device)
            for gate_type in ["forget", "input", "update", "output"]
        }
        self.to(device)

    def _build_circuit(self, gate_type):
        def circuit(inputs, weights):
            wires = self.quantum_wires[gate_type]
            qml.templates.AngleEmbedding(inputs, wires=wires)
            qml.templates.BasicEntanglerLayers(weights, wires=wires)
          
            results = [qml.expval(qml.PauliZ(wires=wire)) for wire in wires]
            return results

        return circuit

    def forward(self, x, init_states=None):
        x = x.to(self.device)
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            #init_states = init_states.to(self.device)
            h_t, c_t = init_states
            #h_t = h_t[0]
            if h_t.dim() == 3:
                h_t = h_t.squeeze(0)
                c_t = c_t.squeeze(0)
            
            #c_t = c_t[0]
        for t in range(seq_length):
            x_t = x[:, t, :]
           
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.kernel["forget"](y_t)))
            i_t = torch.sigmoid(self.clayer_out(self.kernel["input"](y_t)))
            g_t = torch.tanh(self.clayer_out(self.kernel["update"](y_t)))
            o_t = torch.sigmoid(self.clayer_out(self.kernel["output"](y_t)))
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).to(self.device)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        if self.return_sequences:
            return hidden_seq, (h_t, c_t)
        else:
            return h_t, (h_t, c_t)