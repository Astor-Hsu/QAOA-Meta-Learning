import torch
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import torch.nn as nn


class QKLSTM(nn.Module):
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
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super(QKLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
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

        self.clayer_in = nn.Linear(self.concat_size, self.n_qubits, dtype=torch.float64, device=self.device)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size, dtype=torch.float64, device=self.device)

        weight_shapes = {"weights": (n_qlayers, n_qubits)}

        self.quantum_nodes = {
            k: qml.QNode(self._build_circuit(k), dev, interface="torch")
            for k, dev in self.quantum_devices.items()
        }

        self.kernel = {
            gate_type: qml.qnn.TorchLayer(
                self.quantum_nodes[gate_type],
                weight_shapes,
            ).to(device)
            for gate_type in ["forget", "input", "update", "output"]
        }
        self.to(device)

    def _build_circuit(self, gate_type):
        def circuit(inputs, weights):
            wires = self.quantum_wires[gate_type]
            # Forward pass
            # for layer in range(self.n_qlayers):
            #     for i in range(len(wires)):
            #         qml.RX(weights[layer, i], wires=wires[i])
            #         qml.RY(weights[layer, i], wires=wires[i])
            #         qml.RZ(weights[layer, i], wires=wires[i])
            for wire in wires:
                qml.Hadamard(wires=wire)
            qml.templates.AngleEmbedding(
                torch.cos(inputs**2), wires=wires, rotation="Z"
            )
            qml.templates.AngleEmbedding(
                torch.cos(inputs**2), wires=wires, rotation="Y"
            )
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.templates.AngleEmbedding(
                torch.cos(inputs**2), wires=wires, rotation="Z"
            )
            # Inverse pass
            qml.adjoint(qml.templates.AngleEmbedding)(inputs, wires=wires, rotation="Z")
            for i in range(len(wires) - 1, 0, -1):
                qml.CNOT(wires=[wires[i - 1], wires[i]])
            qml.adjoint(qml.templates.AngleEmbedding)(inputs, wires=wires, rotation="Y")
            qml.adjoint(qml.templates.AngleEmbedding)(inputs, wires=wires, rotation="Z")
            for wire in wires:
                qml.Hadamard(wires=wire)
            # qml.templates.AngleEmbedding(inputs, wires=wires)
            # qml.templates.BasicEntanglerLayers(weights, wires=wires)
            results = [qml.expval(qml.PauliZ(wires=wire)) for wire in wires]
            return results

        return circuit

    def forward(self, x, init_states=None):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # from (input_dim,) → (1, 1, input_dim)
        elif x.dim() == 2:
            x = x.unsqueeze(1)               # from (batch_size, input_dim) → (batch_size, 1, input_dim)

        x = x.to(self.device)

        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()
            x = x.transpose(0, 1)
            x = x.to(self.device)

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            h_t, c_t = init_states
            h_t = h_t.to(self.device)
            c_t = c_t.to(self.device)
        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            v_t = v_t.to(self.device, dtype=torch.float64)
            y_t = self.clayer_in(v_t).to(self.device)

        

            out_f = self.kernel["forget"](y_t).to(self.device, dtype=torch.float64)
            out_i = self.kernel["input"](y_t).to(self.device, dtype=torch.float64)
            out_g = self.kernel["update"](y_t).to(self.device, dtype=torch.float64)
            out_o = self.kernel["output"](y_t).to(self.device, dtype=torch.float64)

            f_t = torch.sigmoid(self.clayer_out(out_f))
            i_t = torch.sigmoid(self.clayer_out(out_i))
            g_t = torch.tanh(self.clayer_out(out_g))
            o_t = torch.sigmoid(self.clayer_out(out_o))
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            # Ensure h_t is on the correct device and dtype
            h_t = h_t.to(self.device, dtype=torch.float64)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).to(self.device)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        if self.return_sequences:
            return hidden_seq, (h_t, c_t)
        else:
            return h_t, (h_t, c_t)