import pennylane as qml
from pennylane import qchem
import torch 
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQE:
    """
    Variantional quantum eigensolver algorithm
    """
    def __init__(self,
                 data_entry,
                 max_qubits,
                 max_s_params,
                 with_LSTM = True,
                 with_DS = False,
                 ):# if use single and double linear model, add max_s_params
        """
        Args:
         data_entry (list): molecule data
         max_qubits [int]: the max qubits we need in this experiment
         (max_total_params [int]: the max params from linear output in this experiment
         with_LSTM [bool]: if params come from LSTM-FC: True, otherwise: False
        """
        self.data = data_entry
        self.max_qubits = max_qubits
        #self.max_total_params = max_total_params
        self.max_s_params = max_s_params
        self.with_LSTM = with_LSTM
        self.with_DS = with_DS
        """
        To calculate the number of single and double excitation params.need for this molecule, need
         the number of orbitals, electrons to get total_vqe_params, and use this to get the params
         we need from params to obtain uccsd_params for UCCSD ansatz
        """
        self.wires = range(self.max_qubits)
        self.orbitals = len(self.data.hf_state)
        self.electrons = sum(self.data.hf_state)
        self.singles, self.doubles = qchem.excitations(self.electrons, self.orbitals)
        self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(self.singles, self.doubles)
        self.num_single = len(self.s_wires)
        self.num_double = len(self.d_wires)
        self.total_vqe_params = self.num_single + self.num_double

        try:
            if torch.cuda.is_available():
                self.device = qml.device("lightning.gpu", wires = self.wires)
            else:
                self.device = qml.device("lightning.qubit", wires = self.wires)
        except:
            self.device = qml.device("lightning.qubit", wires = self.wires)
            
        self.qnode = qml.QNode(self.ansatz, self.device, interface = "torch", diff_method = "adjoint") # change to adjoint if don't use metric

    def ansatz(self, params):

        if self.with_LSTM:
           if self.with_DS:
               # if use single and double linear model
               params = params.squeeze()
               single_params = params[:self.num_single]
               double_params = params[self.max_s_params: self.max_s_params + self.num_double]
               uccsd_params = torch.cat([single_params, double_params], dim = 0)
           else:
               # if use single linear model
               uccsd_params = params.squeeze(0)[:self.total_vqe_params]
        
        else:
            uccsd_params = params
        # UCCSD ansatz
        qml.UCCSD(uccsd_params, wires = range(self.orbitals), s_wires = self.s_wires, d_wires = self.d_wires, init_state = self.data.hf_state)

        return qml.expval(self.data.hamiltonian)
    """
    if need metric, and this takes a long time
    def get_metric_fn(self):
        def metric_fn(p):
            p = torch.tensor(p, requires_grad=True, dtype=torch.float32)
            p = p[:self.total_vqe_params]
            metric = qml.metric_tensor(self.qnode, approx="diag")(p)
            metric_diag = torch.diag(metric)[:self.total_vqe_params]
            metric_diag = metric_diag / (torch.max(torch.abs(metric_diag)) + 1e-8)
            if self.total_vqe_params < self.max_total_params:
              metric_diag = torch.nn.functional.pad(metric_diag, (0, self.max_total_params - self.total_vqe_params), value=1.0)
            return metric_diag
        return metric_fn
    """
    def get_loss_function(self):
        #def loss_fn(theta):
        #  theta = theta.to(self.device, dtype=torch.float32)
        #  theta.requires_grad_(True)
        #  energy = self.qnode(theta)
        #  grad = torch.autograd.grad(energy, theta, create_graph=True)[0]
        #  grad_norm = torch.norm(grad, p=2)
        #  return energy, grad_norm

        return  lambda theta: self.qnode(theta) #loss_fn 

class VQEOptimizer:
    """
    VQE optimization by ADAM or SGD
    """
    def __init__(self, vqe_problem:VQE):
        self.vqe_problem = vqe_problem
        self.cost_function = vqe_problem.get_loss_function()

    def run_optimization(self, initial_params, optimizer = 'Adam', max_iter = 500, learning_rate = 0.01, conv_tol = 1e-6):
        """
        Args:
         initial_params [list]
         optimizer [str]:　ADAM or SGD
         max_iter [int]: max interation
         learning_rate [float]: learning rate of VQE
         conv_tol [float]: convergence

        Output:
         conv_iter [int]: convergence iteration
         param_history[-1] [float]: final optimizedd params
         energy_history[-1] [float]: final optimized energy
         param_history [list]
         energy_history [list]

        """
        params = torch.tensor(initial_params, requires_grad= True, dtype = torch.float32)

        if optimizer == 'ADAM':
            opt = optim.Adam([params], lr = learning_rate)

        elif optimizer == 'SGD':
            opt = optim.SGD([params], lr = learning_rate)

        energy_history = [self.cost_function(params).item()]
        param_history = [params.detach().clone()]
        conv_iter = max_iter

        print("\n--- Starting VQE Optimization ---")

        for iteration in range(max_iter):
            opt.zero_grad()
            energy = self.cost_function(params)
            energy.backward()
            opt.step()

            param_history.append(params.detach().clone())
            energy_history.append(energy.item())

            if (iteration+1)%50 == 0:
                print(f"Step = {iteration+1}/{max_iter}, Energy = {energy_history[-1]:.8f} Ha")
            if iteration > 0:
                conv = abs(energy_history[-1] - energy_history[-2])
                if conv <= conv_tol:
                    conv_iter = iteration + 1
                    print(f"  Convergence reached at step {conv_iter}")
                    break

        print(f"Optimization finished, final energy: {energy_history[-1]:.8f} Ha")
        return conv_iter, param_history[-1], energy_history[-1], param_history, energy_history
