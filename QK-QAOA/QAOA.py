import pennylane as qml
from pennylane import qaoa
import torch 
import torch.optim as optim

class QAOA:

    def __init__(self, graph, n_layers=1, with_meta = True):
        self.graph = graph
        self.n_layers = n_layers
        self.total_qaoa_params = 2 * self.n_layers
        self.wires = range(len(self.graph.nodes))
        self.cost_h, self.mixer_h = qaoa.maxcut(self.graph)
        self.with_meta = with_meta

        self.device = qml.device("default.qubit", wires=len(self.wires))
        self.qnode = qml.QNode(self._circuit, self.device, interface = "torch", diff_method = "backprop")

    def _qaoa_layer(self, gamma, alpha):
      qaoa.cost_layer(gamma, self.cost_h)
      qaoa.mixer_layer(alpha, self.mixer_h)

    def _circuit(self, params):
      if self.with_meta:
        params = params[:self.total_qaoa_params]
        qaoa_params = params.reshape(2, self.n_layers)

      else:
        qaoa_params = params.reshape(2, self.n_layers)
        
      for w in self.wires:
        qml.Hadamard(wires=w)
       
      qml.layer(self._qaoa_layer, self.n_layers, qaoa_params[0], qaoa_params[1])

      return qml.expval(self.cost_h)

    def get_loss_function(self):
      return lambda theta: -self.qnode(theta)

class QAOAptimizer:
    """
    QAOA optimization by ADAM or SGD
    """
    def __init__(self, qaoa_problem: QAOA):
        self.qaoa_problem = qaoa_problem
        self.cost_function = qaoa_problem.get_loss_function()

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
        params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float32)

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