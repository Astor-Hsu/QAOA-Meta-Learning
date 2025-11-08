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
      return lambda theta: self.qnode(theta)

class QAOAptimizer:
    """
    QAOA optimization by ADAM or SGD
    """
    def __init__(self, qaoa_problem: QAOA):
        self.qaoa_problem = qaoa_problem
        self.cost_function = qaoa_problem.get_loss_function()

    def run_optimization(self, initial_params, optimizer = 'ADAM', max_iter = 500, learning_rate = 0.01, conv_tol = 1e-6):
        """
        Args:
         initial_params [list]: initial parameters of QAOA
         optimizer [str]: the type of optimizer, ADAM or SGD
         max_iter [int]: max iteration
         learning_rate [float]: learning rate of QAOA
         conv_tol [float]: convergence

        Output:
         conv_iter [int]: convergence iteration
         param_history[-1] [float]: final optimizedd params
         cost_history[-1] [float]: final optimized cost
         param_history [list]
         cost_history [list]

        """
        params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float32)

        if optimizer == 'ADAM':
            opt = optim.Adam([params], lr = learning_rate)

        elif optimizer == 'SGD':
            opt = optim.SGD([params], lr = learning_rate)

        cost_history = [self.cost_function(params).item()]
        param_history = [params.detach().clone()]
        conv_iter = max_iter

        print("\n--- Starting QAOA Optimization ---")

        for iteration in range(max_iter):
            opt.zero_grad()
            cost = self.cost_function(params)
            cost.backward()
            opt.step()

            param_history.append(params.detach().clone())
            cost_history.append(cost.item())

            if (iteration+1)%50 == 0:
                print(f"Step = {iteration+1}/{max_iter}, Cost = {cost_history[-1]:.8f}")
            if iteration > 0:
                conv = abs(cost_history[-1] - cost_history[-2])
                if conv <= conv_tol:
                    conv_iter = iteration + 1
                    print(f"  Convergence reached at step {conv_iter}")
                    break

        print(f"Optimization finished, final cost: {cost_history[-1]:.8f}")
        return conv_iter, param_history[-1], cost_history[-1], param_history, cost_history
