#  Meta-Learning for Quantum Optimization via  Quantum Sequence Model

<br>
<p align="center">
  <img src="img/Overview.png" alt="Overview" width="100%" height="auto">
</p>

## Introduction
This is the offical repository of the paper Meta-Learning for Quantum Optimization via Quantum Sequence Model

The Quantum Approximate Optimization Algorithm (QAOA) is a leading approach for solving combinatorial optimization problems on near-term quantum processors. However, finding good variational parameters remains a significant challenge due to the non-convex energy landscape, often resulting in slow convergence and poor solution quality. In this work, we propose a quantum meta-learning framework that trains advanced quantum sequence models to generate effective parameter initialization policies. We investigate four classical or quantum sequence models, including the Quantum Kernel-based Long Short-Term Memory (QK-LSTM), as learned optimizers in a "learning to learn" paradigm. Our numerical experiments on the Max-Cut problem demonstrate that the QK-LSTM optimizer achieves superior performance, obtaining the highest approximation ratios and exhibiting the fastest convergence rate across all tested problem sizes ($n=10$ to $13$). Crucially, the QK-LSTM model achieves perfect parameter transferability by synthesizing a single, fixed set of near-optimal parameters, leading to a remarkable sustained acceleration of convergence even when generalizing to larger problems. This capability, enabled by the compact and expressive power of the quantum kernel architecture, underscores its effectiveness. The QK-LSTM, with only 43 trainable parameters, substantially outperforms the classical LSTM (56 parameters) and other quantum sequence models, establishing a robust pathway toward highly efficient parameter initialization for variational quantum algorithms in the NISQ era.

<div style="display: flex; align-items: center;">
<div style="flex: 1;">

## Installation
1. Create a new conda environment with Python 3.10.18:
   ```bash
   conda create --name [name] python=3.10
   ```
2. Activate the newly created environment:
   ```bash
   conda create [name] 
   ```
3. Install the required package using pip:
   ```bash
   pip install -r requirements.txt
   ```
## Options
 - `--Train_and_Test`: Default is `True`. Use `--Train_and_Test True` if you want to train and then test the model.
 - `--Only_train`: Default is `False`. Use `--Only_train True` if only tain the model.
 - `--Only_test`: Default is `False`. Use `--Only_test True` if you already have the model parameters pth file and you just want to test it.
 ### Device Arguments
  - `--device`: Default is `cpu`. Use `--device` to set the device, cpu or gpu.
  - `--backend_sequence`: Default is `lightning.qubit`. Use `--backend_sequence` to set the pennylane backend for the quantum sequence moodel.
  - `--backend_QAOA`: Default is `default.qubit`. Use `--backend_QAOA` to set the pennylane backend for the QAOA.
### Data Load and Save Path
  - `--dataset_save_path`: Default is `datasets.pkl`. Use `--dataset_save_path` to load your dataset. 
  - `--model_save_path`: Default is `models_default`. Use `--model_save_path` to set the file name to save model parameters.
   - `--Results_save_path`: Default is `results_default`. Use `--Results_save_path` to set the file name to save result and image.
  - `--load_path`: Default is `None`. Use `--load_path` to load the model parameters if you need.
### Model Arguments
 - `--model_type`: Use `--model_type` to set the type of sequence model. 
 - `--mapping_type`: Default is `ID`. Use `--mapping_type` to set the type of mapping model if you need.
 - `--layers`: Default is `1`. Use `--layers` to set the number of layers for the sequence model.
 - `--input_feature_dim`: Default is `2`. Use `--input_feature_dim` to set the number of parameter input to the sequence model (doesn't inclue one parameter for cost).
 - `--max_total_params`: Default is `2`. Use `--max_total_params` to set the number of parameter for the QAOA ansatz.
 - `--qubits`: Default is `4`. Use `--qubits` to set the number of qubits for the QK-LSTM and QLSTM model.
 - `--loss_function_type`: Default is `weighted`. Use `--loss_function_type` to set the type of meta-loss function.
 - `--lr_sequence`: Default is `6e-6`. Use `--lr_sequence` to set the learning rate for sequence model.
 - `--lr_mapping`: Default is `1e-4`. Use `--lr_mapping` to set the learning rate for mapping model.
 - `--epochs`: Default is `5`. Use `--epochs` to set the number of training epochs.
 - `--steps_recurrent_loop_train`: Default is `10`. Use `--steps_recurrent_loop_train` to set the number of recurrent steps during training.
 - `--conv_tol_sequence`: Default is `1e-5`. Use `--conv_tol_sequence` to set the convergence tolerance for training.
 - `--time_out`: Default is `7200`. Use `--time_out` to set the timeout in seconds for training.
 - `--continue_train`: Default is `False`. Use `--continue_train` if yoy already have the model parameters file and you plan to continue tain.
 - `--steps_recurrent_loop_test`: Default is `10`. Use `--steps_recurrent_loop_test` to set the number of recurrent steps during testing (Phase I).

 ### QAOA MaxCut Arguments
 - `--qaoa_layers`: Default is `1`. Use `--qaoa_layers` to set the number of layers in QAOA MaxCut.
 - `--qaoa_optimizer`: Default is `ADAM`. Use `--qaoa_optimizer` to set the optimizer for the QAOA optimization.
 - `--lr_qaoa`: Default is `1e-3`. Use `--lr_qaoa` to set the learning rate for the QAOA optimization.
 - `--max_iter_qaoa`: Default is `300`. Use `--max_iter_qaoa` to set the max iterations for QAOA optimization.
 - `--conv_tol_qaoa`: Default is `1e-6`. Use `--conv_tol_qaoa` to set the convergence tolerance for QAOA optimization.


## Usage
### Prepare dataset
The dataset pkl format file contains train, val., and test set. Each graphs generated via networkx package. You can generate or design graph by yourself, but the graph must has .nodes, .nodes and .edges.data("weight") attribute. 

### Training
If you only train model, don't want to test the model later, just type `True` at `Only_train` arg. and `False` at `Train_and_Test` An example is:
```bash
python main.py -- Train_and_Test False --Only_train True --dataset_save_path [dataset_path.pkl] --model_type QK --mapping_type Linear
```

Otherwist, 
```bash
python main.py -- Train_and_Test True --dataset_save_path [dataset_path.pkl] --model_type QK --mapping_type Linear 
```

### Testing
If you already have the model, you can just load pth file into the model and then test it

```bash
python main.py -- Train_and_Test False --Only_test True --dataset_save_path [dataset_path.pkl] --load_path [your_model_params._path.pth] --model_type QK --mapping_type Linear
```

### Results
When saving results, the output folder follows this format:
```
QAOA_Random_node_[nodes_of_the_graph]_edge_[edges_of_the_graph]_[i].npz
```
and
```
[args.Results_save_path]_node_[nodes_of_the_graph]_edge_[edges_of_the_graph]_[i].csv
```
and
```
Result_[args.Results_save_path]_node_[nodes_of_the_graph]_edges_[edges_of_the_graph]_[i].svg
```

When saving model checkpoints, the folder follows this format:
```
best_[args.model_type]_model_[args.model_save_path].pth
```
and
```
[Model_save_path]_[model_type]_[lr_sequence]_[lr_mapping].pth
```


## News

<h2 id="citation">🔖 Citation</h2>

📚 If you find our work or this code to be useful in your own research, please kindly cite our paper :-)

<h2 id="authors">🖌️ Authors</h2>

[Yu-Cheng Lin](https://github.com/Xiezhihaa), [Yu Hsu](https://github.com/Astor-Hsu).
