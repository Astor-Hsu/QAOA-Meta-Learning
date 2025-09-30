import h5py
import pennylane as qml
from pennylane import qchem
import pandas as pd

class molecule_data:
  """
  Download molecule data from pennylane and data processing
  """
  def __init__(self, molecule_list, ):
    self.molecule_list = molecule_list

  @staticmethod
  def load_data(molecule, basis = "STO-3G", folder_path = "./datasets/"):
    """
    Download data from pennylane

    Args:
     molecule (str, [float]): str for molecule name, [float] for bondlength
     basis (str): default: STO-3G, other: CC-PVDZ and 6-31G
     folder_path (str): path to the directory used for saving datasets. Defaults to './datasets'

    Return:
        DataFrame[:class:`~pennylane.data.Dataset`]

    """
    data_list = []

    print("\n--- Starting Load Data ---")
    for mol_name, bondlengths in molecule:
      if bondlengths is None:
        data = qml.data.load("qchem", molname = mol_name, basis = basis, folder_path = folder_path)
      else:
        data = qml.data.load("qchem", molname = mol_name, basis = basis,folder_path = folder_path, bondlength = bondlengths)

      for entry in data:
        data_list.append(entry)

      df = pd.DataFrame(data_list)

    print("\n--- Complete Load Data ---")
    print(df.head())
    return df

  @staticmethod
  def data_params(df, train_split_index):
    """
    Calculate the basis params like the max qubits we need for this experiment
    Args:
     df [DataFrame]: the molecule dataset for train and test
     train_split_indes [int]: to split the dataset into train set and test set
     
    Return:
     max_qubits [int]: the max qubits we need for this experiment
     max_s_params [int]: the max single params we need for this experiment
     max_d_params [int]: the max double params we need for this experiment
     max_total_params [int]: max_s_params + max_d_params
     train_set [list]
     test_set [list]

    """
    # define max qubits

    max_qubits = 0

    for i in range(len(df[0])):
      if len(df[0][i].hf_state) > max_qubits:
        max_qubits = len(df[0][i].hf_state)

    # find the max params need for single and double
    max_s_params = 0
    max_d_params = 0

    for data in df[0]:
      electrons = sum(data.hf_state)
      orbitals = len(data.hf_state)

      single, double = qchem.excitations(electrons, orbitals)
      s_w, d_w = qml.qchem.excitations_to_wires(single, double)
      num_single = len(s_w)
      num_double = len(d_w)
      max_s_params = max(max_s_params, num_single)
      max_d_params = max(max_d_params, num_double)
    max_total_params = max_s_params + max_d_params

    # split the dataset into train set and test set
    df_list = list(df[0])
    train_set = df_list[:train_split_index]
    test_set = df_list[train_split_index:]

    return max_qubits, max_s_params, max_d_params, max_total_params, train_set, test_set

