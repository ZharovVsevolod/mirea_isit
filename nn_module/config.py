from dataclasses import dataclass
from typing import Literal

@dataclass
class Dataset_Path:
  train_name: str
  test_name: str
  truth_name: str
  raw_train_dest: str
  raw_test_dest: str
  raw_truth_dest: str
  train_dest: str
  test_dest: str

@dataclass
class Dataset:
  time_lenght: int
  id_var: int

@dataclass
class Model_Rnn:
  name: Literal["lstm", "gru", "rnn"]
  input_size: int
  hidden_size: int
  num_layers: int
  drop: float

@dataclass
class Training:
   epochs: int
   batch_size: int
   lr: float

@dataclass
class Params:
    dataset_path: Dataset_Path
    dataset: Dataset
    model: Model_Rnn
    training: Training