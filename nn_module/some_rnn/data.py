from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class Airplane_Dataset(Dataset):
    def __init__(self, df, id_var, time_lenght) -> None:
        super().__init__()
        sensor_cols = ['s' + str(i) for i in range(1,22)]
        sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
        sequence_cols.extend(sensor_cols)

        self.sequence, self.labels = self.full_get_seq_and_labels(
            df=df,
            id_var=id_var,
            time_lenght=time_lenght, 
            sequence_cols=sequence_cols
        )
    
    def gen_sequence(self, id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]
    
    def gen_labels(self, id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]
    
    def get_seq_and_labels(self, df, id_var, sequence_length, sequence_cols):
        seq_gen = (list(self.gen_sequence(df[df['id']==id_var], sequence_length, sequence_cols)))
        seq_gen = torch.tensor(seq_gen)

        label_gen = [self.gen_labels(df[df['id']==id_var], sequence_length, ['label1'])]
        label_array = np.concatenate(label_gen, dtype=float)
        new_label = [float(x) for x in label_array]
        new_label = torch.tensor(new_label).unsqueeze(1)

        return seq_gen, new_label
    
    def full_get_seq_and_labels(self, df, id_var, time_lenght, sequence_cols):
        if id_var is not None:
            return self.get_seq_and_labels(df, id_var, time_lenght, sequence_cols)
        
        id_var = np.linspace(2, 100)
        seq, lab = self.get_seq_and_labels(df, 1, time_lenght, sequence_cols)
        for id in id_var:
            seq_train, labels_train = self.get_seq_and_labels(df, id, time_lenght, sequence_cols)
            seq = torch.cat((seq, seq_train), dim=0)
            lab = torch.cat((lab, labels_train), dim=0)
        return seq, lab
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequence[index], self.labels[index]

class DataModule(L.LightningDataModule):
    def __init__(self, train_dest, test_dest, id_var=None, time_lenght=50, batch_size=128) -> None:
        super().__init__()
        self.train_dest = train_dest
        self.test_dest = test_dest
        self.id_var = id_var
        self.time_lenght = time_lenght
        self.batch_size = batch_size
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.train_dest)
            self.train_dataset = Airplane_Dataset(train_df, self.id_var, self.time_lenght)
            test_df = pd.read_csv(self.test_dest)
            self.test_dataset = Airplane_Dataset(test_df, self.id_var, self.time_lenght)
        if stage == "test" or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()