from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from typing import Literal

class Twice_LSTM(L.LightningModule):
    def __init__(self, model_name, input_size, hidden_size, num_layers, time_lenght, drop, lr) -> None:
        super().__init__()
        if model_name == "lstm":
            self.rnn = nn.LSTM(
                input_size = input_size, 
                hidden_size = hidden_size, 
                num_layers = num_layers,
                batch_first = True,
                dropout = drop
            )
        if model_name == "gru":
            self.rnn = nn.GRU(
                input_size = input_size, 
                hidden_size = hidden_size, 
                num_layers = num_layers,
                batch_first = True,
                dropout = drop
            )
        if model_name == "rnn":
            self.rnn = nn.RNN(
                input_size = input_size, 
                hidden_size = hidden_size, 
                num_layers = num_layers,
                batch_first = True,
                dropout = drop
            )

        self.head = nn.Linear(
            in_features = time_lenght,
            out_features=1
        )

        self.metric_accuracy = BinaryAccuracy()
        self.metric_precision = BinaryPrecision()
        self.metric_recall = BinaryRecall()
        self.metric_f1 = BinaryF1Score()
        self.lr = lr
        self.save_hyperparameters()
    
    def forward(self, x) -> Any:
        y_hat, _ = self.rnn(x)
        y_hat = y_hat[:, :, -1]
        y_hat = self.head(y_hat)
        return torch.sigmoid(y_hat)
    
    def loss(self, y, y_hat):
        return F.binary_cross_entropy(y_hat, y)
    
    def log_everything(self, step_type:Literal["train", "val", "test"], y, y_hat, loss_func):
        self.log(f"{step_type}_loss", loss_func)
        self.log(f"{step_type}_acc", self.metric_accuracy(y_hat, y))
        self.log(f"{step_type}_precision", self.metric_precision(y_hat, y))
        self.log(f"{step_type}_recall", self.metric_recall(y_hat, y))
        self.log(f"{step_type}_f1", self.metric_f1(y_hat, y))

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        y_hat = self(x)
        loss_func = self.loss(y, y_hat)
        self.log_everything(step_type="train", y=y, y_hat=y_hat, loss_func=loss_func)
        return loss_func

    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        y_hat = self(x)
        loss_func = self.loss(y, y_hat)
        self.log_everything(step_type="val", y=y, y_hat=y_hat, loss_func=loss_func)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer