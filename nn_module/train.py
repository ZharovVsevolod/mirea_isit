from some_rnn.data import DataModule
from some_rnn.models import Twice_LSTM
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

import hydra
from hydra.core.config_store import ConfigStore
from config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def models_train(cfg: Params):
    run_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    dm = DataModule(
        train_dest=cfg.dataset_path.train_dest,
        test_dest=cfg.dataset_path.test_dest,
        time_lenght=cfg.dataset.time_lenght,
        # id_var=cfg.dataset.id_var,
        batch_size=cfg.training.batch_size
    )
    model = Twice_LSTM(
        model_name=cfg.model.name,
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        time_lenght=cfg.dataset.time_lenght,
        drop=cfg.model.drop,
        lr=cfg.training.lr
    )

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project="rnn", name="gru_all_id", save_dir=run_directory + "/wandb")

    checkpoint = ModelCheckpoint(
        dirpath=run_directory + "/checkpoint",
        save_top_k=1,
        monitor="val_loss"
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint],
        default_root_dir=run_directory + "/trainer",
        log_every_n_steps=5,
        # fast_dev_run=5
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()

if __name__ == "__main__":
    L.seed_everything(1702)
    models_train()