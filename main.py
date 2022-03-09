import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
import torch
from HebrewDataModule import HebrewDataModule
from MenakBert import MenakBert
from dataset import textDataset
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import gdown
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42, workers=True)

# MODEL_LINK = "https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing"
MODEL = "tavbert"

# TRAIN_PATH = 'hebrew_diacritized/train'
# VAL_PATH = 'hebrew_diacritized/validation'
# TEST_PATH = 'hebrew_diacritized/test_modern'
# TRAIN_PATH = r"hebrew_diacritized/check/train"
# VAL_PATH = "hebrew_diacritized/check/val"
# TEST_PATH = "hebrew_diacritized/check/test"
# Val_BatchSize = 32
# train_data = [TRAIN_PATH]
# val_data = [VAL_PATH]
# test_data = [TEST_PATH]
# DROPOUT = 0.1
# Train_BatchSize = 32
# LR = 1e-5
# MAX_EPOCHS = 100
# MIN_EPOCHS = 5
# MAX_LEN = 100
# MIN_LEN = 10


def setup_model(base_path, train_data, val_data, test_data, model, maxlen, minlen, lr, dropout, train_batch_size,
                val_batch_size, max_epochs, min_epochs, weighted_loss):
    # init data module
    if not os.path.exists("tavbert"):
        os.mkdir("tavbert")
        gdown.download_folder("https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing",
                              output="tavbert")

    dm = HebrewDataModule(
        train_paths=base_path + '/' + train_data,
        val_path=base_path + '/' + val_data,
        test_paths=base_path + '/' + test_data,
        model=model,
        max_seq_length=maxlen,
        min_seq_length=minlen,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size)
    dm.setup()

    weights = None
    if weighted_loss:
        tmp = dm.train_data.counter['N']
        n_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
        n_weights = (n_weights.sum() / n_weights).to(device)

        tmp = dm.train_data.counter['D']
        d_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
        d_weights = (d_weights.sum() / d_weights).to(device)

        tmp = dm.train_data.counter['S']
        s_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
        s_weights = (s_weights.sum() / s_weights).to(device)

        weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}

    steps_per_epoch = len(dm.train_data) // train_batch_size
    total_training_steps = steps_per_epoch * max_epochs
    warmup_steps = total_training_steps // 5

    # init module
    model = MenakBert(model=model,
                      dropout=dropout,
                      train_batch_size=train_batch_size,
                      lr=lr,
                      max_epochs=max_epochs,
                      min_epochs=min_epochs,
                      n_warmup_steps=warmup_steps,
                      n_training_steps=total_training_steps,
                      weights=weights)
    return model, dm

def setup_trainer():
    # config training
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min"
    )
    logger = CSVLogger("lightning_csv_logs", name="nikkud_logs")
    # logger = TensorBoardLogger("lightning_logs", name="nikkud_logs")
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5)

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=20,
        gpus=1,
        progress_bar_refresh_rate=1,
        log_every_n_steps=1
    )
    return trainer


def eval_model(trainer, dm, val_path, maxlen):
    # eval
    tokenizer = dm.tokenizer
    val_dataset = textDataset(
        [val_path],
        maxlen,
        tokenizer
    )

    sample_batch = next(iter(DataLoader(val_dataset, batch_size=8, num_workers=2)))

    trained_model = MenakBert.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
    )
    trained_model.eval()
    trained_model.freeze()

    # what is this
    input = sample_batch["input_ids"]
    mask = sample_batch["attention_mask"]
    _, predictions = trained_model(input, mask)

# pred = np.argmax(predictions['N'].cpu().detach().numpy(), axis=-1)
# labels = sample_batch["label"]['N'].cpu().detach().numpy()
#
# plt.figure(figsize=(12, 7))
# classes = ["none", "rafa", "shva", "hataph segol", "hataph patah", "hataph kamats", "hirik", "chere", "segol", "phatah",
#            "kamats", "hulam(full)", "hulam", "kubuch", "shuruk", "phatah"]
# ax = ConfusionMatrixDisplay.from_predictions(labels.flatten(), pred.flatten(), display_labels=classes, normalize="true")
# fig = plt.gcf()
# fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
# plt.show()

@hydra.main(config_path="config", config_name="config")
def runModel(cfg: DictConfig):
    model, dm = setup_model(cfg.base_path, cfg.dataset.train_path, cfg.dataset.val_path, cfg.dataset.test_path, MODEL,
                            cfg.dataset.max_len, cfg.dataset.min_len, cfg.hyper_params.lr, cfg.hyper_params.dropout,
                            cfg.hyper_params.train_batch_size, cfg.hyper_params.val_batch_size,
                            cfg.hyper_params.max_epochs, cfg.hyper_params.min_epochs, cfg.hyper_params.weighted_loss)
    trainer = setup_trainer()
    # trainer.tune(model)
    trainer.fit(model, dm)
    trainer.test(model, dm)

    #eval_model(complete_trainer, dm, cfg.datset.val_path, cfg.dataset.max_len)


if __name__ == '__main__':
    runModel()
