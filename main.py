import csv

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
from HebrewDataModule import HebrewDataModule
from MenakBert import MenakBert
from dataset import textDataset
from pytorch_lightning import Trainer, seed_everything
import gdown
import os
from transformers import AutoTokenizer
from evaluation import compare_by_file_from_model, compare_by_file_from_checkpoint
from pathlib import Path
from metrics import all_stats

from pathlib import Path
# from run_tests import CSV_HEAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_LINK = "https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing"
MODEL = "tau/tavbert-he"

BASE_PATH = Path()

TRAIN_PATH = BASE_PATH / "hebrew_diacritized" / "dev_mock_data" / "train"
VAL_PATH = BASE_PATH / "hebrew_diacritized" / "dev_mock_data" / "validation"
TEST_PATH = BASE_PATH / "hebrew_diacritized" / "dev_mock_data" / "test"
Val_BatchSize = 32
train_data = [TRAIN_PATH]
val_data = [VAL_PATH]
test_data = [TEST_PATH]
DROPOUT = 0.1
Train_BatchSize = 32
LR = 1e-5
MAX_EPOCHS = 100
MIN_EPOCHS = 5
MAX_LEN = 100
MIN_LEN = 10


def setup_dm(train_data, val_data, test_data, model, maxlen, minlen, train_batch_size, val_batch_size):
    dm = HebrewDataModule(
        train_paths=train_data,
        val_path=val_data,
        test_paths=test_data,
        model=model,
        max_seq_length=maxlen,
        min_seq_length=minlen,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size)
    dm.setup()
    return dm


def calc_weights(train_data):
    tmp = train_data.counter['N']
    n_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    n_weights = (n_weights / n_weights.sum()).to(device)

    tmp = train_data.counter['D']
    d_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    d_weights = (d_weights / d_weights.sum()).to(device)

    tmp = train_data.counter['S']
    s_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    s_weights = (s_weights / s_weights.sum()).to(device)

    weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}
    return weights


# def setup_model(data_len, lr, dropout, train_batch_size, max_epochs, min_epochs, linear_size, weights=None):
#     # init data module
#     if not os.path.exists(MODEL):
#         os.mkdir(MODEL)
#         gdown.download_folder("https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing",
#                               output=MODEL)

#     # steps_per_epoch =
#     # total_training_steps =
#     # warmup_steps =

#     # init module
#     model = MenakBert(model=MODEL,
#                       dropout=dropout,
#                       train_batch_size=train_batch_size,
#                       lr=lr,
#                       linear_size=linear_size,
#                       max_epochs=max_epochs,
#                       min_epochs=min_epochs,
#                     #   n_warmup_steps=warmup_steps,
#                     #   n_training_steps=total_training_steps,
#                       weights=weights)
#     return model


def setup_trainer(max_epochs):
    # config training
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="f1_score_N",
        mode="max"
    )

    logger = TensorBoardLogger("lightning_logs", name="nikkud_logs")
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5)

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=max_epochs,
        gpus=1,
        progress_bar_refresh_rate=100,
        log_every_n_steps=100
    )
    return trainer


if __name__ == '__main__':
    seed_everything(42, workers=True)
    # runModel()
    params = {
        "train_data": TRAIN_PATH,
        "val_data": VAL_PATH,
        "test_data": TEST_PATH,
        "model": MODEL,
        "maxlen": MAX_LEN,
        "minlen": MIN_LEN,
        "lr": LR,
        "dropout": DROPOUT,
        "train_batch_size": Train_BatchSize,
        "val_batch_size": Val_BatchSize,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "weighted_loss": True,
        "linear_size": 1024
    }
    
    base_path = params['train_data']
    dirs = ['religion', 'pre_modern', 'early_modern', 'modern']
    testpaths = [None, None, None, params["test_data"]]
    data_modules = []

    for testpath, directory in zip(testpaths,dirs):
        train_path = base_path / directory
        dm = HebrewDataModule(
            train_paths=[train_path],
            val_path=[params['val_data']],
            test_paths=[testpath],
            model=MODEL,
            max_seq_length=params['maxlen'],
            min_seq_length=params['minlen'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['val_batch_size'])
        dm.setup()
        data_modules.append(dm)
        len(data_modules[0].train_data)
        model = MenakBert(model=MODEL,
                          dropout= params["dropout"],
                          train_batch_size= params["train_batch_size"],
                          lr= params["lr"],
                          linear_size= params["linear_size"],
                          max_epochs= params["max_epochs"],
                          min_epochs= params["min_epochs"],
                          #   n_warmup_steps=warmup_steps,
                          #   n_training_steps=total_training_steps,
                          weights=params['weighted_loss'])
    for i, dm in enumerate(data_modules):
        steps_per_epoch = len(dm.train_data) // params['train_batch_size']
        total_training_steps = steps_per_epoch * params['max_epochs']
        warmup_steps = total_training_steps // 5
        model.n_warmup_steps = warmup_steps
        model.n_training_steps = total_training_steps
        trainer = setup_trainer(params['max_epochs'])
        trainer.fit(model, dm)
    trainer.test(model, data_modules[-1])
    
    model, dm = setup_model( **params)
    trainer = setup_trainer(params['max_epochs'])
    # trainer.tune(model)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    with open("result_tabel.csv", "a") as f:
        # writer = csv.writer(f)
        fin = params.copy()
        fin["acc_S"] = model.final_acc_S
        fin["acc_D"] = model.final_acc_D
        fin["acc_N"] = model.final_acc_N
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(params)
    
    
    model, dm = setup_model(**params)
    model, dm = setup_model(train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            model=MODEL,
                            maxlen=MAX_LEN,
                            minlen=MIN_LEN,
                            lr=LR,
                            dropout=DROPOUT,
                            train_batch_size=Train_BatchSize,
                            val_batch_size=Val_BatchSize,
                            max_epochs=MAX_EPOCHS,
                            min_epochs=MIN_EPOCHS,
                            weighted_loss=True)
    trainer = setup_trainer(MAX_EPOCHS)
    # with cProfile.Profile() as pr:
    #     trainer.fit(model, dm)
    # stat = pstats.Stats(pr)
    # stat.dump_stats(filename="run_time.prof")
    trainer.test(model, dm)
    CSV_HEAD = [
        "train_data",
        "val_data",
        "test_data",
        "model",
        "maxlen",
        "minlen",
        "lr",
        "dropout",
        "train_batch_size",
        "val_batch_size",
        "max_epochs",
        "min_epochs",
        "weighted_loss",
        "acc_S",
        "acc_D",
        "acc_N",
    ]
    # with open("result_tabel.csv", "a") as f:
    #     # writer = csv.writer(f)
    #     fin = params.copy()
    #     fin["acc_S"] = model.final_acc_S.item()
    #     fin["acc_D"] = model.final_acc_D.item()
    #     fin["acc_N"] = model.final_acc_N.item()
    #     writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
    #     writer.writerow(fin)
