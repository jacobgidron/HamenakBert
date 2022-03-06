from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
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


seed_everything(42)

# MODEL_LINK = "https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing"
MODEL = "tavbert"

# TRAIN_PATH = 'hebrew_diacritized/train'
# VAL_PATH = 'hebrew_diacritized/validation'
# TEST_PATH = 'hebrew_diacritized/test_modern'
TRAIN_PATH = r"hebrew_diacritized/check/train"
VAL_PATH = "hebrew_diacritized/check/val"
TEST_PATH = "hebrew_diacritized/check/test"
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

def setup_model(train_data, val_data, test_data):
    # init data module
    if not os.path.exists("tavbert"):
        os.mkdir("tavbert")
        gdown.download_folder("https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing", output="tavbert")

    dm = HebrewDataModule(
        train_paths=train_data,
        val_path=val_data,
        test_paths=test_data,
        model=MODEL,
        max_seq_length=MAX_LEN,
        min_seq_length=MIN_LEN,
        train_batch_size=Train_BatchSize,
        val_batch_size=Val_BatchSize)
    dm.setup()


    steps_per_epoch = len(dm.train_data) // Train_BatchSize
    total_training_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = total_training_steps // 5

# init module
    model = MenakBert(model=MODEL,
                      dropout= DROPOUT,
                      train_batch_size= Train_BatchSize,
                      lr= LR,
                      max_epochs=MAX_EPOCHS,
                      min_epochs= MIN_EPOCHS,
                      n_warmup_steps=warmup_steps,
                      n_training_steps=total_training_steps)
    return model, dm

def train_model(model, dm):
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
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=7)

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=MAX_EPOCHS,
        min_epochs=MIN_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=1,
        log_every_n_steps=1
    )
    # trainer.tune(model)
    return trainer

def eval_model(trainer, dm):
    # eval
    tokenizer = dm.tokenizer
    val_dataset = textDataset(
        [VAL_PATH],
        MAX_LEN,
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


if __name__ == '__main__':
    model, dm = setup_model(train_data, val_data, test_data)
    trainer = train_model(model, dm)
    trainer.fit(model, dm)
    trainer.test(model, dm)
