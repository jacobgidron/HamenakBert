from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from HebrewDataModule import HebrewDataModule
from MenakBert import MenakBert, Train_BatchSize, MAX_LEN, MAX_EPOCHS
from dataset import textDataset
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

seed_everything(42)

TRAIN_PATH = 'hebrew_diacritized/train'
VAL_PATH = 'hebrew_diacritized/validation'
TEST_PATH = 'hebrew_diacritized/test_modern'

train_data = [TRAIN_PATH]
val_data = [VAL_PATH]
test_data = [TEST_PATH]

# init data module
dm = HebrewDataModule(train_data, val_data, test_data)
dm.setup()

steps_per_epoch = len(dm.train_data) // Train_BatchSize
total_training_steps = steps_per_epoch * MAX_EPOCHS
warmup_steps = total_training_steps // 5

# init module
model = MenakBert(n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)

# config training
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="train_loss",
    mode="min"
)
logger = TensorBoardLogger("lightning_logs", name="nikkud_logs")
early_stopping_callback = EarlyStopping(monitor='train_loss', patience=200)

trainer = Trainer(
    logger=logger,
    auto_lr_find=True,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=20,
    gpus=1,
    progress_bar_refresh_rate=1,
    log_every_n_steps=1
)
trainer.tune(model)
trainer.fit(model, dm)

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

input = sample_batch["input_ids"]
mask = sample_batch["attention_mask"]
_, predictions = trained_model(input, mask)

pred = np.argmax(predictions['N'].cpu().detach().numpy(), axis=-1)
labels = sample_batch["label"]['N'].cpu().detach().numpy()

plt.figure(figsize=(12, 7))
classes = ["none", "rafa", "shva", "hataph segol", "hataph patah", "hataph kamats", "hirik", "chere", "segol", "phatah",
           "kamats", "hulam(full)", "hulam", "kubuch", "shuruk", "phatah"]
print(len(classes))
ax = ConfusionMatrixDisplay.from_predictions(labels.flatten(), pred.flatten(), display_labels=classes, normalize="true")
fig = plt.gcf()
fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
plt.show()
