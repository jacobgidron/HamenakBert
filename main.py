import cProfile
import pstats

import hydra
from omegaconf import DictConfig, OmegaConf
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
from metrics import format_output_y1

from run_tests import CSV_HEAD

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


def setup_model(base_path, train_data, val_data, test_data, maxlen, minlen, lr, dropout, train_batch_size,
                val_batch_size, max_epochs, min_epochs, weighted_loss, **kwargs):
    print(MODEL)
    # init data module
    if not os.path.exists(MODEL):
        os.mkdir(MODEL)
        gdown.download_folder("https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing",
                              output=MODEL)

    dm = HebrewDataModule(
        train_paths=[base_path + '/' + path for path in train_data],
        val_path=[base_path + '/' + path for path in val_data],
        test_paths=[base_path + '/' + path for path in test_data],
        # train_paths=train_data,
        # val_path=val_data,
        # test_paths=test_data,
        model=MODEL,
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
    model = MenakBert(model=MODEL,
                      dropout=dropout,
                      train_batch_size=train_batch_size,
                      lr=lr,
                      max_epochs=max_epochs,
                      min_epochs=min_epochs,
                      n_warmup_steps=warmup_steps,
                      n_training_steps=total_training_steps,
                      weights=weights)
    return model, dm


def setup_trainer(max_epochs):
    # config training
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min"
    )
    # logger = CSVLogger("lightning_csv_logs", name="nikkud_logs")
    logger = TensorBoardLogger("lightning_logs", name="nikkud_logs")
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5)

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=max_epochs,
        gpus=1,
        progress_bar_refresh_rate=1,
        log_every_n_steps=100
    )
    return trainer


def test_by_file(model, trainer, test_path, target_dir, tokenizer, max_len, min_len):
    trained_model = MenakBert.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
    )
    trained_model.freeze()
    model.freeze()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for root, dirs, files in os.walk(test_path):
        for name in files:
            curr_in = os.path.join(root, name)
            curr_out = os.path.join(target_dir, name)
            val_dataset = textDataset(
                [curr_in],
                max_len,
                min_len,
                tokenizer
            )
            loader = DataLoader(val_dataset, batch_size=100, num_workers=40)
            preds = trainer.predict(model=trained_model, dataloaders=loader, return_predictions=True)
            preds = torch.argmax(preds, dim=-1)
            with open(curr_out, 'a', encoding='utf8') as f:
                for sent in range(len(val_dataset)):
                    line = format_output_y1(val_dataset[sent], preds['N'][sent], preds['D'][sent], preds['S'][sent])
                    f.write(f'{line}\n')


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



# def create_testfiles(model, test_path, tokenizer):




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
    os.environ['TOKENIZERS_PARALLELISM']='true'
    global MODEL
    MODEL = f"{cfg.base_path}/tavbert"
    print(OmegaConf.to_yaml(cfg))

    print(os.getcwd())

    # all model params from the CFG
    params = {
        "train_data": [cfg.dataset.train_path],
        "val_data": [cfg.dataset.val_path],
        "test_data": [cfg.dataset.test_path],
        "maxlen": cfg.dataset.max_len,
        "minlen": cfg.dataset.min_len,
        "lr": cfg.hyper_params.lr,
        "dropout": cfg.hyper_params.dropout,
        "train_batch_size": cfg.hyper_params.train_batch_size,
        "val_batch_size": cfg.hyper_params.val_batch_size,
        "max_epochs": cfg.hyper_params.max_epochs,
        "min_epochs": cfg.hyper_params.min_epochs,
        "weighted_loss": cfg.hyper_params.weighted_loss,
        "path": os.getcwd()
    }

    model, dm = setup_model(cfg.base_path, **params)
    trainer = setup_trainer(params['max_epochs'])
    # trainer.tune(model)
    with cProfile.Profile() as pr:
        trainer.fit(model, dm)
    stat = pstats.Stats(pr)
    stat.dump_stats(filename="run_time.prof")
    trainer.test(model, dm)

    with open(f"{cfg.base_path}/result_tabel.csv", "a") as f:
        # writer = csv.writer(f)
        fin = params.copy()
        fin["acc_S"] = model.final_acc_S
        fin["acc_D"] = model.final_acc_D
        fin["acc_N"] = model.final_acc_N

        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(fin)


# eval_model(complete_trainer, dm, cfg.datset.val_path, cfg.dataset.max_len)

# def run_with_globals():
#     model, dm = setup_model(train_data=train_data, val_data=val_data, test_data=test_data, maxlen=MAX_LEN,
#                             minlen=MIN_LEN, lr=LR, dropout=DROPOUT, train_batch_size=Train_BatchSize,
#                             val_batch_size=Val_BatchSize, max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS,
#                             weighted_loss=True)
#     trainer = setup_trainer()
#     trainer.fit(model, dm)
#     trainer.test(model, dm)


if __name__ == '__main__':
    runModel()

    # # run_with_globals()
    # params = {
    #     "train_data": train_data,
    #     "val_data": val_data,
    #     "test_data": test_data,
    #     "model": MODEL,
    #     "maxlen": MAX_LEN,
    #     "minlen": MIN_LEN,
    #     "lr": LR,
    #     "dropout": DROPOUT,
    #     "train_batch_size": Train_BatchSize,
    #     "val_batch_size": Val_BatchSize,
    #     "max_epochs": MAX_EPOCHS,
    #     "min_epochs": MIN_EPOCHS,
    #     "weighted_loss": True
    # }
    #
    # model, dm = setup_model(**params)
    # # model, dm = setup_model(train_data=train_data,
    # #                         val_data=val_data,
    # #                         test_data=test_data,
    # #                         model=MODEL,
    # #                         maxlen=MAX_LEN,
    # #                         minlen=MIN_LEN,
    # #                         lr=LR,
    # #                         dropout=DROPOUT,
    # #                         train_batch_size=Train_BatchSize,
    # #                         val_batch_size=Val_BatchSize,
    # #                         max_epochs=MAX_EPOCHS,
    # #                         min_epochs=MIN_EPOCHS,
    # #                         weighted_loss=True)
    # trainer = setup_trainer()
    # trainer.fit(model, dm)
    # trainer.test(model, dm)
    # with open("result_tabel.csv", "a") as f:
    #     # writer = csv.writer(f)
    #     fin = params.copy()
    #     fin["acc_S"] = model.final_acc_S
    #     fin["acc_D"] = model.final_acc_D
    #     fin["acc_N"] = model.final_acc_N
    #     writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
    #     writer.writerow(params)
