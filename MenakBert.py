import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoModel, get_linear_schedule_with_warmup
from dataset import NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
from datasets import load_metric
from HebrewDataModule import HebrewDataModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np


class MenakBert(LightningModule):
    def __init__(self,
                 model,
                 lr,
                 dropout,
                 train_batch_size,
                 max_epochs,
                 min_epochs,
                 n_training_steps=None,
                 n_warmup_steps=None
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)
        self.dropout = nn.Dropout(0.2)
        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.train_batch_size = train_batch_size
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, label=None):
        """
        :return: tuple of 3 tensor batch,sentence_len,classes
        """

        last_hidden_state = self.model(input_ids, attention_mask)['last_hidden_state']
        n = self.linear_N(last_hidden_state)
        d = self.linear_D(last_hidden_state)
        s = self.linear_S(last_hidden_state)
        output = dict(N=n, D=d, S=s, L=last_hidden_state)

        loss = 0
        if label is not None:
            loss_n = F.cross_entropy(n.permute(0, 2, 1), label["N"].long(), ignore_index=-1)
            loss_d = F.cross_entropy(d.permute(0, 2, 1), label["D"].long(), ignore_index=-1)
            loss_s = F.cross_entropy(s.permute(0, 2, 1), label["S"].long(), ignore_index=-1)
            loss = loss_n + loss_d + loss_s
        return loss, output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_epoch_end(self, output_results):
        logits_N = torch.cat([tmp["predictions"]["N"] for tmp in output_results])
        logits_D = torch.cat([tmp["predictions"]["D"] for tmp in output_results])
        logits_S = torch.cat([tmp["predictions"]["S"] for tmp in output_results])

        pred_N = np.argmax(logits_N.cpu().detach().numpy(), axis=-1)
        pred_D = np.argmax(logits_D.cpu().detach().numpy(), axis=-1)
        pred_S = np.argmax(logits_S.cpu().detach().numpy(), axis=-1)

        labels_N = torch.cat([tmp["labels"]["N"] for tmp in output_results]).cpu().detach().numpy()
        labels_D = torch.cat([tmp["labels"]["D"] for tmp in output_results]).cpu().detach().numpy()
        labels_S = torch.cat([tmp["labels"]["S"] for tmp in output_results]).cpu().detach().numpy()

        N_classes = ["none", "rafa", "shva", "hataph segol", "hataph patah", "hataph kamats", "hirik", "chere", "segol",
                     "phatah",
                     "kamats", "hulam(full)", "hulam", "kubuch", "shuruk", "phatah"]
        S_classes = ["NONE", "mask", "sin", "shin"]
        D_classes = ["NONE", "RAFE", "DAGESH"]

        cf_matrix_N = confusion_matrix(labels_N.flatten(),
                                       pred_N.flatten(),
                                       labels=[i for i in range(NIQQUD_SIZE)],
                                       normalize="true")
        cf_matrix_S = confusion_matrix(labels_S.flatten(),
                                       pred_S.flatten(),
                                       labels=[i for i in range(SIN_SIZE)],
                                       normalize="true")
        cf_matrix_D = confusion_matrix(labels_D.flatten(),
                                       pred_D.flatten(),
                                       labels=[i for i in range(DAGESH_SIZE)],
                                       normalize="true")

        df_cm_N = pd.DataFrame(cf_matrix_N, index=[i for i in N_classes], columns=[i for i in N_classes])
        df_cm_D = pd.DataFrame(cf_matrix_D, index=[i for i in D_classes], columns=[i for i in D_classes])
        df_cm_S = pd.DataFrame(cf_matrix_S, index=[i for i in S_classes], columns=[i for i in S_classes])
        self.log(value=torch.tensor(cf_matrix_N).long(), name="N Confusion Matrix test !!!")
        # self.log(value=cf_matrix_N, name="N Confusion Matrix")
        # self.log(value=cf_matrix_D, name="D Confusion Matrix")
        # self.log(value=cf_matrix_S, name="S Confusion Matrix")

        # display confusion matrix
        # plt.figure(figsize=(12, 7))
        # ax = ConfusionMatrixDisplay.from_predictions(labels_N.flatten(), pred_N.flatten(),
        #                                              labels=[i for i in range(16)], display_labels=classes,
        #                                              normalize="true")
        # fig = plt.gcf()
        # fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        # plt.savefig("N Confusion Matrix")
        # plt.close(fig)

        # plt.figure(figsize=(12, 7))
        # ax = ConfusionMatrixDisplay.from_predictions(labels_D.flatten(), pred_D.flatten(),
        #                                              labels=[i for i in range(3)],
        #                                              display_labels=["NONE", "RAFE", "DAGESH"],
        #                                              normalize="true")
        # fig = plt.gcf()
        # fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        # plt.savefig("D Confusion Matrix")

        # plt.figure(figsize=(12, 7))
        # ax = ConfusionMatrixDisplay.from_predictions(labels_S.flatten(), pred_S.flatten(),
        #                                              labels=[i for i in range(4)],
        #                                              display_labels=["NONE", "mask", "sin", "shin"],  # verify
        #                                              normalize="true")
        #
        # fig = plt.gcf()
        # fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        # plt.savefig("S Confusion Matrix")
        # plt.close(fig)


if __name__ == "__main__":
    temp_Val_BatchSize = 32
    seed_everything(42)

    TRAIN_PATH = '/content/Bert_data/train'
    VAL_PATH = '/content/Bert_data/val'
    TEST_PATH = '/content/Bert_data/test'

    train_data = [TRAIN_PATH]
    val_data = [VAL_PATH]
    test_data = [TEST_PATH]
    dm = HebrewDataModule(train_data, val_data, test_data,
                          # train_batch_size=Train_BatchSize,
                          val_batch_size=temp_Val_BatchSize)
    dm.setup()

    model = MenakBert()
