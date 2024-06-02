import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoModel, get_linear_schedule_with_warmup
from dataset import NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE, PAD_INDEX
from HebrewDataModule import HebrewDataModule
import numpy as np
from torchmetrics import F1Score

from pre_processing import name_of, DAGESH, NIQQUD, NIQQUD_SIN


class MenakBert(LightningModule):
    def __init__(self,
                 model,
                 lr,
                 dropout,
                 train_batch_size,
                 max_epochs,
                 min_epochs,
                 linear_size = 1024,
                 weights=False,
                 n_training_steps=None,
                 n_warmup_steps=None,
                 ):
        super().__init__()
        import traceback
        print(traceback.format_stack())
        self.model = AutoModel.from_pretrained(model)
        self.model.hidden_dropout_prob = dropout
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)
        self.linear_up = nn.Linear(768, linear_size)
        self.linear_down = nn.Linear(linear_size, 768)
        self.reluLayer1 = nn.ReLU()
        self.reluLayer2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.train_batch_size = train_batch_size
        self.save_hyperparameters()
        self.f1 = F1Score(ignore_index=PAD_INDEX, mdmc_average="samplewise")
        self.weights = weights
        self.full_weights = None

    def update_steps(self,data_len,train_batch_size):
        steps_per_epoch = data_len // train_batch_size
        total_training_steps = steps_per_epoch * self.max_epochs

        self.n_training_steps = total_training_steps
        self.n_warmup_steps   = total_training_steps // 5
        self.train_batch_size = train_batch_size
        

    def forward(self, input_ids, attention_mask, label=None):
        """
        :return: tuple of 3 tensor batch, sentence_len, classes
        """

        last_hidden_state = self.model(input_ids, attention_mask)['last_hidden_state']
        large = self.linear_up(last_hidden_state)
        drop = self.dropout(large)
        active1 = self.reluLayer1(drop)
        small = self.linear_down(active1)
        active2 = self.reluLayer2(small)
        n = self.linear_N(active2)
        d = self.linear_D(active2)
        s = self.linear_S(active2)
        output = dict(N=n, D=d, S=s)

        loss = 0
        if label is not None:
            if self.weights:
                if not self.full_weights:
                    n_weights = torch.tensor([2.4920e-01, 2.6323e-01, 8.8410e-02, 1.0692e-03, 1.2552e-02, 1.4649e-04,
                                              7.2875e-02, 2.8470e-02, 4.6895e-02, 7.4532e-02, 8.5893e-02, 4.8047e-02,
                                              2.6159e-05, 2.8658e-02], device=self.device)
                    s_weights = torch.tensor([9.6249e-01, 3.2961e-04, 3.4374e-02, 2.8073e-03], device=self.device)
                    d_weights = torch.tensor([0.3914, 0.5126, 0.0961], device=self.device)
                    self.full_weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}
                loss_n = F.cross_entropy(n.permute(0, 2, 1), label["N"].long(), ignore_index=PAD_INDEX,
                                         weight=self.full_weights['N'])
                loss_d = F.cross_entropy(d.permute(0, 2, 1), label["D"].long(), ignore_index=PAD_INDEX,
                                         weight=self.full_weights['D'])
                loss_s = F.cross_entropy(s.permute(0, 2, 1), label["S"].long(), ignore_index=PAD_INDEX,
                                         weight=self.full_weights['S'])
            else:
                loss_n = F.cross_entropy(n.permute(0, 2, 1), label["N"].long(), ignore_index=PAD_INDEX)
                loss_d = F.cross_entropy(d.permute(0, 2, 1), label["D"].long(), ignore_index=PAD_INDEX)
                loss_s = F.cross_entropy(s.permute(0, 2, 1), label["S"].long(), ignore_index=PAD_INDEX)
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

    def validation_epoch_end(self, output_results):
        logits_N = torch.cat([tmp["predictions"]["N"] for tmp in output_results])
        logits_D = torch.cat([tmp["predictions"]["D"] for tmp in output_results])
        logits_S = torch.cat([tmp["predictions"]["S"] for tmp in output_results])

        pred_N = torch.argmax(logits_N, dim=-1)
        pred_D = torch.argmax(logits_D, dim=-1)
        pred_S = torch.argmax(logits_S, dim=-1)

        labels_N = torch.cat([tmp["labels"]["N"] for tmp in output_results])
        labels_D = torch.cat([tmp["labels"]["D"] for tmp in output_results])
        labels_S = torch.cat([tmp["labels"]["S"] for tmp in output_results])
        # self.log('valid_acc_N', self.f1, on_step=True, on_epoch=True) todo check how to use on_step=True, on_epoch=True to make the loss comut
        # self.log('valid_acc_S', self.valid_acc, on_step=True, on_epoch=True)
        # self.log('valid_acc_D', self.valid_acc, on_step=True, on_epoch=True)
        pec_S = self.f1(pred_S, labels_S)
        pec_D = self.f1(pred_D, labels_D)
        pec_N = self.f1(pred_N, labels_N)
        self.log('f1_score_S', pec_S, prog_bar=True, logger=True)
        self.log('f1_score_D', pec_D, prog_bar=True, logger=True)
        self.log('f1_score_N', pec_N, prog_bar=True, logger=True)
        return {'f1_score_S': pec_S,
                'f1_score_D': pec_D,
                'f1_score_N': pec_N,
                }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_epoch_end(self, output_results):
        logits_N = torch.cat([tmp["predictions"]["N"] for tmp in output_results])
        logits_D = torch.cat([tmp["predictions"]["D"] for tmp in output_results])
        logits_S = torch.cat([tmp["predictions"]["S"] for tmp in output_results])

        pred_N = torch.argmax(logits_N, dim=-1)
        pred_D = torch.argmax(logits_D, dim=-1)
        pred_S = torch.argmax(logits_S, dim=-1)

        labels_N = torch.cat([tmp["labels"]["N"] for tmp in output_results])
        labels_D = torch.cat([tmp["labels"]["D"] for tmp in output_results])
        labels_S = torch.cat([tmp["labels"]["S"] for tmp in output_results])

        self.final_acc_S = self.f1(pred_S, labels_S)
        self.final_acc_D = self.f1(pred_D, labels_D)
        self.final_acc_N = self.f1(pred_N, labels_N)

        labels_N = labels_N.cpu().detach().numpy()
        labels_D = labels_D.cpu().detach().numpy()
        labels_S = labels_S.cpu().detach().numpy()

        pred_N = pred_N.cpu().detach().numpy()
        pred_D = pred_D.cpu().detach().numpy()
        pred_S = pred_S.cpu().detach().numpy()

        N_classes = ["none", "rafa", "shva", "hataph segol",
                     "hataph patah", "hataph kamats",
                     "hirik", "chere", "segol", "phatah",
                     "kamats", "hulam", "kubuch", "shuruk"]
        S_classes = ["NONE", "mask", "sin", "shin"]
        D_classes = ["NONE", "RAFE", "DAGESH"]

        plt.figure(figsize=(12, 7))
        ax = ConfusionMatrixDisplay.from_predictions(labels_N.flatten(), pred_N.flatten(),
                                                     labels=[i for i in range(NIQQUD_SIZE)],
                                                     display_labels=N_classes,
                                                     # display_labels=["none"] + [name_of(char) for char in NIQQUD],
                                                     normalize="true")
        fig = plt.gcf()
        fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        self.logger.experiment.add_figure("N Confusion Matrix", fig)
        plt.close(fig)

        plt.figure(figsize=(12, 7))
        ax = ConfusionMatrixDisplay.from_predictions(labels_D.flatten(), pred_D.flatten(),
                                                     labels=[i for i in range(DAGESH_SIZE)],
                                                     display_labels=D_classes,
                                                     # display_labels=["none"] + [name_of(char) for char in DAGESH],
                                                     normalize="true")
        fig = plt.gcf()
        self.logger.experiment.add_figure("D Confusion Matrix", fig)
        plt.close(fig)

        fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        plt.figure(figsize=(12, 7))
        ax = ConfusionMatrixDisplay.from_predictions(labels_S.flatten(), pred_S.flatten(),
                                                     labels=[i for i in range(SIN_SIZE)],
                                                     display_labels=S_classes,
                                                     # display_labels=["none"] + [name_of(char) for char in NIQQUD_SIN],
                                                     normalize="true")
        fig = plt.gcf()
        fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        self.logger.experiment.add_figure("S Confusion Matrix", fig)
        plt.close(fig)

        # self.log('valid_acc_N', self.f1, on_step=True, on_epoch=True) todo check how to use on_step=True, on_epoch=True to make the loss comut
        # self.log('valid_acc_S', self.valid_acc, on_step=True, on_epoch=True)
        # self.log('valid_acc_D', self.valid_acc, on_step=True, on_epoch=True)

        return {'train_epoch_f1_precision_S': self.final_acc_S,
                'train_epoch_f1_precision_D': self.final_acc_D,
                'train_epoch_f1_precision_N': self.final_acc_N,
                }


if __name__ == "__main__":
    temp_Val_BatchSize = 32
    seed_everything(42)

    TRAIN_PATH = '/content/Bert_data/train'
    VAL_PATH = '/content/Bert_data/validation'
    TEST_PATH = '/content/Bert_data/test'

    train_data = [TRAIN_PATH]
    val_data = [VAL_PATH]
    test_data = [TEST_PATH]
    dm = HebrewDataModule(train_data, val_data, test_data,
                          # train_batch_size=Train_BatchSize,
                          val_batch_size=temp_Val_BatchSize)
    dm.setup()

    model = MenakBert()
