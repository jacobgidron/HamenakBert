import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from dataset import textDataset, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
from datasets import load_metric
import HebrewDataModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

MAX_LEN = 100  # todo make global


class MenakBert(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("tau/tavbert-he")
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

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
        return loss

if __name__ == "__main__":
    seed_everything(42)

    TRAIN_PATH = '/content/Bert_data/train'
    VAL_PATH = '/content/Bert_data/val'
    TEST_PATH = '/content/Bert_data/test'

    train_data = [TRAIN_PATH]
    val_data = [VAL_PATH]
    test_data = [TEST_PATH]
    dm = HebrewDataModule(train_data, val_data, test_data)
    dm.setup()

    model = MenakBert()