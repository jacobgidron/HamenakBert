from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from dataset import textDataset
# from MenakBert import  , MODEL
from transformers import AutoTokenizer


class HebrewDataModule(LightningDataModule):

    def __init__(
            self,
            train_paths,
            val_path,
            model,
            max_seq_length: int,
            min_seq_length: int,
            train_batch_size: int,
            val_batch_size: int,
            test_paths=None,
            **kwargs,
    ):
        super().__init__()
        self.model = model
        self.train_paths = train_paths
        self.val_paths = val_path
        self.test_paths = test_paths
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.min_seq_length = min_seq_length
        self.val_batch_size = val_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)

    def setup(self, stage: str = None):
        self.train_data = textDataset(
            self.train_paths,
            self.max_seq_length,
            self.min_seq_length,
            self.tokenizer
        )

        self.val_data = textDataset(
            self.val_paths,
            self.max_seq_length,
            self.min_seq_length,
            self.tokenizer
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, num_workers=12)

    def test_dataloader(self):
        self.test_data = textDataset(
            self.test_paths,
            self.max_seq_length,
            self.min_seq_length,
            self.tokenizer
        )
        return DataLoader(self.test_data, batch_size=self.val_batch_size, num_workers=12)


if __name__ == '__main__':
    dm = HebrewDataModule("tau/tavbert-he")
    dm.prepare_data()
    for batch in dm.train_dataloader():
         break