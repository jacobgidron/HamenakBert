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

model = MenakBert.load_from_checkpoint(Path(r"C:\Users\jacob\Downloads\Telegram Desktop\HamenakBert\full_traind\checkpoints") / "epoch=17-step=57761.ckpt")