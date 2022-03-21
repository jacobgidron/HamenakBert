import os
from MenakBert import MenakBert
from dataset import textDataset
from torch.utils.data import DataLoader
import torch
from metrics import format_output_y1

def compare_by_file(
                 test_path,
                 output_dir,
                 processed_dir,
                 tokenizer,
                 max_len,
                 min_len,
                 checkpoint_path=r'C:\Users\itsid\PycharmProjects\HamenakBert\lightning_logs\nikkud_logs\full_traind\checkpoints\epoch=17-step=57761.ckpt'
):
    trained_model = MenakBert.load_from_checkpoint(checkpoint_path)
    trained_model.freeze()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    for root, dirs, files in os.walk(test_path):
        for name in files:
            curr_in = os.path.join(root, name)
            curr_out = os.path.join(output_dir, name)
            curr_pre = os.path.join(processed_dir, name)
            val_dataset = textDataset(
                [curr_in],
                max_len,
                min_len,
                tokenizer
            )
            loader = DataLoader(val_dataset, batch_size=100, num_workers=12)

            with open(curr_out, 'a', encoding='utf8') as f:
                for batch in loader:
                    _, preds = trained_model(batch['input_ids'], batch['attention_mask'])
                    preds['N'] = torch.argmax(preds['N'], dim=-1)
                    preds['D'] = torch.argmax(preds['D'], dim=-1)
                    preds['S'] = torch.argmax(preds['S'], dim=-1)
                    for sent in range(len(preds['N'])):
                        line = format_output_y1(val_dataset[sent]['input_ids'], preds['N'][sent], preds['D'][sent], preds['S'][sent], tokenizer)
                        f.write(f'{line}\n')
            with open(curr_pre, 'a', encoding='utf8') as f:
                for sent in range(len(val_dataset)):
                    line = format_output_y1(val_dataset[sent]['input_ids'],
                                            val_dataset[sent]['label']['N'],
                                            val_dataset[sent]['label']['D'],
                                            val_dataset[sent]['label']['S'],
                                            tokenizer)
                    f.write(f'{line}\n')


def create_compare_file(
                 test_path,
                 target_dir,
                 tokenizer,
                 max_len,
                 min_len
                 ):
    trained_model = MenakBert.load_from_checkpoint(
        r'C:\Users\itsid\PycharmProjects\HamenakBert\lightning_logs\nikkud_logs\full_traind\checkpoints\epoch=17-step=57761.ckpt'
    )
    trained_model.freeze()
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
            with open(curr_out, 'a', encoding='utf8') as f:
                for sent in range(len(val_dataset)):
                    line = format_output_y1(val_dataset[sent]['input_ids'],
                                            val_dataset[sent]['label']['N'],
                                            val_dataset[sent]['label']['D'],
                                            val_dataset[sent]['label']['S'],
                                            tokenizer)
                    f.write(f'{line}\n')