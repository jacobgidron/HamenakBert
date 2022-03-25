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

    inner_out = os.path.join(output_dir, "inner")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(inner_out)

    inner_pro = os.path.join(processed_dir, "inner")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        os.mkdir(inner_pro)
    counter = 0
    for root, dirs, files in os.walk(test_path):
        counter += 1
        for name in files:
            new_name = str(counter) + "_" + name
            curr_in = os.path.join(root, name)
            curr_out = os.path.join(inner_out, new_name)
            curr_pro = os.path.join(inner_pro, new_name)

            val_dataset = textDataset(
                [curr_in],
                max_len,
                min_len,
                tokenizer
            )
            loader = DataLoader(val_dataset, batch_size=32, num_workers=8)

            with open(curr_out, 'a', encoding='utf8') as f_out:
                with open(curr_pro, 'a', encoding='utf8') as f_pro:
                    for batch in loader:
                        _, preds = trained_model(batch['input_ids'], batch['attention_mask'])
                        preds['N'] = torch.argmax(preds['N'], dim=-1)
                        preds['D'] = torch.argmax(preds['D'], dim=-1)
                        preds['S'] = torch.argmax(preds['S'], dim=-1)

                        for sent in range(len(preds['N'])):
                            line_out = format_output_y1(batch['input_ids'][sent], preds['N'][sent], preds['D'][sent], preds['S'][sent], tokenizer)
                            f_out.write(f'{line_out}\n')
                            line_pro = format_output_y1(batch['input_ids'][sent], batch['label']['N'][sent], batch['label']['D'][sent], batch['label']['S'][sent], tokenizer)
                            f_pro.write(f'{line_pro}\n')
                # for sent in range(len(val_dataset)):
                #     line = format_output_y1(val_dataset[sent]['input_ids'],
                #                             torch.from_numpy(val_dataset[sent]['label']['N']),
                #                             torch.from_numpy(val_dataset[sent]['label']['D']),
                #                             torch.from_numpy(val_dataset[sent]['label']['S']),
                #                             tokenizer)
                #     f_pro.write(f'{line}\n')


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

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("tavbert", use_fast=True)
    compare_by_file(r"hebrew_diacritized/data/test", r"predicted", r"expected", tokenizer, 100, 5)
