from typing import Tuple, List
import random
import numpy as np

from cachier import cachier
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import pre_processing
import utils


class CharacterTable:
    MASK_TOKEN = ''

    def __init__(self, chars):
        # make sure to be consistent with JS
        self.chars = [CharacterTable.MASK_TOKEN] + chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [
            [self.char_indices[c] for c in cs] for cs in css
        ]

    def __repr__(self):
        return repr(self.chars)


letters_table = CharacterTable(pre_processing.SPECIAL_TOKENS + pre_processing.VALID_LETTERS)
dagesh_table = CharacterTable(pre_processing.DAGESH)
sin_table = CharacterTable(pre_processing.NIQQUD_SIN)
niqqud_table = CharacterTable(pre_processing.NIQQUD)

LETTERS_SIZE = len(letters_table)
NIQQUD_SIZE = len(niqqud_table)
DAGESH_SIZE = len(dagesh_table)
SIN_SIZE = len(sin_table)

tokenize = AutoTokenizer.from_pretrained("tau/tavbert-he")

nikud_to_id_dict = {}
id_to_nikud_dict = {}

count = 0
for niqqud in range(NIQQUD_SIZE ):
    for sin in range(SIN_SIZE):
        for dag in range(DAGESH_SIZE):
            nikud_to_id_dict[(niqqud, dag, sin)] = count
            id_to_nikud_dict[count] = (niqqud, sin, dag)
            count += 1


def print_tables():
    print('const ALL_TOKENS =', letters_table.chars, end=';\n')
    print('const niqqud_array =', niqqud_table.chars, end=';\n')
    print('const dagesh_array =', dagesh_table.chars, end=';\n')
    print('const sin_array =', sin_table.chars, end=';\n')


def from_categorical(t):
    return np.argmax(t, axis=-1)


def merge(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            if pre_processing.can_dagesh(t):
                sentence.append(dagesh_table.indices_char[d].replace(pre_processing.RAFE, ''))
            if pre_processing.can_sin(t):
                sentence.append(sin_table.indices_char[s].replace(pre_processing.RAFE, ''))
            if pre_processing.can_niqqud(t):
                sentence.append(niqqud_table.indices_char[n].replace(pre_processing.RAFE, ''))
        res.append(''.join(sentence))
    return res


class textDataset(Dataset):
    def __init__(self, base_paths, maxlen):
        """
        Args:
        """

        def pad(ords, dtype='int32', value=0):
            return utils.pad_sequences(ords, maxlen=maxlen+1, dtype=dtype, value=value)

        self.data = []
        corpora = read_corpora(base_paths)
        for (filename, heb_items) in corpora:
            text, normalized, dagesh, sin, niqqud = zip(
                *(zip(*line) for line in pre_processing.split_by_length(heb_items, maxlen)))


            niqqud = pad(niqqud_table.to_ids(niqqud))
            dagesh = pad(dagesh_table.to_ids(dagesh))
            sin = pad(sin_table.to_ids(sin))

            for i in range(len(text)):
                item = {
                    "text": "".join(normalized[i]),
                    "y1": {'N': niqqud[i], 'D': dagesh[i], 'S': sin[i]},
                    "y2": [nikud_to_id_dict[(niqqud[i][j], dagesh[i][j], sin[i][j])] for j in range(len(niqqud[i]))]
                }
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def read_corpora(base_paths):
    return tuple(
        [(filename, list(pre_processing.iterate_file(filename))) for filename in utils.iterate_files(base_paths)])


if __name__ == '__main__':
    # data = Data.concatenate([Data.from_text(x, maxlen=64) for x in read_corpora(['test1.txt'])])
    # data.print_stats()
    # print(np.concatenate([data.normalized[:1], data.sin[:1]]))
    # res = merge(data.text[:1], data.normalized[:1], data.niqqud[:1], data.dagesh[:1], data.sin[:1])
    # print(res)
    # train_dict = {}
    # train_dict["test"] = get_xy(load_data(tuple(['test1.txt', 'test2.txt']), maxlen=16).shuffle())
    testData = textDataset(tuple(['test1.txt', 'test2.txt']), 16)
    print_tables()
    print(letters_table.to_ids(["שלום"]))

# load_data.clear_cache()
