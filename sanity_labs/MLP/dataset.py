import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def replace_ob_letters(letter):
    if letter == 'V':
        return 1
    if letter == 'W':
        return 9
    if letter == 'Y':
        return 14


class PeptideDataset(Dataset):

    def __init__(self, data_path, pos):

        # extract all polypeptides from text files
        data_file = os.path.join(data_path, 'pos_A0201.txt' if pos else 'neg_A0201.txt')
        with open(data_file, 'r') as pos_file:
            data = np.array(pos_file.read().splitlines())

        # convert letters to numbers in the range [1-20] (watch out for 'U', 'X', and 'Z')
        data = np.array([[(ord(x_i) - ord('A')) if ord(x_i) <= ord('T') else replace_ob_letters(x_i) for x_i in x] for x in data])
        data = torch.stack([F.one_hot(torch.tensor(x).to(torch.int64), num_classes=20) for x in data])

        self.data = data.float()
        self.labels = torch.tile(torch.tensor([pos]), [data.shape[0]])
        self.labels = self.labels.long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def split(self, ratio):
        train_size = int(ratio * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])
