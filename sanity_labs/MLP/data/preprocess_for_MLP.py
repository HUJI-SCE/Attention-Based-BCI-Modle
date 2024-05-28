import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from math import floor, ceil

from sanity_labs.utils.utils import DirMan


def replace_ob_letters(letter):
    if letter == 'V':
        return 1
    if letter == 'W':
        return 9
    if letter == 'Y':
        return 14


def train_test_split(total, ratio):
    lens = floor(ratio * len(total)), ceil(len(total) - ratio * len(total))
    return total.split(lens)


def extract_data(data_path):
    """
    This method preprocesses teh pep_data in the given path into one-hot vectors.

    :data_path: path to the file from the data needs to be extracted
    """

    # # extract all polypeptides from text files
    label_value = int('pos' in data_path)
    with open(data_path, 'r') as class_data:
        data = np.array(class_data.read().splitlines())

    # convert letters to numbers in the range [1-20] (watch out for 'U', 'X', and 'Z')
    data = np.array([[(ord(x_i) - ord('A')) if ord(x_i) <= ord('T') else replace_ob_letters(x_i)
                      for x_i in x] for x in data])
    data = torch.stack([F.one_hot(torch.tensor(x, dtype=torch.int64), num_classes=20).to(torch.float32).flatten()
                        for x in data])
    labels = label_value * torch.ones((data.shape[0], 1), dtype=torch.float32)

    return data, labels


def balance(train_pos, train_neg, test_pos, test_neg):
    """
    :pos_neg_ratio: ratio between datasets, must be > 1  and a digit
    """

    # create new - balanced - train and test sets
    train_ratio = len(train_neg) // len(train_pos)
    test_ratio = len(test_neg) // len(test_pos)
    train = torch.cat([train_pos.tile([train_ratio, 1])[:len(train_neg)], train_neg])
    test = torch.cat([test_pos.tile([test_ratio, 1])[:len(test_neg)], test_neg])

    return train, test


class PeptideDataset(Dataset):

    def __init__(self, data, labels):
        """
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_preprocessed_data(path_to_data):
    """
    This function runs through the stages of preprocessing the pep_data
    """

    # specify source and target data transformations in a dictionary
    src_tar_dict = {
        os.path.join(path_to_data, 'pep_data', '', 'neg_A0201.txt'): ['neg', 'neg_lbl'],
        os.path.join(path_to_data, 'pep_data', '', 'pos_A0201.txt'): ['pos', 'pos_lbl']
    }

    # extract data from relevant directories using the DirMan (directory manager) class
    with DirMan(os.path.join(path_to_data, 'onehot_pulse'), src_tar_dict, extract_data) as data_list:
        neg, neg_lbl, pos, pos_lbl = data_list

    # split datasets into train and test sets
    train_ratio = 0.9
    train_neg, test_neg = train_test_split(neg, train_ratio)
    train_pos, test_pos = train_test_split(pos, train_ratio)
    train_neg_lbl, test_neg_lbl = train_test_split(neg_lbl, train_ratio)
    train_pos_lbl, test_pos_lbl = train_test_split(pos_lbl, train_ratio)

    # balance (+/-) disproportion in samples
    train, test = balance(train_pos, train_neg, test_pos, test_neg)
    train_lbl, test_lbl = balance(train_pos_lbl, train_neg_lbl, test_pos_lbl, test_neg_lbl)

    # convert to Dataset objects, each containing both inputs and labels
    train = PeptideDataset(train, train_lbl)
    test = PeptideDataset(test, test_lbl)

    # convert to DataLoader objects
    train = DataLoader(train, sampler=torch.randperm(len(train)))
    test = DataLoader(test, sampler=torch.randperm(len(test)))

    return train, test


if __name__ == '__main__':
    pass
