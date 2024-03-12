import os.path

import mne.io
import numpy as np
from typing import Tuple
import pickle

path_to_pickle = ['.', 'neural_net_demo', 'prerpocessed_no_solution', '']


def store_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def get_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def calculate_solution(status_window: np.ndarray) -> bool:
    """

    """


def preprocess(raw: np.ndarray, window_interval: float = 0.15, frequency: int = 512, cutoff_index: int = 3_008_700)\
        -> Tuple[Tuple[np.array, bool], Tuple[np.array, bool], Tuple[np.array, bool]]:
    """
    This function applies a series of transformations to the raw data, from the .fif file (retrieved with mne),
    to appropriate it for the network training.

    raw: The raw mne object - retrieved using mne.io.read_raw_fif(...)
    window_interval: the time duration of one batch of data - this batch will be propagated through
                     the network - one propagation per batch.
                     Additionally, the CNN kernel will be the same size as this interval (the number of samples in this interval).
                     (at least at the first layer).
    frequency:
    """

    # ignore everything after index 3_008_700 - unclear data behavior
    window_size = round(frequency * window_interval)
    data_cutoff_index = round(cutoff_index / window_size) * window_size
    train_test_validate_ratios = 0.6 * data_cutoff_index, 0.8 * data_cutoff_index

    # build training set tensor and match with solution set
    data_set = np.lib.stride_tricks.sliding_window_view(raw[:data_cutoff_index][0],
                                                        (len(raw.ch_names), window_size))

    shuffled_indices = np.random.permutation(data_set.shape[1])

    data_set[0] = data_set[0][shuffled_indices]

    # divide into training set, test set, and validation set, with the following ratios: 60%-20%-20%
    training_set, test_set, validation_set = np.split(data_set, train_test_validate_ratios)
    
    # label all training set points with a no-press label
    training_solution = np.zeros(training_set.shape)
    test_solution = np.zeros(test_set.shape)
    validation_solution = np.zeros(validation_set.shape)
    
    # fixme: for now using solution sets with all 'no-press' label until more data is found
    # divide solution set into training solution, test_solution, and validation_solution
    # training_solution, test_solution, validation_solution = np.split(solution_set, train_test_validate_ratios)

    # convert status channel to boolean
    training_set = training_set[:-1], calculate_solution(training_solution)
    test_set = test_set[:-1], calculate_solution(test_solution)
    validation_set = validation_set[:-1], calculate_solution(validation_solution)
    
    # save to pickle
    store_pickle(training_set, os.path.join(path_to_pickle + ['training_set.pickle']))
    store_pickle(training_set, os.path.join(path_to_pickle + ['test_set.pickle']))
    store_pickle(training_set, os.path.join(path_to_pickle + ['validation_set.pickle']))

    return training_set, test_set, validation_set


if __name__ == '__main__':
    # get pickle of raw in mne object form
    mne_raw = mne.io.read_raw_fif('./neural_net_demo/data/raw_complete.fif')
    x, y, z = preprocess(mne_raw)
    print('done')
