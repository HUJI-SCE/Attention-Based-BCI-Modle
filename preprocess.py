import numpy as np
from typing import Tuple


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
    data_set = np.lib.stride_tricks.sliding_window_view(raw[:][:data_cutoff_index],
                                                        (window_size, len(raw.ch_names)))
    # shuffle training set
    np.random.shuffle(data_set)

    # divide into training set, test set, and validation set, with the following ratios: 60%-20%-20%
    training_set, test_set, validation_set = np.split(data_set, train_test_validate_ratios)

    # divide solution set into training solution, test_solution, and validation_solution
    training_solution, test_solution, validation_solution = np.split(solution_set, train_test_validate_ratios)

    # rmagal: temporary commented because we still need to get solution set
    # convert status channel to boolean
    #training_set = training_set[:-1], calculate_solution(training_solution)
    #test_set = test_set[:-1], calculate_solution(test_solution)
    #validation_set = validation_set[:-1], calculate_solution(validation_solution)

    return training_set, test_set, validation_set
