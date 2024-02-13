import numpy as np
from typing import Tuple


def calculate_status(status_window: np.array) -> bool:
    """
    This function converts the status window into a boolean which represents whether a button
    is currently (for this batch) being pressed by the user or not.
    The function will return the state of the last recording.

    TODO: This function may be more complicated in the future

    status_window: The array of recordings of the button state either being pressed or not.
                   If the button is pressed the value is int-max otherwise it is 0.
    return: A boolean representing whether the button is currently (for this batch),
            pressed by the user or not.
    """
    return status_window[-1] == 6.5536e+04


def preprocess(raw, window_interval: float = 0.15, frequency: int = 512, cutoff_index: int = 3_008_700)\
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

    # convert status channel to boolean
    training_set = training_set[:-1], calculate_status(training_set[-1])
    test_set = test_set[:-1], calculate_status(test_set[-1])
    validation_set = validation_set[:-1], calculate_status(validation_set[-1])

    return training_set, test_set, validation_set
