import os.path
import pickle
from scipy.io import loadmat
import numpy as np

path_to_pickle = ['data', 'pickled_data']


def store_pickle(obj, file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'wb') as f:
        pickle.dump(obj, f)


def get_pickle(file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def matlab_to_numpy(path_to_mat_file: str, mat_format, force: bool = False):
    """
    This function converts a given .mat file (via the given path), to a numpy array.

    The .mat file is converted according to the given format (providing flexibility),
    according to the following rules(rules for mat_format parameter):

    mat_format is a dictionary containing 2 entries:
        - 'data set key path' - a list of keys to get to the data set.
        - 'solution set key path' - a list of keys ot get to the solution set.
    The way the key path is used (in both cases) is as exemplified:
    set = mat_object[key_0][key_1][key_2]...

    The resulting numpy array contains the data and the solution set in the following format:

        - dataset - matrix(channel_count, Fs * T): this matrix contains the EEG recordings, there are channel_count rows
        (number of channels in the EEG device), and Fs * T (sampling frequency times the total duration of the
        session).

        - solution set - vector(Fs * T): This vector contains a marker (an integer representing a keypress on
        the keyboard) for each sample throughout the session.

    The function pickles the data and stores it in a (pre-existing) directory "kayas_data"

    The Force Flag:
    This function is meant to run only once, and if a pickle file with the name "dataset.pickle"
    is present in the path "neural_net_demo/data/pickled_data/kayas_data", the function will do nothing.
    Unless the force flag is set to true, this means that a new file needs to be loaded and pickled.

    :param: path_to_mat_file: path to the matlab file
    :param: mat_format: the format of the given matlab file
    :param: force: force flag
    """
    # check if file already exists
    kayas_path = path_to_pickle + ['kayas_data', 'dataset.pickle']

    if os.path.exists(os.path.join(*kayas_path)) and not force:
        return

    # load data from matlab file
    mat_object = loadmat(path_to_mat_file)

    # get arrays of dataset and solution set from key paths
    data_set = mat_object
    for key in mat_format['data set key path']:
        data_set = data_set[key]

    solution_set = mat_object
    for key in mat_format['solution set key path']:
        solution_set = solution_set[key]

    # merge data and solution set to one numpy array
    data_set = np.concatenate((data_set, solution_set), axis=1)

    # save to pickle files
    store_pickle(data_set.T, kayas_path)


def preprocess_to_pickle(raw: np.ndarray, raw_meta_data=None,
                         window_interval: float = 0.15, frequency: int = 1000,
                         train_test_val_ratio=None):
    """
    This function applies a series of transformations to the raw data, from the .fif file (retrieved with mne),
    to appropriate it for the network training.

    raw: The raw mne object - retrieved using mne.io.read_raw_fif(...)
    window_interval: the time duration of one batch of data - this batch will be propagated through
                     the network - one propagation per batch.
                     Additionally, the CNN kernel will be the same size as this interval (the number of samples
                     in this interval).
                     (at least at the first layer).
    frequency:the frequency of the samples [Hz]

    todo: push all default parameters into metadata.
    """

    # default values
    # set default values of train_test_val_ratio
    if train_test_val_ratio is None:
        train_test_val_ratio = [0.6, 0.2, 0.2]

    # default value of meta data is 22 - for 22 channels in kaya's dataset
    if raw_meta_data is None:
        raw_meta_data = [22]
    channel_count = raw_meta_data[0]

    # set parameters
    window_size = round(frequency * window_interval)
    train_test_val_lengths = train_test_val_ratio * raw.shape[1]

    # build batched training set tensor and match with solution set
    data_set = np.lib.stride_tricks.sliding_window_view(raw, (channel_count, window_size))[0]
    store_pickle(data_set, path_to_pickle + ['kayas_data', 'batched.pickle'])

    # apply train-test-validation ratios to data
    data_set = np.split(data_set, train_test_val_lengths)

    # get solution set
    train_set, test_set, validate_set = data_set[:, :, :-1]
    train_solution, test_solution, validate_solution = data_set[:, :, -1]

    # save to pickle

    store_pickle(train_set, path_to_pickle + ['kayas_data', 'train_set.pickle'])
    store_pickle(test_set, path_to_pickle + ['kayas_data', 'test_set.pickle'])
    store_pickle(validate_set, path_to_pickle + ['kayas_data', 'validate_set.pickle'])

    store_pickle(train_solution, path_to_pickle + ['kayas_data', 'train_solution.pickle'])
    store_pickle(test_solution, path_to_pickle + ['kayas_data', 'test_solution.pickle'])
    store_pickle(validate_solution, path_to_pickle + ['kayas_data', 'validate_solution.pickle'])


if __name__ == '__main__':

    '''Preprocessing for Zur's old Data (deprecated)'''
    # # get pickle of raw in mne object form
    # raw_numpy = get_pickle('data/pickled_data/zurs_data/zur_BIGVZD_numpy.pickle')
    # # mne_raw = mne.io.read_raw_fif('./neural_net_demo/data/raw_complete.fif')
    # x, y, z = preprocess(raw_numpy)
    # print('done')

    '''Preprocessing for Kaya's hip Data'''
    format_dictionary = {
                        'data set key path': ['o', 0, 0, 5],
                        'solution set key path': ['o', 0, 0, 4]
    }
    matlab_to_numpy('matlab_files/5F-SubjectH-160804-5St-SGLHand-HFREQ.mat', format_dictionary)
    dataset = get_pickle(path_to_pickle + ['kayas_data', 'dataset.pickle'])
    preprocess_to_pickle(dataset)
