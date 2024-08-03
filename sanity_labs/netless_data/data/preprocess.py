import os.path

from torch.nn.functional import one_hot
from torch import tensor, int64
from scipy.io import loadmat
import numpy as np

from utils import store_pickle, get_pickle


# todo: convert Preprocessor to dataclass

class Preprocessor:
    path_to_pickle = ['..', 'data', 'pickle']

    def __init__(self, format_dictionary, mat_path, batch_parameters, train_test_val_ratio):
        self.format_dictionary = format_dictionary
        self.mat_path = mat_path
        self.window_interval, self.frequency = batch_parameters
        self.train_test_val_ratio = train_test_val_ratio
        self.raw_length = 0
        self.dataset_path = None
        self.shuffle = None

        '''parameter intermediate calculations'''
        # convert ratio list to a cumulative sum of the fractions
        for i in range(1, len(train_test_val_ratio)):
            self.train_test_val_ratio[i] += self.train_test_val_ratio[i - 1]

        self.window_size = round(self.frequency * self.window_interval)

        # convert matlab file to numpy array
        self.matlab_to_numpy()

        self.get_shuffle()

    def matlab_to_numpy(self, force: bool = False):
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

            - dataset - matrix(channel_count, Fs * T): this matrix contains the EEG recordings,
            there are channel_count rows (number of channels in the EEG device),
            and Fs * T (sampling frequency times the total duration of the session).

            - solution set - vector(Fs * T): This vector contains a marker (an integer representing a keypress on
            the keyboard) for each sample throughout the session.

        The function pickles the data and stores it in a (pre-existing) directory "kayas_data"

        The Force Flag:
        This function is meant to run only once, and if a pickle file with the name "dataset.pickle"
        is present in the path "sanity_labs/data/pickle/kayas_data", the function will do nothing.
        Unless the force flag is set to true, this means that a new file needs to be loaded and pickled.

        :param: path_to_mat_file: path to the matlab file
        :param: mat_format: the format of the given matlab file
        :param: force: force flag
        """

        '''preamble'''
        # parameters
        kayas_path = Preprocessor.path_to_pickle + ['kayas_data', 'dataset.pickle']

        # check if file already exists
        if os.path.exists(os.path.join(*kayas_path)) and not force:
            self.dataset_path = kayas_path
            self.raw_length = get_pickle(kayas_path)[0].shape[1]
            return

        # todo: if the directory doesn't exist -> create the directory

        '''extract data form matlab file'''
        # load data from matlab file
        mat_object = loadmat(self.mat_path)

        # get arrays of dataset and solution set from key paths
        data_set = mat_object
        for key in self.format_dictionary['data set key path']:
            data_set = data_set[key]

        solution_set = mat_object
        for key in self.format_dictionary['solution set key path']:
            solution_set = solution_set[key]

        # store length of raw data
        self.raw_length = data_set.shape[1]

        '''encode solution set into a one-hot encoding'''
        # get all possible solutions and map them to ordinal numbers [0 - max]
        for ordinal, option in enumerate(np.unique(solution_set)):
            solution_set[solution_set == option] = ordinal

        # make solution set into one-hot matrix
        solution_set = one_hot(tensor(solution_set, dtype=int64).flatten()).float()

        # pack data and solution set
        data_set = data_set.T, solution_set

        # save to pickle file
        store_pickle(data_set, kayas_path)

        self.dataset_path = kayas_path

    def get_shuffle(self):
        """
        This function takes a numpy nd-array of the data, and generates shuffled index list.

        The Shuffle (a.k.a the returned value):
        The shuffle (shuffled index list), is a triplet of index lists (for the 3 sets:
        train, test, and validate), each containing a random series of indices from the original numpy array.
        This shuffle is to be used to train the netless_data, without generating any copies of the data.

        How to Train the Network:
        To train the netless_data, the train_shuffle(the first element of the shuffle triplet)
        is iterated over, and for each iteration the batch of samples starting
        at the current index of the train_shuffle, is fed into the netless_data.

        :return: shuffle - the triples of shuffled index lists.
        """

        # generate random shuffle
        rng = np.random.default_rng()
        shuffle = rng.permutation(self.raw_length - self.window_size + 1)

        # get ratio indices
        train_test_val_indices = np.multiply(self.train_test_val_ratio,
                                                  shuffle.shape[0]).astype(int)[:-1]

        # split shuffle into three sets (train/test/validate)
        self.shuffle = np.split(shuffle, train_test_val_indices)

    def get_shuffle_set(self, set_name):
        set_name_key = {
            'TRAIN': 0,
            'TEST': 1,
            'VAL': 2
        }

        return self.shuffle[set_name_key[set_name]]

    def get_dataset(self):
        return get_pickle(self.dataset_path)


if __name__ == '__main__':
    pass
