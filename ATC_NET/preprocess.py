""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle




############### ARAN CODE
import torch
# import mne
# from mne.preprocessing import ICA
# import numpy as np
# import pickle
# from os.path import exists


# def pre_ica(X: np.ndarray):
#     sfreq = 256
#     channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
#                      'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
#                      'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
#                      'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',
#                      'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8',
#                      'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
#     channel_types = ['eeg'] * 32 + ['eog'] * 15

    ## create mne info
    # info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    # montage = mne.channels.make_standard_montage('standard_1020')
    # info.set_montage(montage)
    #
    ## convert data to mne
    # raw = mne.io.RawArray(X, info)
    # raw.info['bads'] = ['T7']

    ## apply band-pass filter to 1-40Hz
    # raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40)
    #
    # return raw_filtered


# def ica(raw_filtered: np.ndarray, new=False):
#
#     pickle_path = 'ica.pickle'
#
#     # use saved ICA object, if one exists
#     if exists(pickle_path) and not new:
#         with open(pickle_path, 'rb') as f:
#             ica = pickle.load(f)
#             return ica.get_sources(raw_filtered).get_data()
#
#     if new:
#         print('ICA object file doesn\'t exist, must create new one')
#
#     # apply ICA
#     ica = ICA(n_components=22, random_state=42, max_iter=800)
#     ica.fit(raw_filtered)
#
#     # ica.plot_components()
#
#     # Find the ICA components that match the EOG pattern
#     eog_indices, eog_scores = ica.find_bads_eog(raw_filtered)
#     # ica.plot_scores(eog_scores)
#
#     # Inspect the identified components
#     # ica.plot_properties(raw_filtered, picks=eog_indices)
#
#     # If you want to remove the identified EOG artifacts
#     ica.exclude = eog_indices
#     ica.apply(raw_filtered)
#
#     # Plot the cleaned data
#     # raw_filtered.plot()
#
#     # save ICA object
#     with open('ica.pickle', 'wb') as f:
#         pickle.dump(ica, f)
#
#     return ica.get_sources(raw_filtered).get_data()

############### ARAN CODE


# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%
def load_data_LOSO (data_path, subject, dataset): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model.
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)
        # elif (dataset == 'HGD'):
        #     X1, y1 = load_HGD_data(path, sub+1, True)
        #     X2, y2 = load_HGD_data(path, sub+1, False)
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


def load_data_LOSO(data_path, subject, dataset):
    """ Loading and Dividing of the data set based on the
    'Leave One Subject Out' (LOSO) evaluation approach.
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that
    separate subjects (not visible in the training data) are usedto evaluate
    the model.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """

    X_train, y_train = [], []
    for sub in range(0, 9):
        path = data_path + 's' + str(sub + 1) + '/'

        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub + 1, True)
            X2, y2 = load_BCI2a_data(path, sub + 1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _ = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _ = load_CS2R_data_v2(path, sub, False)
        # elif (dataset == 'HGD'):
        #     X1, y1 = load_HGD_data(path, sub+1, True)
        #     X2, y2 = load_HGD_data(path, sub+1, False)

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if (sub == subject):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test

#%%
def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    In this approach, we used the same training and testing datas the original
    competition, i.e., 288 x 9 trials in session 1 for training,
    and 288 x 9 trials in session 2 for testing.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts
    """

    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250

    # Define MI trial window
    fs = 250  # sampling rate
    t1 = int(1.5 * fs)  # start time_point
    t2 = int(6 * fs)  # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0

    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']

    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :22])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)
    # data_return[288][22][1125]
    # class_return[288]
    return data_return, class_return

def load_ELIS_data(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    In this approach, we use 512 x 32 trials for training,
    and 512 x 8 trials in for testing.
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset ELIS is available on
            # insert prof Zur's data repo url
        subject: int
            number of subject in [1, .. ,9]
            converted to that subjects code (e.g.: 1 -> 'BIJVZD')
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts
    """


    if training:
        X = np.load('Zurs_Dataset/subjects/BIJVZD/SMOTEed_eeg_data.npy')
        y = np.load('Zurs_Dataset/subjects/BIJVZD/SMOTEed_eeg_labels.npy')
    else:
        X = np.load('Zurs_Dataset/subjects/EFFEUS/SMOTEed_eeg_data.npy')
        y = np.load('Zurs_Dataset/subjects/EFFEUS/SMOTEed_eeg_labels.npy')

    return X, y




#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


#%%
def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True):
    
    # Load and split the dataset into training and testing 
    if LOSO:
        """ Loading and Dividing of the dataset based on the 
        'Leave One Subject Out' (LOSO) evaluation approach. """ 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        """ Loading and Dividing of the data set based on the subject-specific 
        (subject-dependent) approach.
        In this approach, we used the same training and testing data as the original
        competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1 
        for training, and 288 x 9 trials in session 2 for testing.  
        """

        if (dataset == 'ELIS'):
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_ELIS_data(path, subject+1, True)
            X_test, y_test = load_ELIS_data(path, subject+1, False)
        elif (dataset == 'BCI2a'):
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    #  BCI2a X.shape = (288, 22, 1125)
    #  BCI2a y.shape = (288,)

    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)

    # Prepare training data
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)

    # Prepare testing data
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)

    if(dataset=='ELIS'):
        y_train_onehot = y_train
        y_test_onehot = y_test
    else:
        y_train_onehot = to_categorical(y_train)
        y_test_onehot = to_categorical(y_test)
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

