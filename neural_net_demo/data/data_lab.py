import mne
import numpy as np
import pickle

'''
This function let's you view the data in an interactive window,
the way it works is you change the variable stimulus_channel_list to contain 
all strings of names of channels you want to view.

The channel names are:
'A1', 'A2', ..., 'A32',
'B1', 'B2', ..., 'B32',
'C1', 'C2', ..., 'C32',
'D1', 'D2', ..., 'D32',
'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8',
'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status'

Resp - short for respiratory system(breathing).
EXG - other signals, may contain blink monitoring channels [not sure].
GSR - Galvanic Skin Response, which measures the electrical conductance of the skin, 
      and is influenced by sweat gland activity and is often used as an indicator of 
      emotional arousal or stress [not sure].
ERG - Electrocardiogram (monitoring of heart activity) [not sure].
Plet is short for Plethysmography (changes in blood volume).
Temp - temperature.
Status - this channel records key presses.
'''


def view_data():
    raw = mne.io.read_raw_fif('raw_complete.fif', preload=True)
    with open('pickled_data/zurs_data/zur_BIGVZD_numpy.pickle', 'wb') as numpy_file:
        pickle.dump(raw[:][0], numpy_file)

    with open('pickled_data/zurs_data/zur_BIGVZD_numpy.pickle', 'rb') as numpy_file:
        un_pickled_data = pickle.load(numpy_file)

    print("")
    # Select the stimulus channel(s) based on channel names or indices
    # Replace with the actual name or index of your stimulus channel
    stimulus_channel_list = ['A1']

    # Plot the data from the selected channel(s)
    raw.pick(stimulus_channel_list).plot(block=True)


if __name__ == '__main__':
    view_data()
