from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import scipy.io
import models
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from imblearn.over_sampling import SMOTE
from os.path import join
from tensorflow.keras.metrics import Precision, Recall
from collections import Counter


def preprocess(subject_count=15, trial_count=40, sound_count=30, repeat_count=2):
    """
    Collects all eeg clips form all subjects, orders them, and adds labels of space-bar presses.

    First we establish teh order of the clip presentation - all subjects get the
    same clips in the same order, so needs to be done only once.

    :param subject_count: how many subjects
    :param trial_count: how many trials, per subject
    :param sound_count: how many sounds are presented to each subject
    :param repeat_count: how many times each sound is repeated, per trial
    :return: an array of X-y dataset, per subject: [shape: X:(15, 40, 128, 256), y:(40, 1, 256)]
    """

    subject_list = [
        #'BIJVZD',
        'EFFEUS'
        # ,'GQEVXE', 'HGWLOI', 'HITXMV', 'HNJUPJ', 'NFICHK', 'RHQBHE', 'RMAALZ', 'TQZHZT',
    # 'TUZEZT', 'UOBXJO', 'WWDVDF' 'YMKSWS', 'ZLIDEI']
        ]

    '''establish the order of the clips'''
    # get the order of the sounds
    clip_order = np.zeros((40, 60)).astype(int)  # (trial_count, sound_count * 2)
    for t in range(trial_count):
        with open(join('subjects','stim_order', f'stim{t + 1}.txt')) as f:
            temp = np.array(f.readlines()).astype(int)
            unique = set()

        # increase all second appearances, in the array, by the array length (=30)
        for i in range(temp.size):
            if temp[i] in unique:
                temp[i] += 30
            unique.add(temp[i])

        # write into clip_order line
        clip_order[t, :] = temp

    # decrease 1 to convert to index range [0, 59]
    clip_order -= 1

    for subject in subject_list:

        '''collect clips'''
        # stack eeg's for all sounds per subject : [resulting shape: (30, 80, 128, 256)]
        clip_paths = [join('subjects',f'{subject}', f'{subject}_clip{i + 1}.mat') for i in range(sound_count)]
        clip_eeg = np.stack(
            [loadmat(clip_paths[i])['clipeeg'].transpose(2, 1, 0)
             for i in range(len(clip_paths))])

        '''order the clips'''
        # every slice of the form ordered[:, i, :, :] should be of the same trial: [resulting shape: (60, 40, 128, 256)]
        clip_eeg = np.concatenate([clip_eeg[:, ::2, :, :], clip_eeg[:, 1::2, :, :]], axis=0)

        for i in range(clip_order.shape[0]):
            clip_eeg[:, i, :, :] = clip_eeg[clip_order[i], i, :, :]

    np.save('ordered_clips.npy', clip_eeg)

def create_clicks_array(subject_count=15, trial_count=40, sound_count=30, repeat_count=2):

    subject_list = ['EFFEUS']  # , 'BIJVZD', 'GQEVXE', 'HGWLOI', 'HITXMV', 'HNJUPJ', 'NFICHK', 'RHQBHE', 'RMAALZ', 'TQZHZT',
    # 'TUZEZT', 'UOBXJO', 'WWDVDF' 'YMKSWS', 'ZLIDEI']

    #TODO: create a list for subject in subject_list
    subject = subject_list[0]
    behavioral_path = join('subjects', f'{subject}')

    resptms, results, aborted, trknms, trkorder, dettms, corr = load_behavioral_data(behavioral_path)
    # Create a 2D array with 40 rows and 256*60 columns for sound samples
    #  40 trials
    #  256 time samples per 2 seconds, 120 seconds per trial
    response_labels = np.zeros((trial_count, 256 * 60))
    mat_v7=True
    click_counter = 0
    if(mat_v7==True):
        for trial_idx, trial in enumerate(resptms):  # for every trial

            for click in trial:  # time of spacebar clicked in this trial
                sample_index = round(click * 128)  # for every 2 sec, 256 samples. sampling rate = 128
                response_labels[trial_idx][sample_index] = 1
                click_counter+=1
                print(f"spacebar_clicked at {click}, mapped to {sample_index}")

    else:
        for trial_idx in range(resptms.shape[0]):  # for every trial
            for click in resptms[trial_idx][0]:  # time of spacebar clicked in this trial
                sample_index = round(click[0] * 128)  # for every 2 sec, 256 samples. sampling rate = 128
                print(f"spacebar_clicked at {click[0]}, mapped to {sample_index}")
                response_labels[trial_idx][sample_index] = 1
                click_counter+=1
    response_labels_reordered = np.zeros((trial_count, 256 * 60))
    print(f'Counted {click_counter} clicks in resptms')
    # Reorder the trial dimension
    for trkorder_idx, numerical_idx in enumerate(trkorder):
        response_labels_reordered[int(numerical_idx) - 1] = response_labels[trkorder_idx]
    # Update the response_labels array
    response_labels = response_labels_reordered

    click_counter=0
    for trial in response_labels:
        for click in trial:
            if click==1:
                click_counter+=1

    print(f'Counted {click_counter} clicks in ordered_response_labels')
    np.save('ordered_response_labels.npy', response_labels)

def load_behavioral_data(subject):
    # Load the .mat file
    subject_list = ['EFFEUS']  # , 'BIJVZD', 'GQEVXE', 'HGWLOI', 'HITXMV', 'HNJUPJ', 'NFICHK', 'RHQBHE', 'RMAALZ', 'TQZHZT',
    # 'TUZEZT', 'UOBXJO', 'WWDVDF' 'YMKSWS', 'ZLIDEI']
    mat_file_path = join(f'{subject}', f'ShortClipTwoBack_res_sbj{subject_list[0]}.mat')

    mat_is_v7 = True
    if (mat_is_v7):
        import h5py
        # Open the MATLAB v7.3 file
        with h5py.File(mat_file_path, 'r') as mat_data:
            results = mat_data['results']

            # Helper function to extract data from HDF5 dataset
            def extract_data(item):
                return np.array(item).squeeze()

            def extract_references(ref_array, file):
                data_list = []
                for ref in ref_array:
                    for r in ref:
                        dereferenced = file[r]
                        specific_trial_clicks = np.array(dereferenced)
                        data_list.append(specific_trial_clicks.squeeze(axis = 0))
                return data_list

            aborted = extract_data(results['aborted'])
            trknms = extract_data(results['trknms'])
            trkorder = extract_data(results['trkorder'])
            dettms = extract_data(results['dettms'])
            resptms = extract_data(results['resptms'])
            resptms = extract_references(results['resptms'], mat_data)
            corr = extract_data(results['corr'])

            #corr = results['corr'][()]
            # Print the values
 #           print(f"aborted: {aborted}")
 #           print(f"trknms: {trknms}")
            print(f"trkorder: {trkorder}")
 #           print(f"dettms: {dettms}")
            print(f"resptms: {resptms}")
 #           print(f"corr: {corr}")
    else:
        mat_data = scipy.io.loadmat(mat_file_path)
        # Extract relevant data from the .mat file
        results = mat_data['results']
        aborted = results['aborted'][0,0][0,0]
        trknms = results['trknms'][0,0]
        trkorder = results['trkorder'][0,0][0]
        dettms = results['dettms'][0,0]
        resptms = results['resptms'][0,0]
        corr = results['corr'][0,0]

    # Check if the experiment was aborted
    if aborted:
        raise ValueError("The experiment was aborted. Data might be incomplete.")

    return resptms, results, aborted, trknms, trkorder, dettms, corr



def plot_eeg_channel_amp(eeg_data_i):
    # Dimensions of the clipeeg variable
    time_points = eeg_data_i.shape[0]  # 256 samples (time)
    channels = eeg_data_i.shape[1]  # 128 channels
    presentations = eeg_data_i.shape[2]  # 2 presentations
    # Accessing data for the first presentation
    eeg_first_presentation = eeg_data_i[:, :, 0]
    # Accessing data for the second presentation
    eeg_second_presentation = eeg_data_i[:, :, 1]
    # Example to plot the EEG data from the first channel for the first presentation
    plt.plot(eeg_first_presentation[:, 0])
    plt.title('EEG Data from First Channel - First Presentation')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.show()
    # Saving the plot to a file instead of displaying it
    plt.plot(eeg_first_presentation[:, 0])
    plt.title('EEG Data from First Channel - First Presentation')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.savefig('eeg_first_presentation_channel1.png')
    plt.close()


def segment_eeg_data_with_overlap(eeg, labels, window_duration, overlap, n_classes=2):
    """
    Segments EEG data with overlapping windows and assigns labels.

    Parameters:
    - eeg: np.array, shape [sound, trial, electrode, time]
    - labels: np.array, shape [trial, samples]
    - window_duration: int, duration of each window in samples
    - overlap: float, proportion of overlap between consecutive windows (0 < overlap < 1)
    - n_classes: int, number of classes for classification

    Returns:
    - segmented_data: np.array, shape [num_segments, electrode, window_duration]
    - segmented_labels: np.array, shape [num_segments, n_classes]
    """
    num_sounds, num_trials, num_electrodes, _ = eeg.shape
    stride = int(window_duration * (1 - overlap))
    segmented_data = []
    segmented_labels = []

    for trial in range(num_trials):
        for start in range(0, labels.shape[1] - window_duration + 1, stride):
            end = start + window_duration
            label_segment = labels[trial, start:end]

            # Assign window label: 1 if any label in the window is 1, else 0
            window_label = 1 if np.any(label_segment == 1) else 0
            one_hot_label = np.eye(n_classes)[window_label]

            # Extract EEG data for this window
            eeg_segment = np.concatenate(eeg[:, trial, :, start:end], axis=1)

            segmented_data.append(eeg_segment)
            segmented_labels.append(one_hot_label)

    segmented_data = np.array(segmented_data)
    segmented_labels = np.array(segmented_labels)

    return segmented_data, segmented_labels


def train(X,y, in_chans, in_samples, tcn_kernel):
    global history
    # Define and compile the model with the new window size
    model = models.ATCNet_(
        # Dataset parameters
        n_classes=2, in_chans=in_chans, in_samples=in_samples,
        # Sliding window (SW) parameter
        n_windows=5,
        # Attention (AT) block parameter
        attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
        # Convolutional (CV) block parameters
        eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
        # Temporal convolutional (TC) block parameters
        tcn_depth=2, tcn_kernelSize=tcn_kernel, tcn_filters=32, tcn_dropout=0.3, tcn_activation='elu')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'
                      , Precision(), Recall()
                           ])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verify the new shapes
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

    # Verify the class distribution in the training and validation sets
    print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
    print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32,
                        callbacks=[early_stopping])
    # Evaluate the model
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    # Assuming `model` is your trained model
    plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

    # Assuming your validation data is X_val and y_val
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted classes
    y_true = np.argmax(y_val, axis=1)           # Get the true classes

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred_classes))
def interpolate_eeg(data, original_length, target_length):
    interpolated_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], target_length))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data_data = data[i,j,k].repeat(2)
                data_data_1 = data_data[1:]
                data_data_2 = data_data[:-1]
                a = interpolated_data[i][j][k]
                a[:-1] = (data_data_1 + data_data_2) / 2
                a[-1]  = data[i,j,k][-1] # last element
                interpolated_data[i][j][k] = a
    return interpolated_data


# Interpolation function for clicks data
def interpolate_clicks(clicks, target_length):
    interpolated_clicks = np.zeros((clicks.shape[0], target_length), dtype=int)
    for i in range(clicks.shape[0]):
        clicks_clicks = clicks[i].repeat(2)
        interpolated_clicks[i] = clicks_clicks
    return interpolated_clicks


def interpolate_data():
    global X, y, total_sounds
    eeg = np.load('ordered_clips.npy')
    clicks_array = np.load('ordered_response_labels.npy')

    # Apply interpolation to double the number of time samples
    eeg_interpolated = interpolate_eeg(eeg, original_samples, target_samples)
    clicks_array_interpolated = interpolate_clicks(clicks_array, target_samples * total_sounds)
    eeg = eeg_interpolated
    clicks_array = clicks_array_interpolated
    # Initialize the inputs and labels for the model
    X = []
    y = []
    total_sounds, total_trials, electrode_num, time_samples = eeg.shape
    segment_times = 0
    label_1_count = 0
    label_0_count = 0

    total_count=0
    for trial in range(total_trials):
        # Flatten the eeg for the current trial to be [total_samples, electrode_num]
        eeg_flat = eeg[:, trial, :, :].reshape(total_sounds * time_samples, electrode_num)

        # Get the corresponding clicks array for the trial
        clicks = clicks_array[trial]

        click_counter=0
        click_indices=[]
        for click_idx in range(len(clicks)):
            if clicks[click_idx]==1:
                click_counter+=1
                click_indices.append(click_idx)


        print(f'Counted {click_counter/2} clicks in trial #{trial}')
        print(f'indices of clicks are in: {click_indices}')
        total_count+=click_counter
        # for 60 2-sec sounds sounded in a trial (in total 512*60) step every 2 seconds (512 time samples) and label:
        end = eeg_flat.shape[0]
        step = in_samples
        for start_idx in range(0, end, step):
            end_idx = start_idx + in_samples
            window_data = eeg_flat[start_idx:end_idx].T  # Transpose to get (electrode_num, in_samples)
            X.append(window_data[np.newaxis, :])  # Add new axis for the 1 in shape

            # Determine if there was a click when window ended
            click_window_start = end_idx - 256 # check if there was a click at the second leading up to
            if(end_idx + 128 > end):
                click_window_end = end
            else:
                click_window_end = end_idx+128 # give half a sec grace period
            if np.any(clicks[click_window_start:click_window_end] == 1):
                label = 1
                label_1_count += 1
            else:
                label = 0
                label_0_count += 1
            y.append(label)
            segment_times += 1

    print("Original data has a total of ", total_count/2, " clicks")
    print("segmented ", segment_times, " times")
    print("labeled 1 ", label_1_count, " times")
    print("labeled 0 ", label_0_count, " times")
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    np.save('elaborated_eeg_data.npy', X)
    np.save('elaborated_labels.npy', y)


def SMOTE_data():
    global X, y, X_resampled, y_resampled
    X = np.load('elaborated_eeg_data.npy')
    y = np.load('elaborated_labels.npy')
    # Reshape X to a 2D array for SMOTE
    X = X.reshape((X.shape[0], -1))
    # Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Convert y_resampled back to one-hot encoding
    y_resampled = np.eye(2)[y_resampled]
    # Reshape X_resampled back to its original shape
    X_resampled = X_resampled.reshape((X_resampled.shape[0], 1, 128, 512))
    # Verify the new class distribution
    print("Original dataset shape:", Counter(y))
    print("Resampled dataset shape:", Counter(np.argmax(y_resampled, axis=1)))
    np.save("SMOTEed_eeg_data", X_resampled)
    np.save("SMOTEed_eeg_labels", y_resampled)


if __name__ == '__main__':
    # Constants
    n_classes = 2
    in_chans = 128
    original_samples = 256
    in_samples = 512
    target_samples = 512
    tcn_kernel= 4
    n_windows = 5
    total_sounds = 60
    # Parameter: number of time samples corresponding to 1 second
    param_samples = 128  # Adjust this to change the time before click for labeling

    #preprocess()
    create_clicks_array()
    interpolate_data()
    SMOTE_data()

    X = np.load('SMOTEed_eeg_data.npy')
    y = np.load('SMOTEed_eeg_labels.npy')
    # Parameters
    training = False
    if(training): train(X,y, in_chans=in_chans, in_samples=in_samples, tcn_kernel=tcn_kernel)
