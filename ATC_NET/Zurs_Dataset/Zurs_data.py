import os
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.svm import SVC

import models
from scipy.io import loadmat
from imblearn.over_sampling import SMOTE
from os.path import join

import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
    precision_score, recall_score, roc_curve, auc
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall
from keras.utils import plot_model


def preprocess(subject, trial_count=40, sound_count=30):
    """
        Preprocesses EEG data by collecting, ordering, and saving EEG clips for a given subject.

        This function reads EEG clips for a specified subject, orders them according to the predefined order of sound stimuli,
        and saves the ordered clips as a NumPy array. The clip order is determined by a text file that specifies the order
        of sound stimuli for each trial. This ordering process ensures that the EEG data from all subjects is aligned
        according to the same presentation order.

        Parameters:
        ----------
        subject : str
            Identifier of the subject whose data is being processed.
        trial_count : int, optional, default=40
            Number of trials per subject.
        sound_count : int, optional, default=30
            Number of sounds presented to each subject.

        Returns:
        -------
        None
    """

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

    np.save(f'ordered_clips_{subject}.npy', clip_eeg)


def detect_mat_version(file_path):
    """
    Detects the version of a MATLAB .mat file.

    This function reads the header of a .mat file to determine its version. MATLAB .mat files can be of different versions,
    including MATLAB 5.0 (traditional MAT-files) and HDF5 (introduced in MATLAB v7.3). The function returns the version number
    or raises an error if the version is unknown.

    Parameters:
    ----------
    file_path : str
        Path to the .mat file.

    Returns:
    -------
    int
        Version number of the .mat file (5 or 7).

    Raises:
    ------
    ValueError
        If the .mat file version is unknown.
    """
    with open(file_path, 'rb') as f:
        header = f.read(128).decode('latin1')
        if 'MATLAB 5.0 MAT-file' in header:
            return 5
        elif 'HDF5' in header:
            return 7
        else:
            raise ValueError("Unknown .mat file version")


def create_clicks_array(subject, trial_count=40):
    """
        Creates a 2D array representing spacebar clicks during trials for a given subject.

        This function processes behavioral data from a .mat file and generates an array indicating
        the timing of spacebar clicks during trials. The array has dimensions of (trials, time samples),
        where each entry is 1 if a spacebar was pressed at that time and 0 otherwise.
        The resulting array is saved as a .npy file.

        Parameters:
        ----------
        subject : str
            Identifier of the subject whose data is being processed.
        trial_count : int, optional, default=40
            Number of trials per subject.

        Returns:
        -------
        None
    """

    behavioral_path = join('subjects', f'{subject}', f'ShortClipTwoBack_res_sbj{subject}.mat')

    resptms, results, aborted, trknms, trkorder, dettms, corr = load_behavioral_data(subject, behavioral_path)
    # Create a 2D array with 40 rows and 256*60 columns for sound samples
    #  40 trials
    #  256 time samples per 2 seconds, 120 seconds per trial
    response_labels = np.zeros((trial_count, 256 * 60))

    mat = detect_mat_version(behavioral_path)
    click_counter = 0
    if mat==7:
        for trial_idx, trial in enumerate(resptms):
            for click in trial: # time of spacebar clicked in this trial
                sample_index = round(click * 128)  # for every 2 sec, 256 samples. sampling rate = 128
                response_labels[trial_idx][sample_index] = 1
                click_counter+=1
    else:
        for trial_idx in range(resptms.shape[0]):
            for click in resptms[trial_idx][0]:
                sample_index = round(click[0] * 128)
                response_labels[trial_idx][sample_index] = 1
                click_counter+=1

    response_labels_reordered = np.zeros((trial_count, 256 * 60))
    print(f'Counted {click_counter} clicks in resptms')

    # Reorder the trial dimension
    for trkorder_idx, numerical_idx in enumerate(trkorder):
        response_labels_reordered[int(numerical_idx) - 1] = response_labels[trkorder_idx]
    # Update the response_labels array
    response_labels = response_labels_reordered
    np.save(f'ordered_response_labels_{subject}.npy', response_labels)


def load_behavioral_data(mat_file_path):
    """
        Loads behavioral data from a .mat file for a given subjects mat file path.

        This function loads and processes behavioral data from a MATLAB .mat file. The data includes information
        about trials, such as aborted trials, track names, order of tracks, detection times, response times, and correctness.
        The function handles both MATLAB v5 and v7.3 formats.

        Parameters:
        ----------
        mat_file_path : str
            Path to the .mat file containing the behavioral data.

        Returns:
        -------
        tuple
            A tuple containing the following elements:
            - resptms: Response times.
            - results: Results data structure.
            - aborted: Aborted trials indicator.
            - trknms: Track names.
            - trkorder: Order of tracks.
            - dettms: Detection times.
            - corr: Correctness of responses.

        Raises:
        ------
        ValueError
            If the experiment was aborted.
    """

    mat = detect_mat_version(mat_file_path)
    if mat==7:
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
                        dereferenced = dereferenced[:]
                        specific_trial_clicks = np.array(dereferenced)
                        if(np.all(dereferenced==0)):
                            continue
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
    """
        Plots the amplitude of an EEG channel over time for a single presentation.

        This function generates and saves a plot of EEG data for the first channel
        of the first presentation in the provided EEG data array. The plot displays
        the amplitude of the EEG signal over time points.

        Parameters:
        ----------
        eeg_data_i : np.array
            EEG data array with shape (time_points, channels, presentations).

        Returns:
        -------
        None
    """
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

    This function segments EEG data using a sliding window approach with a specified overlap.
    Each segment is assigned a label based on the presence of specific events (e.g., spacebar clicks)
    within the window. The function returns the segmented EEG data and corresponding one-hot encoded labels.

    Parameters:
    ----------
    eeg : np.array
        The EEG data array with shape (sound, trial, electrode, time).
    labels : np.array
        The labels array with shape (trial, samples).
    window_duration : int
        Duration of each window in samples.
    overlap : float
        Proportion of overlap between consecutive windows (0 < overlap < 1).
    n_classes : int, optional, default=2
        Number of classes for classification.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - segmented_data: np.array, shape (num_segments, electrode, window_duration)
        - segmented_labels: np.array, shape (num_segments, n_classes)
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


def custom_hinge_loss(y_true, y_pred):
    """
    Calculate the hinge loss for a Support Vector Machine (SVM).

    Hinge loss is used in SVM to penalize misclassifications and
    samples that fall within the margin boundaries.

    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True labels for each sample. Labels must be -1 or 1.

    y_pred : array-like, shape (n_samples,)
        Predicted decision values from the SVM (w * x + b).

    Returns:
    --------
    loss : float
        The average hinge loss across all samples.
    --------
    """
    # Calculate hinge loss for each sample
    loss_per_sample = np.maximum(0, 1 - y_true * y_pred)

    # Return the mean hinge loss
    return np.mean(loss_per_sample)


def test(model, Z, w, subject):
    """
        Evaluates a trained model on test data and saves the metrics.

        This function evaluates a trained model on a given test dataset,
        prints out metrics such as loss, accuracy, precision, and recall,
        and saves these metrics along with the confusion matrix.
        Predicted labels and true labels are also saved for further analysis.

        Parameters:
        ----------
        model : keras.Model
            The trained model to be evaluated.
        Z : np.array
            The test dataset features.
        w : np.array
            The test dataset labels.
        subject : str
            Identifier of the subject whose data is being tested.

        Returns:
        -------
        None
    """

    # Verify the class distribution in the testing set
    print("Testing set class distribution:", Counter(np.argmax(w, axis=1)))
    if isinstance(model, SVC):
        # Flattening the input data for SVM as it requires 2D input (samples, features)
        # Convert w from one-hot encoding to class labels
        w = np.argmax(w, axis=1)  # w.shape will be (4574,)
        # Squeeze Z to remove the singleton dimension
        Z = np.squeeze(Z, axis=1)  # Z.shape will be (4574, 128, 512)
        # Calculate the mean across the last two dimensions (128, 512)
        Z = np.mean(Z, axis=2)  # Z.shape will be (4574,128)
        # Now Z and w are ready to be used for SVM training
        y_pred_classes = model.predict(Z)
        y_true_classes = w
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes)
        recall = recall_score(y_true_classes, y_pred_classes)
        loss = custom_hinge_loss(y_true_classes,y_pred_classes)
    else:
        # Predict the labels for the testing set
        y_pred = model.predict(Z)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(w, axis=1)
        # Evaluate the model on the testing data
        loss, accuracy, precision, recall = model.evaluate(Z, w)

    # Assumes binary classification with class 1 probabilities
    if isinstance(model, SVC):
        y_probs = y_pred_classes
    else:
        y_probs = y_pred[:, 1]

    # Call the separate ROC plotting function
    plot_and_save_roc_curve(y_true_classes, y_probs,file_name=os.path.join(results_folder, f'testing_stage_{subject}_roc_curve'))

    print(f'Testing loss: {loss}, Testing accuracy: {accuracy}, Testing precision: {precision}, Testing recall: {recall}')
    save_metrics(accuracy, loss, precision, recall, subject, y_pred_classes, y_true_classes,stage = "Testing")


def draw_learning_curves(history, subjects):
    """
       Plots and saves learning curves for training and validation accuracy/loss.

       This function plots the learning curves for training and validation accuracy and loss
       over epochs, using the history of a Keras model's training process. The plots are saved
       as PNG files.

       Parameters:
       ----------
       history : keras.callbacks.History
           The history object returned by the model's fit method.
       subjects : list
           List of subjects used in training.

       Returns:
       -------
       None
   """

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    subjects_str = ', '.join(subjects)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(os.path.join(results_folder, 'accuracy_validation.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(join('Model loss'))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(os.path.join(results_folder, 'loss_validation.png'))
    print('saved acc, loss graphs for subject training validation')
    plt.close()


def save_metrics(accuracy, loss, precision, recall, subject, y_pred_classes, y_true_classes, stage):
    """
    Saves the evaluation metrics and confusion matrix to a file.
    This function saves the accuracy, loss, precision, and recall metrics,
    along with the confusion matrix, to a text file. The file is named according
    to the subject and the stage of evaluation (e.g., training, testing).

    Parameters:
    ----------
    accuracy : float
        Accuracy of the model.
    loss : float
        Loss value of the model.
    precision : float
        Precision of the model.
    recall : float
        Recall of the model.
    subject : str
        Identifier of the subject.
    y_pred_classes : np.array
        Predicted class labels.
    y_true_classes : np.array
        True class labels.
    stage : str
        The stage of evaluation (e.g., 'training', 'testing').

    Returns:
    -------
    None
    """

    cm = save_confusion_mat(subject, y_pred_classes, y_true_classes)
    # Save metrics and confusion matrix
    with open(os.path.join(results_folder, f'{stage}_results_{subject}.txt'), 'w') as f:
        f.write(
            f'{stage} loss: {loss}\n{stage} accuracy: {accuracy}\n{stage} precision: {precision}\n{stage} recall: {recall}\n')
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true_classes, y_pred_classes))


def save_confusion_mat(subject, y_pred_classes, y_true_classes):
    """
    Generates and saves a confusion matrix plot.

    This function generates a confusion matrix based on the predicted and true class labels,
    displays it, and saves the plot as a PNG file. The filename is based on the subject identifier.

    Parameters:
    ----------
    subject : str
        Identifier of the subject.
    y_pred_classes : np.array
        Predicted class labels.
    y_true_classes : np.array
        True class labels.

    Returns:
    -------
    np.array
        The confusion matrix.
    """

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if subject:
        plt.savefig(os.path.join(results_folder, f'confusion_matrix_{subject}.png'))
    else:
        plt.savefig(os.path.join(results_folder, f'train_validation_confusion_matrix.png'))
    plt.close()
    return cm


def load_trained_model(model_path):
    """
    Loads a pre-trained Keras model from a file.

    This function loads and returns a Keras model that has been saved to the specified file path.

    Parameters:
    ----------
    model_path : str
        Path to the saved model file.

    Returns:
    -------
    keras.Model
        The loaded Keras model.
    """
    return load_model(model_path)


def train_atcnet(X, y, dataset_conf):
    """
    Train the ATCNet model on the provided dataset.

    Parameters:
    ----------
    X : np.array
        The input EEG data.
    y : np.array
        The labels for the EEG data.
    dataset_conf : dict
        Dictionary containing dataset configuration variables.

    Returns:
    -------
    model : keras.Model
        The trained ATCNet model.
    """
    model = models.ATCNet_(
        n_classes=2, in_chans=dataset_conf['in_chans'], in_samples=dataset_conf['in_samples'],
        n_windows=5, attention='mha',
        eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
        tcn_depth=2, tcn_kernelSize=dataset_conf['tcn_kernel'], tcn_filters=32, tcn_dropout=0.3, tcn_activation='elu'
    )

    return compile_and_train_model(model, X, y, dataset_conf)


def train_shallowconvnet(X, y, dataset_conf):
    """
    Train the ShallowConvNet model on the provided dataset.

    Parameters:
    ----------
    X : np.array
        The input EEG data.
    y : np.array
        The labels for the EEG data.
    dataset_conf : dict
        Dictionary containing dataset configuration variables.

    Returns:
    -------
    model : keras.Model
        The trained ShallowConvNet model.
    """
    model = models.ShallowConvNet(nb_classes=2, Chans=dataset_conf['in_chans'], Samples=dataset_conf['in_samples'])
    return compile_and_train_model(model, X, y, dataset_conf)


def plot_and_save_roc_curve(y_true, y_probs, file_name='roc_curve'):
    """
    Plots the ROC curve, checks various thresholds, and saves it to a file.

    Parameters:
    ----------
    y_true : np.array
        True binary labels.
    y_probs : np.array
        Predicted probabilities for the positive class.
    file_name : str
        The name of the file where the plot will be saved.

    Returns:
    -------
    None
    """
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Compute Youden's Index and Distance to Top-Left Corner
    youden_index = tpr - fpr
    optimal_idx_youden = np.argmax(youden_index)
    optimal_threshold_youden = thresholds[optimal_idx_youden]

    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_idx_distance = np.argmin(distances)
    optimal_threshold_distance = thresholds[optimal_idx_distance]


    # Write metrics to a text file
    with open(os.path.join(f"{file_name}.txt"), 'w') as file:
        file.write(f"Optimal threshold by Youden's Index: {optimal_threshold_youden:.2f}\n")
        file.write(f"Optimal threshold by distance to top-left: {optimal_threshold_distance:.2f}\n")

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Mark the optimal thresholds on the ROC curve
    plt.scatter(fpr[optimal_idx_youden], tpr[optimal_idx_youden], color='green', label=f'Optimal Youden (thresh = {optimal_threshold_youden:.2f})')
    plt.scatter(fpr[optimal_idx_distance], tpr[optimal_idx_distance], color='red', label=f'Optimal Dist (thresh = {optimal_threshold_distance:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot to a file
    plt.savefig(os.path.join(f"{file_name}.png"))
    plt.close()


def compile_and_train_model(model, X, y, dataset_conf):
    """
    Compile and train a Keras model on the provided dataset.

    Parameters:
    ----------
    model : keras.Model
        The Keras model to be compiled and trained.
    X : np.array
        The input EEG data.
    y : np.array
        The labels for the EEG data.
    dataset_conf : dict
        Dictionary containing dataset configuration variables.

    Returns:
    -------
    model : keras.Model
        The trained Keras model.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=dataset_conf['train_to_val_percentage'],
                                                      random_state=42)

    print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
    print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=dataset_conf['epochs'],
                        batch_size=dataset_conf['batch_size'], callbacks=[early_stopping])

    loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
    print(f'Validation loss: {loss}, Validation accuracy: {accuracy}')

    y_pred = model.predict(X_val)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Call the separate ROC plotting function
    y_probs = y_pred[:, 1]  # Assumes binary classification with class 1 probabilities
    plot_and_save_roc_curve(y_true, y_probs,
                            file_name=os.path.join(results_folder, f'training_stage_roc_curve'))

    print('Plot Learning Curves .......')
    draw_learning_curves(history,train_subjects)
    save_metrics(accuracy, loss, precision, recall, subject="train_subjects", y_pred_classes=y_pred_classes,
                 y_true_classes=y_true, stage="training")

    return model


def train_svm(X, y, dataset_conf):
    """
    Train an SVM model on the provided dataset.

    Parameters:
    ----------
    X : np.array
        The input EEG data.
    y : np.array
        The labels for the EEG data.
    dataset_conf : dict
        Dictionary containing dataset configuration variables.

    Returns:
    -------
    model : sklearn.svm.SVC
        The trained SVM model.
    """
    # Flattening the input data for SVM as it requires 2D input (samples, features)
    # Convert y from one-hot encoding to class labels
    y = np.argmax(y, axis=1)  # y_labels.shape will be (4574,)
    # Squeeze X to remove the singleton dimension
    X = np.squeeze(X, axis=1)  # X_squeezed.shape will be (4574, 128, 512)
    # Calculate the mean across the last two dimensions (128, 512)
    X = np.mean(X, axis=2)  # X_mean.shape will be (4574,128)
    # Now X_mean and y are ready to be used for SVM training
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=dataset_conf['train_to_val_percentage'],
                                                      random_state=42)

    print("Training set class distribution:", Counter(y_train))
    print("Validation set class distribution:", Counter(y_val))

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred_classes = model.predict(X_val)

    # Call the separate ROC plotting function
    y_probs = y_pred_classes # Assumes binary classification with class 1 probabilities
    plot_and_save_roc_curve(y_train, y_probs,
                            file_name=os.path.join(results_folder, f'training_stage_roc_curve'))


    accuracy = accuracy_score(y_val, y_pred_classes)
    precision = precision_score(y_val, y_pred_classes)
    recall = recall_score(y_val, y_pred_classes)
    loss = custom_hinge_loss(y_val, y_pred_classes)
    print(f'Validation accuracy: {accuracy}')
    print(f'Validation precision: {precision}')
    print(f'Validation recall: {recall}')

    save_metrics(accuracy, None, precision, recall, subject="train_subjects", y_pred_classes=y_pred_classes,
                 y_true_classes=y_val, stage="training")

    return model


def train(subjects, X,y, in_chans, in_samples, tcn_kernel, epochs, batch_size, train_to_val_percentage):
    """
    Trains a deep learning model on EEG data.

    This function defines, compiles, and trains a deep learning model (ATCNet) on the provided EEG data.
    The data is split into training and validation sets, and the model's performance is evaluated after training.
    Training metrics, confusion matrices, and learning curves are saved for later analysis.

    Parameters:
    ----------
    subjects : list
        List of subject identifiers used for training.
    X : np.array
        The input EEG data.
    y : np.array
        The labels for the EEG data.
    in_chans : int
        Number of input channels (electrodes).
    in_samples : int
        Number of time samples per input window.
    tcn_kernel : int
        Size of the kernel for the Temporal Convolutional Network (TCN).

    Returns:
    -------
    None
    """
    global history
    # Define and compile the model with the new window size
    if choose_model == "ATCNet":
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
    elif choose_model == "ShallowConvNet":
        model = models.ShallowConvNet(nb_classes=2, Chans=in_chans, Samples=in_samples)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'
                      , Precision(), Recall()
                           ])

    # Split the data into training and validation sets
    # print(X.shape)
    # print(y.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_to_val_percentage, random_state=42)

    # # Verify the new shapes
    # print("X_train shape:", X_train.shape)
    # print("X_val shape:", X_val.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_val shape:", y_val.shape)

    # Verify the class distribution in the training and validation sets
    print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
    print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping])
    # Evaluate the model
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    # Assuming `model` is your trained model
    # plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

    # Assuming your validation data is X_val and y_val
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted classes
    y_true = np.argmax(y_val, axis=1)           # Get the true classes

    print('Plot Learning Curves ....... ')
    draw_learning_curves(history, subjects)
    save_metrics(accuracy, loss, precision, recall,
                 subject="train_subjects", y_pred_classes=y_pred_classes,
                 y_true_classes=y_true, stage="training")
    # cm = confusion_matrix(y_true, y_pred_classes)
    # print("Confusion Matrix:\n", cm)
    # print("Classification Report:\n", classification_report(y_true, y_pred_classes))
    return model


def interpolate_eeg(data, target_length):
    """
    Interpolates EEG data to increase the number of time samples.

    This function interpolates the EEG data by repeating samples and averaging adjacent pairs to
    double the temporal resolution of the data.

    Parameters:
    ----------
    data : np.array
        The EEG data array to be interpolated.
    target_length : int
        The target number of time samples after interpolation.

    Returns:
    -------
    np.array
        The interpolated EEG data.
    """

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


def interpolate_clicks(clicks, target_length):
    """
    Interpolates click data to match the length of the interpolated EEG data.

    This function interpolates click data to ensure that its length matches the interpolated EEG data.
    It doubles the array size by repeating elements.

    Parameters:
    ----------
    clicks : np.array
        The click data array to be interpolated.
    target_length : int
        The target number of samples after interpolation.

    Returns:
    -------
    np.array
        The interpolated click data.
    """

    interpolated_clicks = np.zeros((clicks.shape[0], target_length), dtype=int)
    for i in range(clicks.shape[0]):
        clicks_clicks = clicks[i].repeat(2)
        interpolated_clicks[i] = clicks_clicks
    return interpolated_clicks


def interpolate_data(subject):
    """
    Interpolates both EEG and click data for a specified subject.

    This function loads, interpolates, and saves both EEG and click data for a given subject.
    It ensures that both datasets have the same number of time samples after interpolation.

    Parameters:
    ----------
    subject : str
        Identifier of the subject whose data is being interpolated.

    Returns:
    -------
    None
    """

    global total_sounds
    eeg = np.load(f'ordered_clips_{subject}.npy')
    clicks_array = np.load(f'ordered_response_labels_{subject}.npy')
    # EEFFOUS: eeg[60][40][128][256] clicks_array[40][15360]
    # BIJVZD:  eeg[60][40][128][256] clicks_array[40][15360]

    # Apply interpolation to double the number of time samples
    eeg_interpolated = interpolate_eeg(eeg, dataset_conf["target_samples"])

    clicks_array_interpolated = interpolate_clicks(clicks_array, dataset_conf['target_samples'] * total_sounds)

    eeg = eeg_interpolated
    clicks_array = clicks_array_interpolated
    # Initialize the inputs and labels for the model
    X = []
    y = []
    total_sounds, total_trials, electrode_num, time_samples = eeg.shape
    segment_times = 0
    label_1_count = 0
    label_0_count = 0

    for trial in range(total_trials):
        # Flatten the eeg for the current trial to be [total_samples, electrode_num]
        eeg_flat = eeg[:, trial, :, :].reshape(total_sounds * time_samples, electrode_num)

        # Get the corresponding clicks array for the trial
        clicks = clicks_array[trial]

        # for 60 2-sec sounds sounded in a trial (in total 512*60) step every 2 seconds (512 time samples) and label:
        end = eeg_flat.shape[0]
        step = dataset_conf['in_samples']
        for start_idx in range(0, end, step):
            end_idx = start_idx + dataset_conf['in_samples']
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

    print("segmented ", segment_times, " times")
    print("labeled 1 ", label_1_count, " times")
    print("labeled 0 ", label_0_count, " times")
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    np.save(f'interpolated_ordered_clips_{subject}.npy', X)
    np.save(f'interpolated_ordered_response_labels_{subject}.npy', y)


def SMOTE_data(subject):
    """
    Balances the EEG dataset using Synthetic Minority Over-sampling Technique (SMOTE).

    This function applies SMOTE to balance the class distribution in the EEG dataset.
    It reshapes the EEG data and labels, applies SMOTE to generate synthetic examples
    for the minority class, and saves the resampled dataset.

    Parameters:
    ----------
    subject : str
        Identifier of the subject whose data is being balanced.

    Returns:
    -------
    None
    """
    # EEFFOUS X[2400][1][128][512] y[2400]
    # BIJVZD  X[2400][1][128][512] y[2400]

    X = np.load(f'interpolated_ordered_clips_{subject}.npy')
    y = np.load(f'interpolated_ordered_response_labels_{subject}.npy')
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

    np.save(f"subjects/{subject}/SMOTEed_eeg_data_{subject}.npy", X_resampled)
    np.save(f"subjects/{subject}/SMOTEed_eeg_labels_{subject}.npy", y_resampled)


def model_3d_representation(SMOTE_eeg, SMOTE_labels, sample_index):
    """
    Models a 3D representation of a single EEG sample and highlights time points where a 'click' occurs.

    This function visualizes the EEG data of a single sample in 3D space, where the x-axis represents the time dimension,
    the y-axis represents the electrode index, and the z-axis represents the amplitude of the EEG signal. Each electrode's
    signal over time is plotted as a separate line in 3D space, with 'click' points highlighted according to the
    corresponding time points in the SMOTE_labels array.

    Parameters:
    ----------
    SMOTE_eeg : np.array
        The EEG data array after SMOTE, with shape (4574, 1, 128, 512).
        4574 is the number of samples, 128 is the number of electrodes, and 512 is the time dimension.
    SMOTE_labels : np.array
        One-hot encoded class labels for the EEG data, with shape (4574, 2).
        The first 512 values in SMOTE_labels correspond to time points where clicks occur.
    sample_index : int
        Index of the sample to be visualized.

    Returns:
    -------
    None
    """
    # Extract the single sample's EEG data: shape (128, 512)
    sample_data = SMOTE_eeg[sample_index, 0, :, :]

    # Identify the time points where 'clicks' occur based on SMOTE_labels
    if SMOTE_labels[sample_index][0]: clicked = 'a'
    else: clicked = 'not a'

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each electrode's signal over time
    for i in range(sample_data.shape[0]):
        ax.plot(np.arange(sample_data.shape[1]), [i] * sample_data.shape[1], sample_data[i], label=f'Electrode {i+1}')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Electrode Index')
    ax.set_zlabel('Amplitude')
    ax.set_title(f'3D EEG Representation for Sample {sample_index}, there was {clicked} click')

    plt.savefig(os.path.join(results_folder, f'sample_{sample_index}_subject_BIJVZD.png'))




if __name__ == '__main__':

    # Update paths and model configuration here
    dataset_conf = {
        'n_classes': 2,
        'in_chans': 128,
        'original_samples' : 256,
        'in_samples': 512,
        'tcn_kernel': 4,
        'n_windows': 5,
        'n_subjects': 15,
        'total_sounds': 60,
        'batch_size': 64,
        'epochs': 30,
        'train_to_val_percentage': 0.5
    }
    model_arr = ["ShallowConvNet", "ATCNet", "SVM"]
    choose_model = model_arr[2]
    training = True
    testing = True
    model_brainwaves=False
    results_folder = "."

    subject_list = [
        'HGWLOI', 'GQEVXE', 'HITXMV', 'HNJUPJ' ,'RHQBHE', 'RMAALZ', 'TUZEZT', 'UOBXJO', 'WWDVDF', 'YMKSWS' #, 'ZLIDEI', 'BIJVZD', 'EFFEUS'
        #, 'misssing two matrices NFICHK', 'missing 2 matrices TQZHZT',
    ]
    model_path = 'trained_model.h5'

    do_preprocess = False
    if do_preprocess:
        for subject in subject_list:
            preprocess(subject)
            create_clicks_array(subject)
            interpolate_data(subject)
            SMOTE_data(subject)
            if os.path.exists(f'ordered_clips_{subject}.npy'):
                os.remove(f'ordered_clips_{subject}.npy')
            if os.path.exists(f'ordered_response_labels_{subject}.npy'):
                os.remove(f'ordered_response_labels_{subject}.npy')
            if os.path.exists(f'interpolated_ordered_clips_{subject}.npy'):
                os.remove(f'interpolated_ordered_clips_{subject}.npy')
            if os.path.exists(f'interpolated_ordered_response_labels_{subject}.npy'):
                os.remove(f'interpolated_ordered_response_labels_{subject}.npy')



    # Randomly split the subjects into training and testing sets
    train_subjects = random.sample(subject_list, len(subject_list) - 2)
    test_subjects = [sub for sub in subject_list if sub not in train_subjects]

    # train_subjects = ['BIJVZD']
    # test_subjects = train_subjects

    # Determine the next available run number
    existing_folders = [f for f in os.listdir() if f.startswith('results_run_')]
    if existing_folders:
        # Extract run numbers and find the maximum
        run_numbers = [int(f.split('_')[-1]) for f in existing_folders if f.split('_')[-1].isdigit()]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 1

    if model_brainwaves:
        subject=("BIJVZD")
        SMOTE_labels    = np.load(f'subjects/{subject}/SMOTEed_eeg_labels_{subject}.npy')
        SMOTE_eeg       = np.load(f'subjects/{subject}/SMOTEed_eeg_data_{subject}.npy')
        model_3d_representation(SMOTE_eeg, SMOTE_labels, 0)

    if training or testing:
        # Create the results folder with the next consecutive number
        results_folder = f'results_run_{next_run_number}'
        os.makedirs(results_folder)

        # Save the subject split information in the results folder
        with open(os.path.join(results_folder, 'subject_info.txt'), 'w') as f:
            f.write(f"Training Subjects: {', '.join(train_subjects)}\n")
            f.write(f"Testing Subjects: {', '.join(test_subjects)}\n")
            f.write(f"epochs = {dataset_conf['epochs']}\n")
            f.write(f"batch_size = {dataset_conf['batch_size']}\n")
            f.write(f"validation percentage from train = {dataset_conf['train_to_val_percentage']}\n")
            f.write(f"model = {choose_model}")


    if(training):
        print(f'Training on subjects: {train_subjects}')
        # Load and concatenate training data
        X_train, y_train = [], []
        for subject in train_subjects:
            X = np.load(f'subjects/{subject}/SMOTEed_eeg_data_{subject}.npy')
            y = np.load(f'subjects/{subject}/SMOTEed_eeg_labels_{subject}.npy')
            X_train.append(X)
            y_train.append(y)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        if choose_model == "ATCNet":
            model = train_atcnet(X_train, y_train, dataset_conf)
        elif choose_model == "ShallowConvNet":
            model = train_shallowconvnet(X_train, y_train, dataset_conf)
        elif choose_model == "SVM":
            model = train_svm(X_train, y_train,dataset_conf)
        if choose_model != "SVM":
            model.save(model_path)
            print(f'Model saved to {model_path}')
            plot_model(model, to_file=os.path.join(results_folder, f"model_{choose_model}_structure.png"), show_shapes=True, show_layer_names=True)
        


    if testing:
        if not training:
            model = load_trained_model(model_path)
            plot_model(model, to_file=os.path.join(results_folder, f"model_{choose_model}_structure.png"), show_shapes=True, show_layer_names=True)
        print(f'Testing on subjects: {test_subjects}')
        # Test the model on the remaining subjects
        for subject in test_subjects:
            Z = np.load(f'subjects/{subject}/SMOTEed_eeg_data_{subject}.npy')
            w = np.load(f'subjects/{subject}/SMOTEed_eeg_labels_{subject}.npy')
            print(Z.shape)
            print(w.shape)

            test(model, Z, w, subject)

