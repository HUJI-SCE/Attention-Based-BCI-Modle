from pathlib import Path
import streamlit as st

import matplotlib.pyplot as plt
import pickle


# LOAD DATA

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# PLOT DATA

def plot_data(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    st.pyplot(fig)


# ALL SAMPLES CHART

def plot_all_samples(data_dir_path):
    """
    ADD CODE HERE
    Total aggregation function - operates on all the samples in the directory, and provides insight about the dataset as a whole.
    """
    pass


# ALL UPLOADED SAMPLES CHART

def plot_all_uploaded_samples(data_file_paths):
    """
    ADD CODE HERE
    Aggregation function - operates on all the uploaded samples, and provides insight about the selected samples.
    """
    pass


# SAMPLE CHART

def plot_sample(data_file_path):
    """
    ADD CODE HERE
    Operator function - operates on an individual sample and displays an interesting property of the sample.
    """
    pass


# DEFAULT DATA
import os

cwd = os.getcwd()
print(cwd)
exit(0)
default_data_path = 'file.txt'  # CHANGE TO PATH OF DEFAULT PICKLE FILE

default_data = load_data(default_data_path)
plot_data(default_data)

# FILE UPLOADER
uploaded_files = st.sidebar.file_uploader("Choose a pickle file", type='pickle', accept_multiple_files=True)
if uploaded_files:

    data_files_paths = [Path(uploaded_file.name).stem + '.pickle' for uploaded_file in uploaded_files]
    plot_all_uploaded_samples(data_files_paths)

    for uploaded_file in uploaded_files:
        st.title(uploaded_file.name)

        data = load_data(uploaded_file)

        plot_data(data)

        file_stem = Path(uploaded_file.name).stem
        data_file_path = file_stem + '.pickle'
        plot_sample(data_file_path)
