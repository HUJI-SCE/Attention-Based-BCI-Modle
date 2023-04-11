from pathlib import Path
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objs as go
import pickle
import pandas as pd

# LOAD DATA
def load_data(path):
    with open(path, 'rb') as f:
        data = pd.read_pickle(f)
    return data


# PLOT DATA
def plot_data(data):
    # take only 2-dimensional array and create pandas DataFrame from data:
    df = pd.DataFrame(data[0])

    # create a 3D scatter plot with one trace per electrode
    fig = go.Figure()
    for i in range(128):
        fig.add_trace(go.Scatter3d(x=[i] * 256, y=list(range(256)), z=df.iloc[i], mode="lines"))

    # customize the layout of the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Electrode", range=[0, 128]),
            yaxis=dict(title="Sample", range=[0, 256]),
            zaxis=dict(title="Amplitude", range=[df.min().min(), df.max().max()])
        )
    )
    # display the 3D plot using streamlit
    st.plotly_chart(fig)

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
# Construct the path to the input directory and the pickle file
input_dir = os.path.join(os.getcwd(), 'input')
default_data_path = os.path.join(input_dir, 'clip_eeg.pickle')

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
