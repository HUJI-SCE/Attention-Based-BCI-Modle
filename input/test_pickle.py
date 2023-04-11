import pandas as pd
import os
# Construct the path to the input directory and the pickle file
default_data_path = os.path.join(os.getcwd(), 'clip_eeg.pickle')
df = pd.read_pickle(default_data_path)
print(df.shape)