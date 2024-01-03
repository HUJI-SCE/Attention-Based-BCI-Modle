import pickle
from scipy.io import loadmat
import pandas as pd

# store in pickle file
data = loadmat('BIJVZD_clip10.mat')['clipeeg']
with open('clip_eeg.pickle', 'wb') as f:
    pickle.dump(data, f)

# load pickle file as pandas DataFrame and transpose
transposer = pd.read_pickle('clip_eeg.pickle')['clipeeg'].T

# restore as pickle file
pd.to_pickle(transposer, 'clip_eeg.pickle')
