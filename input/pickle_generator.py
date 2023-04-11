import pickle
from scipy.io import loadmat

data = loadmat('BIJVZD_clip10.mat')['clipeeg']
with open('clipeeg.pickle', 'wb') as f:
    pickle.dump(data, f)
