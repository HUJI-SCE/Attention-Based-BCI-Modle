import pandas as pd

df = pd.read_pickle('clipeeg.pickle')
print(df['clipeeg'].shape)