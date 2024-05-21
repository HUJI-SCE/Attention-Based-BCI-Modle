import mne
# Path to your .bdf file
file_path = 'BIJVZD_spmusclass.bdf'

# Load the .bdf file
start_crop = 0
stop_crop = 30
raw = mne.io.read_raw_bdf(file_path, preload=True)
raw.crop(start_crop, stop_crop)

'''
I am saving the data into a file with a name specifying the part saved
'raw_partial_30' is seconds 0-30, two numbers separated by an underscore
e.g. 'a-b' would be from second a to second b(a and b may be floats)
'''
raw.save('../raw_partial_30s.fif', overwrite=True)
