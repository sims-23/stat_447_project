from ii_a_clean_data import *
print(train.columns)

no_days_to_cin_bins = np.arange(start=7, step=14, stop=350)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins, 0, -1)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins,  no_days_to_cin_bins.size, 1000)
print(no_days_to_cin_bins)
train['no_days_to_cin_bin'] = pd.cut(train['no_days_to_cin'], no_days_to_cin_bins)

train['stay_dur_bin'] = pd.cut(train['stay_dur'], [-np.inf, 1, 2, 3, 4, 5, 6, 7,np.inf])


#TODO readme or makefile or something to instruct how to run files
