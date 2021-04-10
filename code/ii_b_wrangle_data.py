from ii_a_clean_data import *

no_days_to_cin_bins = np.arange(start=7, step=14, stop=350)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins, 0, -1)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins,  no_days_to_cin_bins.size, 1000)
data['no_days_to_cin_bin'] = pd.cut(data['no_days_to_cin'], no_days_to_cin_bins).cat.codes

data['stay_dur_bin'] = pd.cut(data['stay_dur'], [-np.inf, 1, 2, 3, 4, 5, 6, 7,np.inf]).cat.codes
data['cnt_bin'] = pd.cut(data['cnt'], [-np.inf, 1, 2, 3, np.inf]).cat.codes



# transform variables with small counts
cat_vars_less_than_10 = ['posa_continent', 'user_location_country', 'user_location_region',
                         'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                         'srch_destination_type_id', 'hotel_continent', 'hotel_country',
                         'stay_dur_bin', 'no_days_to_cin_bin', 'site_name']
cat_vars_less_than_5 = ['user_location_region']

# -4 means other, cannot fit models otherwise
for var in cat_vars_less_than_10:
    series = pd.value_counts(data[var])
    mask = (series / series.sum() * 100).lt(10)
    data[var+"_combined_cats"] = np.where(data[var].isin(series[mask].index), '-4', data[var])
#
for var in cat_vars_less_than_5:
    series = pd.value_counts(data[var])
    mask = (series / series.sum() * 100).lt(5)
    data[var+"_combined_cats"] = np.where(data[var].isin(series[mask].index), '-4', data[var])

#TODO readme or makefile or something to instruct how to run files