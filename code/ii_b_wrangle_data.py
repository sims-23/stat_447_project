from ii_a_clean_data import *
print(train.columns)

no_days_to_cin_bins = np.arange(start=7, step=14, stop=350)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins, 0, -1)
no_days_to_cin_bins = np.insert(no_days_to_cin_bins,  no_days_to_cin_bins.size, 1000)
print(no_days_to_cin_bins)
train['no_days_to_cin_bin'] = pd.cut(train['no_days_to_cin'], no_days_to_cin_bins)

train['stay_dur_bin'] = pd.cut(train['stay_dur'], [-np.inf, 1, 2, 3, 4, 5, 6, 7,np.inf])
train['cnt_bin'] = pd.cut(train['cnt'], [-np.inf, 1, 2, 3, np.inf])



# transform variables with small counts


cat_vars_less_than_10 = ['posa_continent', 'user_location_country', 'user_location_region',
                'is_mobile',
                'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                'srch_destination_type_id',
                'is_booking', 'hotel_continent', 'hotel_country', 'hotel_cluster', 'stay_dur_bin', 'no_days_to_cin_bin', 'site_name']
# cat_vars_less_than_1 = ['srch_destination_id', 'site_name', 'hotel_market', 'user_id', 'user_location_city']

for var in cat_vars_less_than_10:
    series = pd.value_counts(train[var])
    mask = (series / series.sum() * 100).lt(10)
    train[var+"_combined_cats"] = np.where(train[var].isin(series[mask].index), 'Other', train[var])
#
# for var in cat_vars_less_than_1:
#     series = pd.value_counts(train[var])
#     mask = (series / series.sum() * 100).lt(0.1)
#     train[var+"_combined_cats"] = np.where(train[var].isin(series[mask].index), 'Other', train[var])

#TODO readme or makefile or something to instruct how to run files
