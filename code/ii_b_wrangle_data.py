from ii_a_clean_data import *
import numpy as np


# bins variables and also balances variables
def wrangle_data(df):
    no_days_to_cin_bins = np.arange(start=7, step=14, stop=350)
    no_days_to_cin_bins = np.insert(no_days_to_cin_bins, 0, -1)
    no_days_to_cin_bins = np.insert(no_days_to_cin_bins, no_days_to_cin_bins.size, 1000)
    df['no_days_to_cin_bin'] = pd.cut(df['no_days_to_cin'], no_days_to_cin_bins).cat.codes

    df['stay_dur_bin'] = pd.cut(df['stay_dur'], [-np.inf, 1, 2, 3, 4, 5, 6, 7, np.inf]).cat.codes
    df['cnt_bin'] = pd.cut(df['cnt'], [-np.inf, 1, 2, 3, np.inf]).cat.codes

    # transform variables with small counts
    cat_vars_less_than_10 = ['posa_continent', 'user_location_country', 'user_location_region',
                             'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                             'srch_destination_type_id', 'hotel_continent', 'hotel_country',
                             'stay_dur_bin', 'no_days_to_cin_bin', 'site_name']
    cat_vars_less_than_5 = ['user_location_region']

    # 1000 means other, cannot fit models otherwise
    # Other category is for variables less than the 10th or 5th quantile
    for var in cat_vars_less_than_10:
        series = pd.value_counts(df[var])
        mask = (series / series.sum() * 100).lt(10)
        df[var + "_combined_cats"] = np.where(df[var].isin(series[mask].index), '1000', df[var])

    for var in cat_vars_less_than_5:
        series = pd.value_counts(df[var])
        mask = (series / series.sum() * 100).lt(5)
        df[var + "_combined_cats"] = np.where(df[var].isin(series[mask].index), '1000', df[var])

    return df


train = wrangle_data(train)
test = wrangle_data(test)

train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
