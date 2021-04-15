from ii_a_clean_data import *
import numpy as np


'''
@inputs: pandas dataframe, list of categories, and integer quantile value
@outputs: None
@purpose:  balances variables by creating an others category for counts less than a given quantile, 
other category is given label 1000
'''
def combined_cats_counts_less_than_quantile(df, categories, quantile):
    for var in categories:
        series = pd.value_counts(df[var])
        mask = (series / series.sum() * 100).lt(quantile)
        df[var + "_combined_cats"] = np.where(df[var].isin(series[mask].index), '1000', df[var])


'''
@inputs: pandas dataframe
@outputs: pandas dataframe
@purpose: bins variables that are skewed and also balances variables by creating an others category 
for values less than a certain quantile
'''
def wrangle_data(df):
    # binning skewed variables
    no_days_to_cin_bins = np.arange(start=7, step=14, stop=350)
    no_days_to_cin_bins = np.insert(no_days_to_cin_bins, 0, -1)
    no_days_to_cin_bins = np.insert(no_days_to_cin_bins, no_days_to_cin_bins.size, 1000)
    df['no_days_to_cin_bin'] = pd.cut(df['no_days_to_cin'], no_days_to_cin_bins).cat.codes

    df['stay_dur_bin'] = pd.cut(df['stay_dur'], [-np.inf, 1, 2, 3, 4, 5, 6, 7, np.inf]).cat.codes
    df['cnt_bin'] = pd.cut(df['cnt'], [-np.inf, 1, 2, 3, np.inf]).cat.codes

    # categorical variables where we want to make other category as any values in the 10th quantile
    cat_vars_less_than_10 = ['posa_continent', 'user_location_country', 'user_location_region',
                             'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                             'srch_destination_type_id', 'hotel_continent', 'hotel_country',
                             'stay_dur_bin', 'no_days_to_cin_bin', 'site_name']

    # categorical variables where we want to make other category as any values in the 5th quantile
    cat_vars_less_than_5 = ['user_location_region']

    combined_cats_counts_less_than_quantile(df, cat_vars_less_than_10, 10)
    combined_cats_counts_less_than_quantile(df, cat_vars_less_than_5, 5)
    return df


train = wrangle_data(train)
test = wrangle_data(test)

# pickle objects for models
train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
