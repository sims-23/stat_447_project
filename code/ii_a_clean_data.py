from i_download_data import *


cat_vars = ['posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'user_id',
            'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
            'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'hotel_continent', 'hotel_country',
            'hotel_market', 'hotel_cluster']

'''
@inputs: pandas dataframe
@outputs: pandas dataframe
@purpose: given a dataset, it should clean the data. Deals with NA values and outliers where values should not be 
negative. Gains variables from datetime variables.
'''
def clean_data(df):
    # ensure all categorical variables are coded as factor
    for var in cat_vars:
        df[var] = df[var].astype('category')

    # drop column as approximately 36% of data is nas which is too high
    df.drop('orig_destination_distance', axis=1, inplace=True)
    dates_to_vars(df)

    cols_names_nas = df.columns[df.isna().any()].tolist()
    fill_nas(cols_names_nas, df)

    # Removing rows with negative values for no_days_to_cin and stay_dur
    df = df.loc[df['no_days_to_cin'] >= 0.0, :]
    df = df.reset_index(drop=True)

    df = df.loc[df['stay_dur'] >= 0.0, :]
    df = df.reset_index(drop=True)
    return df


'''
@inputs: pandas dataframe
@outputs: None
@purpose: From the original dataset, takes date_time variables to give important information such as duration of stay, 
number of days from click/ booking to checkin, check in day of month, check in day of week (0 for Monday, 6 for Sunday),
etc.    
'''
def dates_to_vars(df):
    df[['srch_ci', 'srch_co', 'date_time']] = df[['srch_ci', 'srch_co', 'date_time']].apply(pd.to_datetime)
    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).dt.days
    df['no_days_to_cin'] = (df['srch_ci'] - df['date_time']).dt.days

    # For hotel check-in
    # day of month (1 to 31)
    df['Cin_day'] = df["srch_ci"].dt.day
    # 0 for Monday, 6 for Sunday
    df['Cin_day_of_week'] = df["srch_ci"].dt.weekday
    df['Cin_week'] = df["srch_ci"].dt.isocalendar().week
    df['Cin_month'] = df["srch_ci"].dt.month
    df['Cin_year'] = df["srch_ci"].dt.year

    df.drop(['srch_ci', 'srch_co', 'date_time'], axis=1, inplace=True)


'''
@inputs: pandas dataframe
@outputs: None
@purpose: Fill nas with max if categorical variable otherwise fill with median
'''
def fill_nas(cols, df):
    for col in cols:
        if col in cat_vars:
            max_occurrence = df[col].mode()[0]
            df[col].fillna(max_occurrence, inplace=True)
        else:
            median = df[col].median()
            df[col].fillna(median, inplace=True)


train = clean_data(train)
test = clean_data(test)
