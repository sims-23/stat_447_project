from i_download_data import *
import numpy as np


def clean_data(df):
    #ensure all categorical variables are coded as factor
    cat_vars = ['posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'user_id',
                'is_mobile',
                'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id',
                'srch_destination_type_id',
                'is_booking', 'hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster']
    for var in cat_vars:
        df[var] = df[var].astype('category')

    # reason is approximately 36% of data is nas - too high
    df.drop('orig_destination_distance', axis=1, inplace=True)

    # TODO: include checks for variables
    dates_to_vars(df)
    # TODO: figure out why this is not working!!!! - DONE

    # Removing rows with negative values of no_days_to_cin
    df = df.loc[df['no_days_to_cin'] >= 0.0, :]
    df = df.reset_index(drop=True)

    # Removing rows with negative values of stay_dur
    df = df.loc[df['stay_dur'] >= 0.0, :]
    df = df.reset_index(drop=True)

    # ????
    cols_names_nas = df.columns.where(df.isna().sum(axis=0) > 0).dropna().tolist()
    fill_nas_with_max(cols_names_nas, df)

    return df


# getting useful info from date variables
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


def fill_nas_with_max(cols, df):
    for col in cols:
        max_occurence = df[col].mode()[0]
        df[col].fillna(max_occurence, inplace=True)

data = clean_data(data)
