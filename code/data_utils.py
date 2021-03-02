import os.path
import pandas as pd

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pandas.plotting import table
import scipy.sparse
from scipy.optimize import approx_fprime

def save_fig(fname, verbose=True):
    path = os.path.join('figs', fname)
    plt.savefig(path, bbox_inches='tight')
    if verbose:
        print("Figure saved as '{}'".format(path))

def save_table_fig(df, fname, verbose=True):
    ax = plt.subplot(frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df, loc='center')
    save_fig(fname, verbose)

def dates_to_vars(df):
    df[['srch_ci', 'srch_co', 'date_time']] = df[['srch_ci', 'srch_co', 'date_time']].apply(pd.to_datetime)
    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).dt.days
    df['no_days_to_checkin'] = (df['srch_ci'] - df['date_time']).dt.days
    # should we do number of weeks to checkin, number of months, number of days?

    # For hotel check-in
    # day of month (1 to 31)
    df['Cin_day'] = df["srch_ci"].dt.day
    # 0 for Monday, 6 for Sunday
    df['Cin_day_of_week'] = df["srch_ci"].dt.weekday
    df['Cin_week'] = df["srch_ci"].dt.isocalendar().week
    df['Cin_month'] = df["srch_ci"].dt.month
    df['Cin_year'] = df["srch_ci"].dt.year

def fill_nas_with_max(cols, df):
    for col in cols:
        max_occurence = df[col].mode()[0]
        df[col].fillna(max_occurence, inplace=True)

#
# def drop_vars(df):





