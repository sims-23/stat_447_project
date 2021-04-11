from ii_b_wrangle_data import data

import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

PLOT_GRAPHS = False


def save_fig(fname, verbose=True):
    path = os.path.join('figs', fname)
    plt.savefig(path, bbox_inches='tight')
    if verbose:
        print("Figure saved as '{}'".format(path))


# correlation
# cmap = sns.diverging_palette(200, 200, 100, as_cmap=True)
# corr_table = data.corr()
# cols_corr = corr_table.columns.size

# plt.figure()
# sns.heatmap(corr_table.loc["hotel_cluster", :].sort_values().values.reshape(cols_corr, 1),
#             yticklabels=list(corr_table.loc["hotel_cluster", :].sort_values().index),
#             xticklabels=['hotel_cluster'], annot=True, linewidths=0.5, center=0, cmap=cmap, vmin=-0.1, vmax=0.1,
#             robust=True)
# save_fig('corr_2')


if PLOT_GRAPHS:

    # count plots
    for var in data.columns:
        plt.figure(figsize=(32, 12))
        sns.countplot(x=var, data=data, palette='rainbow', orient='h')
        save_fig("counts/"+var)

    plt.figure(figsize=(52, 12))
    sns.countplot(x='stay_dur', data=data, palette='rainbow', orient='h')
    save_fig("counts/stay_dur")

    plt.figure(figsize=(22, 12))
    hist1 = sns.histplot(x='no_days_to_cin', data=data, color="#b7c9e2", discrete=None,
                         binwidth=3)
    hist1.margins(x=0)
    save_fig("counts/no_days_to_cin_hist")

    plt.figure(figsize=(32, 12))
    ax = sns.countplot(x='hotel_cluster', data=data, palette='rainbow', orient='h')

    total = float(len(data))
    for patch in ax.patches:
        percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
        x = patch.get_x() + patch.get_width()
        y = patch.get_height()
        ax.annotate(percentage, (x, y), ha='center')
    save_fig('counts/hotel_cluster')

    # no_days_to_cin
    print("ii-b-no-days")
    print(data['no_days_to_cin'].describe())

    plt.figure(figsize=(32, 12))
    sns.boxplot(x="hotel_cluster", y="no_days_to_cin", data=data)
    save_fig("boxplot/no_days_to_cin")

    plt.figure(figsize=(15, 12))
    sns.boxplot(x="hotel_cluster", y="cnt", data=data)
    save_fig("boxplot/cnt")

    # stay_dur
    print(data['stay_dur'].describe())

    plt.figure(figsize=(32, 12))
    ax = sns.countplot(x="stay_dur", data=data, palette='rainbow', orient='h')
    for patch in ax.patches:
        count = '{:.1f}'.format(patch.get_height())
        x = patch.get_x() + patch.get_width()
        y = patch.get_height()
        ax.annotate(count, (x, y), ha='right')
    save_fig("counts/stay_dur")


    plt.figure(figsize=(32, 12))
    sns.boxplot(x="hotel_cluster", y="stay_dur", data=data)
    save_fig("boxplot/stay_dur")

    # hotel_cluster, srch_destination_, no_days_to_cin, user_location_region

    # is_package graphs
    # is package percentage stacked bar plot
    data_is_package_subset = data.loc[:, ["is_package", "hotel_cluster"]]
    df_is_package = pd.DataFrame()
    for cluster in data_is_package_subset.loc[:, "hotel_cluster"].unique():
        subset_is_package_value = data_is_package_subset.loc[
            data_is_package_subset["hotel_cluster"] == cluster, "is_package"]
        is_package_count = subset_is_package_value.where(subset_is_package_value == 1).count()
        row_is_package = pd.DataFrame({
            "hotel_cluster": [cluster],
            "is_package_count": [is_package_count],
            'not_is_package_count': [subset_is_package_value.count() - is_package_count],
            "is_package_proportion": [is_package_count/subset_is_package_value.count()],
            'total percentage': [1],
            "total count": [subset_is_package_value.count()]
        })
        df_is_package = df_is_package.append(row_is_package, ignore_index=True)
    df_is_package = df_is_package.sort_values(by=["hotel_cluster"])
    df_is_package = df_is_package.reset_index(drop=True)
    print(df_is_package["is_package_proportion"].quantile(q=0.9))
    plt.figure(figsize=(32, 12))
    bar1 = sns.barplot(x="hotel_cluster", y='total percentage', data=df_is_package, color='#fffcc4', fill=False)
    bar2 = sns.barplot(x='hotel_cluster', y='is_package_proportion', data=df_is_package, color='#758000')
    top_bar = mpatches.Patch(color='#fffcc4', label='is_package = 0', fill=False)
    bottom_bar = mpatches.Patch(color='#758000', label='is_package = 1')
    plt.legend(handles=[top_bar, bottom_bar])
    save_fig("stacked_barplot_is_package")

    # is_percentage summary graph -
    # based on the above 75% of hotel clusters have less than 30% bookings/clicks generated
    # as a part of a package

    # also only 10% of hotel clusters have more than 47% booking/clicks generated as a part of a package

    # srch_dest_type_id

    plt.figure(figsize=(15, 10))
    ax = sns.countplot(x="srch_destination_type_id", data=data, palette='rainbow', orient='h')
    for patch in ax.patches:
        count = '{:.1f}'.format(patch.get_height())
        x = patch.get_x() + patch.get_width()
        y = patch.get_height()
        ax.annotate(count, (x, y), ha='right')

    save_fig("counts/srch_destination_type_id")

    plt.figure(figsize=(50,10))

    sns.countplot(x ="hotel_cluster", hue = "srch_destination_type_id", data=data)

    save_fig("counts_srch_dest_typ_hotel_clust_3")


def GKtau(x, y, x_name, y_name, dgts = 3):
  #  Compute the joint empirical distribution PIij

  N_ij = pd.crosstab(x, y)
  PI_ij = N_ij/N_ij.values.sum()

  #  Compute the marginals
  PI_iPlus = np.sum(PI_ij, axis=1)
  PI_Plusj = np.sum(PI_ij, axis=0)


  #  Compute marginal and conditional variations

  vx = 1 - np.sum(PI_iPlus**2)
  vy = 1 - np.sum(PI_Plusj**2)
  xy_term = np.sum(PI_ij**2, axis=1)
  vy_barx = 1-np.sum(xy_term/PI_iPlus)
  yx_term = np.sum(PI_ij**2, axis=0)
  vx_bary = 1 - np.sum(yx_term/PI_Plusj)
  #
  #  Compute forward and reverse associations
  tau_xy = (vy-vy_barx)/vy
  tau_yx = (vx-vx_bary)/vx
  #
  #  Form summary dataframe and return

  sum_frame = pd.DataFrame({'x_name': [x_name], 'y_name': [y_name], 'Nx': [N_ij.shape[0]], 'Ny': [N_ij.shape[1]], 'tau_xy': [round(tau_xy, ndigits=dgts)], 'tau_yx': [round(tau_yx, ndigits=dgts)]})
  return sum_frame


# Note that currently for our continuous vars  this would not be userful, unless we bin the variables
# for var in data.columns:
#     print(GKtau(data[var], data['hotel_cluster'], var, 'hotel_cluster'))


# Compute Goodman and Kruskal's Tau and save results in a table
def create_tau_table(df):
    tau_table = pd.DataFrame({"Variable": df.columns})
    for row in tau_table["Variable"]:
        if row != "user_id" and row != "srch_destination_id":
            for col in df.columns:
                if col != "user_id" and col != "srch_destination_id":
                    if row != col:
                        print(f'row: {row}. col: {col}')
                        tau_table.loc[tau_table["Variable"] == row, col] = GKtau(df[col],
                                                                                 df[row], col, row).loc[0, "tau_xy"]
                    else:
                        tau_table.loc[tau_table["Variable"] == row, col] = -5
    return tau_table


# created_tau_table = create_tau_table(data)
# created_tau_table.to_csv("GKtau_table.csv", index=False)

created_tau_table = pd.read_csv("GKtau_table.csv", index_col=0)


if PLOT_GRAPHS:
    cmap = sns.diverging_palette(400, 200, 100, as_cmap=True)
    plt.figure(figsize=(26, 26))
    h = sns.heatmap(created_tau_table,
                    yticklabels=list(created_tau_table.columns), xticklabels=list(created_tau_table.columns),
                    annot=False, linewidths=0.9, center=0, cmap=cmap, vmin=0,
                    vmax=1, robust=True)
    h.set_yticklabels(h.get_yticklabels(), size=18)
    h.set_xticklabels(h.get_xticklabels(), size=18)
    save_fig('cat_vars_association_GKtau')


# Get top 5 associated variables for each of variables:
def sort_associated_cat_vars(tau_t):
    for row in tau_t.index:
        print(f'Displaying results for {row}')
        if len(tau_t.loc[row, :]) > 0:
            print(tau_t.loc[row, :].sort_values(ascending=False)[:5])


# sort_associated_cat_vars(created_tau_table)
