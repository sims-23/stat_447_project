from ii_a_clean_data import train

import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import pandas as pd
import matplotlib.patches as mpatches


def save_fig(fname, verbose=True):
    path = os.path.join('figs', fname)
    plt.savefig(path, bbox_inches='tight')
    if verbose:
        print("Figure saved as '{}'".format(path))


# correlation
cmap = sns.diverging_palette(200, 200, 100, as_cmap=True)
corr_table = train.corr()
cols_corr = corr_table.columns.size

plt.figure()
sns.heatmap(corr_table.loc["hotel_cluster", :].sort_values().values.reshape(cols_corr, 1),
            yticklabels=list(corr_table.loc["hotel_cluster", :].sort_values().index),
            xticklabels=['hotel_cluster'], annot=True, linewidths=0.5, center=0, cmap=cmap, vmin=-0.1, vmax=0.1,
            robust=True)
save_fig('corr_2')

# count plots
# for var in train.columns:
#     plt.figure(figsize=(32, 12))
#     sns.countplot(x=var, data=train, palette='rainbow', orient='h')
#     save_fig("counts/"+var)


# plt.figure(figsize=(52, 12))
# sns.countplot(x='no_days_to_cin', data=train, palette='rainbow', orient='h')
# save_fig("counts/no_days_to_cin")

plt.figure(figsize=(22, 12))
hist1 = sns.histplot(x='no_days_to_cin', data=train, color="#b7c9e2", discrete=None,
                     binwidth=3)
hist1.margins(x=0)
save_fig("counts/no_days_to_cin_hist")

plt.figure(figsize=(32, 12))
ax = sns.countplot(x='hotel_cluster', data=train, palette='rainbow', orient='h')

total = float(len(train))
for patch in ax.patches:
    percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
    x = patch.get_x() + patch.get_width()
    y = patch.get_height()
    ax.annotate(percentage, (x, y), ha='center')
save_fig('counts/hotel_cluster')

# no_days_to_cin
print("ii-b-no-days")
print(train['no_days_to_cin'].describe())

plt.figure()
sns.boxplot(x="hotel_cluster", y="no_days_to_cin", data=train[(train["hotel_cluster"]==39) |(train["hotel_cluster"]==45) |(train["hotel_cluster"]==27) | (train["hotel_cluster"]==74)])
save_fig("pasta")


# hotel_cluster, srch_destination_, no_days_to_cin, user_location_region

# is_package graphs
# is package percentage stacked bar plot
train_is_package_subset = train.loc[:, ["is_package", "hotel_cluster"]]
df_is_package = pd.DataFrame()
for cluster in train_is_package_subset.loc[:, "hotel_cluster"].unique():
    subset_is_package_value = train_is_package_subset.loc[
        train_is_package_subset["hotel_cluster"] == cluster, "is_package"]
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

