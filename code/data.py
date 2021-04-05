import pandas as pd
import data_utils as du
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
import numpy as np

train = pd.read_csv('data/train.csv', nrows=10000)

text = train.isna().sum(axis=0)
# plt.figure(figsize=[3.8, 4.8])
# sns.heatmap(text.values.reshape(text.shape[0],1), cmap=cm.Blues, yticklabels = text.index, cbar=False, vmin=100,
#             vmax=100, linewidths=2,  annot=True, fmt='g', xticklabels=['na_cols'])
# du.save_fig('na_cols')

# determines how many rows contain the max num of nas (3)
# print(train[train.isna().sum(axis=1)==3].shape[0])

# imputations for variables
# reason is approximately 36% of data is nas - too high
train.drop('orig_destination_distance', axis=1, inplace=True)
du.dates_to_vars(train)


# print(train[['srch_ci', 'srch_co', 'stay_dur']])

# print(train.corr()['hotel_cluster'])

cols_names_nas = train.columns.where(train.isna().sum(axis=0)>0).dropna().tolist()
du.fill_nas_with_max(cols_names_nas, train)
# print(train.isna().sum(axis=0))

# plt.figure()
# sns.countplot(x = 'no_days_to_cin', data=train)
# du.save_fig('no_days_cin')

# plt.figure()
# sns.countplot(x = 'stay_dur', data=train)
# du.save_fig('stay_dur')

# same magnitudes (i.e. abs vals) get same colour

cmap = sns.diverging_palette(200, 200, 100, as_cmap=True)
train['log_no_days_cin'] = np.log(train["no_days_to_cin"])
corr_table = train.corr()
cols_corr = corr_table.columns.size
# plt.figure()
# sns.heatmap(train.corr()['hotel_cluster'].values.reshape(cols_corr, 1), yticklabels=corr_table.columns,
#                   xticklabels= ['hotel_cluster'], annot=True, linewidths=1, center=0, cmap=cmap, vmin=-0.1, vmax=0.1)
# du.save_fig('corr')

# A bit updated correlation graph
# plt.figure()
# sns.heatmap(corr_table.loc["hotel_cluster", :].sort_values().values.reshape(cols_corr, 1),
#             yticklabels=list(corr_table.loc["hotel_cluster", :].sort_values().index),
#             xticklabels=['hotel_cluster'], annot=True, linewidths=0.5, center=0, cmap=cmap, vmin=-0.1, vmax=0.1,
#             robust=True)
# du.save_fig('corr_2')

# I think we just need to use few relatively important variables, and that's it:
# hotel_cluster, srch_destination_type_id, hotel_country,is_package, stay_dur, no_days_to_cin, user_location_region

# Count Plot for hotel_cluster
# plt.figure(figsize=(32, 12))
# sns.countplot(x='hotel_cluster', data=train, palette='rainbow', orient='h')
# du.save_fig('count_graph_hotel_cluster')


# is_package bar graph
train_subset = train.loc[:, ["is_package", "hotel_cluster"]]
df_is_package = pd.DataFrame()
print(train_subset.loc[train_subset["hotel_cluster"] == 3, 'is_package'].unique())
for cluster in train_subset.loc[:, "hotel_cluster"].unique():
    subset_is_package_value = train_subset.loc[train_subset["hotel_cluster"] == cluster, "is_package"].unique()
    # print(subset_is_package_value)
    if len(subset_is_package_value) > 1:
        type_of_is_package = "Mixed is_package values"
    else:
        if subset_is_package_value == 1:
            type_of_is_package = "All is_package values are 1"
        else:
            type_of_is_package = 'All is_package values are 0'
    row_is_package = pd.DataFrame({
        "hotel_cluster": [cluster],
        "type_of_is_package": [type_of_is_package]
    })
    df_is_package = df_is_package.append(row_is_package, ignore_index=True)
df_is_package = df_is_package.sort_values(by=["hotel_cluster"])
plt.figure()
sns.countplot(x='type_of_is_package', data=df_is_package, palette='rainbow')
du.save_fig('count_graph_hotel_cluster_by_is_package')
# train.info()
# plt.figure()
# sns.boxplot(x="hotel_cluster", y="is_package", data=train)
# du.save_fig('boxplot')




# plot all columns countplots
# rows = train.columns.size//3 - 1
# fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(12,18))
# fig.tight_layout()
# i = 0
# j = 0
# for col in train.columns:
#     if j >= 3:
#         j = 0
#         i += 1
#     # avoid to plot by date
#     if (train[col].dtype == np.int64) | (train[col].dtype == np.float64) | (train[col].dtype == np.uint32):
#         sns.countplot(x=col, data=train, ax=axes[i][j])
#         j += 1
#
# du.save_fig('all_count_plots')

