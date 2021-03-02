import pandas as pd
import data_utils as du
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv('data/train.csv', nrows=100000)

text = train.isna().sum(axis=0)
du.save_table_fig(text, 'nas_cols')

# determines how many rows contain the max num of nas (3)
print(train[train.isna().sum(axis=1)==3].shape[0])

# imputations for variables
# reason is approximately 36% of data is nas - too high
train.drop('orig_destination_distance', axis=1, inplace=True)

du.dates_to_vars(train)


print(train[['srch_ci', 'srch_co', 'stay_dur']])

print(train.corr()['hotel_cluster'])

# same magnitudes (i.e. abs vals) get same colour
cmap = sns.diverging_palette(240, 240, as_cmap=True)

corr_table = train.corr()
cols_corr = corr_table.columns.size

plt.figure()
fig = sns.heatmap(train.corr()['hotel_cluster'].values.reshape(cols_corr,1),
                  yticklabels=corr_table.columns,  annot=True, linewidths=2, cmap=cmap)
plt.xlabel('hotel_cluster')
du.save_fig('corr')

cols_names_nas = train.columns.where(train.isna().sum(axis=0)>0).dropna().tolist()
du.fill_nas_with_max(cols_names_nas, train)
print(train.isna().sum(axis=0))

