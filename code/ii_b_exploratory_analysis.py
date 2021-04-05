from ii_a_clean_data import train

import matplotlib.pyplot as plt
import seaborn as sns
import os.path

print(train.columns)

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

plt.figure(figsize=(32, 12))
ax = sns.countplot(x='hotel_cluster', data=train, palette='rainbow', orient='h')

total = float(len(train))
for patch in ax.patches:
    percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
    x = patch.get_x() + patch.get_width()
    y = patch.get_height()
    ax.annotate(percentage, (x, y),ha='center')
save_fig('counts/hotel_cluster')

# no_days_to_cin
print(train['no_days_to_cin'].describe())

plt.figure()
sns.boxplot(x="hotel_cluster", y="no_days_to_cin", data=train[(train["hotel_cluster"]==39) |(train["hotel_cluster"]==45) |(train["hotel_cluster"]==27) | (train["hotel_cluster"]==74)])
save_fig("pasta")
# hotel_cluster, srch_destination_, no_days_to_cin, user_location_region

plt.figure()
