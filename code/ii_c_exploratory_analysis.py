import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

train = pd.read_pickle('train.pkl')

# Toggle value to decide if you want to plot graphs
PLOT_GRAPHS = False

'''
@inputs: string for filename, boolean value
@outputs: plot
@purpose: saves figure in figs folder given filename and prints the name of file if verbose is True
'''
def save_fig(fname, verbose=True):
    path = os.path.join('figs', fname)
    plt.savefig(path, bbox_inches='tight')
    if verbose:
        print("Figure saved as '{}'".format(path))


if PLOT_GRAPHS:
    # count plots
    for var in train.columns:
        plt.figure(figsize=(32, 12))
        sns.countplot(x=var, data=train, palette='rainbow', orient='h')
        save_fig("counts/"+var)

    # wider figure needed to properly see plot
    plt.figure(figsize=(52, 12))
    sns.countplot(x='stay_dur', data=train, palette='rainbow', orient='h')
    save_fig("counts/stay_dur")

    # see overall distribution of no_days_to_cin through histogram
    plt.figure(figsize=(22, 12))
    hist1 = sns.histplot(x='no_days_to_cin', data=train, color="#b7c9e2", discrete=None,
                         binwidth=3)
    hist1.margins(x=0)
    save_fig("counts/no_days_to_cin_hist")

    # get count plot with percentages per each category
    plt.figure(figsize=(32, 12))
    ax = sns.countplot(x='hotel_cluster', data=train, palette='rainbow', orient='h')

    total = float(len(train))
    for patch in ax.patches:
        percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
        x = patch.get_x() + patch.get_width()
        y = patch.get_height()
        ax.annotate(percentage, (x, y), ha='center')
    save_fig('counts/hotel_cluster')

    # get boxplots for continuous variables
    plt.figure(figsize=(32, 12))
    sns.boxplot(x="hotel_cluster", y="no_days_to_cin", data=train)
    save_fig("boxplot/no_days_to_cin")

    plt.figure(figsize=(15, 12))
    sns.boxplot(x="hotel_cluster", y="cnt", data=train)
    save_fig("boxplot/cnt")

    plt.figure(figsize=(32, 12))
    sns.boxplot(x="hotel_cluster", y="stay_dur", data=train)
    save_fig("boxplot/stay_dur")


    plt.figure(figsize=(32, 12))
    ax = sns.countplot(x="stay_dur", data=train, palette='rainbow', orient='h')
    for patch in ax.patches:
        count = '{:.1f}'.format(patch.get_height())
        x = patch.get_x() + patch.get_width()
        y = patch.get_height()
        ax.annotate(count, (x, y), ha='right')
    save_fig("counts/stay_dur")

'''
@inputs: list for categorical x variable, list for categorical y variable, string for x's name, string for y's name, and 
integer for digits for rounding
@outputs: dataframe
@purpose: returns one-row  data frame with names of the x and y variables, the number of distinct values Nx and Ny for 
each variable, and the forward and backward associations, tau(x,y) and tau(y,x) for the Goodman and Kruskal tau measure
'''
def gk_tau(x, y, x_name, y_name, dgts=3):
    #  Compute the joint empirical distribution PIij
    n_ij = pd.crosstab(x, y)
    pi_ij = n_ij/n_ij.values.sum()

    #  Compute the marginals
    pi_i_plus = np.sum(pi_ij, axis=1)
    pi_plus_j = np.sum(pi_ij, axis=0)

    # Compute marginal and conditional variations
    vx = 1 - np.sum(pi_i_plus**2)
    vy = 1 - np.sum(pi_plus_j**2)
    xy_term = np.sum(pi_ij**2, axis=1)
    vy_bar_x = 1-np.sum(xy_term/pi_i_plus)
    yx_term = np.sum(pi_ij**2, axis=0)
    vx_bary = 1 - np.sum(yx_term/pi_plus_j)

    #  Compute forward and reverse associations
    tau_xy = (vy-vy_bar_x)/vy
    tau_yx = (vx-vx_bary)/vx

    #  Form summary dataframe and return
    sum_frame = pd.DataFrame({'x_name': [x_name], 'y_name': [y_name], 'Nx': [n_ij.shape[0]], 'Ny': [n_ij.shape[1]],
                              'tau_xy': [round(tau_xy, ndigits=dgts)], 'tau_yx': [round(tau_yx, ndigits=dgts)]})
    return sum_frame

# Compute Goodman and Kruskal's Tau and save results in a table
def create_tau_table(df):
    tau_table = pd.DataFrame({"Variable": df.columns})
    for row in tau_table["Variable"]:
        if row != "user_id" and row != "srch_destination_id":
            for col in df.columns:
                if col != "user_id" and col != "srch_destination_id":
                    if row != col:
                        print(f'row: {row}. col: {col}')
                        tau_table.loc[tau_table["Variable"] == row, col] = gk_tau(df[col],
                                                                                 df[row], col, row).loc[0, "tau_xy"]
                    else:
                        tau_table.loc[tau_table["Variable"] == row, col] = -5
    return tau_table


created_tau_table = create_tau_table(train)
created_tau_table.to_csv("GKtau_table.csv", index=False)

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


#sort_associated_cat_vars(created_tau_table)
