import pandas as pd

data = pd.read_csv('data/train.csv', nrows=200000)
test = pd.read_csv('data/train.csv', skiprows=range(1, 200000),  nrows=50000)

# Check to make sure not overlapping -> it's all good :)
# print(data.iloc[199999,:])
# print(test.iloc[1,:])


def get_df_top_five_clusters(df):
    top_five_most_clusters = df['hotel_cluster'].value_counts()[:5].index.tolist()
    return df[df['hotel_cluster'].isin(top_five_most_clusters)]


data = get_df_top_five_clusters(data)
test = get_df_top_five_clusters(test)

